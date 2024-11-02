"""Node definitions."""
import os
import ast
import glob
import json
import shlex
import yaml
import importlib
import subprocess
from string import Template
import pandas as pd
import logging
import concurrent.futures
from abc import ABC, abstractmethod
from typing import Dict, Any, Type, Callable, List, Union
from .default_functions import read_sql, llm_filter

log: logging.Logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Node(ABC):
    def __init__(
        self,
        config: Dict[str, Any],
        secrets: Dict[str, str],
        runtime_config_params = None,
        custom_functions: Dict[str, Callable] = None,
    ):
        self.config = config
        self.secrets = secrets
        self.functions = {}
        self.runtime_config_params = runtime_config_params
        if custom_functions:
            self.functions.update(custom_functions)


    @abstractmethod
    def execute(self, input_data: Any) -> Any:
        """
        Execute the node's operation.

        :param input_data: The input data for this node, typically the output from the previous node.
        :return: The result of this node's operation.
        """
        pass

    @abstractmethod
    def validate_config(self) -> bool:
        """
        Validate the node's configuration.

        :return: True if the configuration is valid, False otherwise.
        """
        pass


class CSVNode(Node):
    description = """Node to read a CSV file."""

    def execute(self, input_data: Any) -> pd.DataFrame:
        path = os.path.expanduser(self.config["path"])
        return pd.read_csv(path)

    def validate_config(self) -> bool:
        assert "path" in self.config, "CSVNode must have a 'path' configuration."
        return True
    

class YAMLNode(Node):
    description = """Node to read a YAML file."""

    def execute(self, input_data: Any) -> Dict[str, Any]:
        path = os.path.expanduser(self.config["path"])
        with open(path, "r") as file:
            return yaml.safe_load(file)

    def validate_config(self) -> bool:
        assert "path" in self.config, "YAMLNode must have a 'path' configuration."
        return True


class JSONNode(Node):
    description = """Node to read a JSON file."""

    def execute(self, input_data: Any) -> Dict[str, Any]:
        path = os.path.expanduser(self.config["path"])
        with open(path, "r") as file:
            return json.load(file)

    def validate_config(self) -> bool:
        assert "path" in self.config, "JSONNode must have a 'path' configuration."
        return True


class JSONDataFrameNode(Node):
    description = """Node to read a JSON file into a pandas DataFrame."""

    def execute(self, input_data: Any) -> pd.DataFrame:
        path = os.path.expanduser(self.config["path"])
        return pd.read_json(path, **self.config.get("read_json_kwargs", {}))

    def validate_config(self) -> bool:
        assert "path" in self.config, "JSONDataFrameNode must have a 'path' configuration."
        return True


class FileDirectoryNode(Node):
    description = """Node to read text and markdown files from a directory or list of directories, supporting regex patterns."""

    def execute(self, input_data: Any) -> pd.DataFrame:
        paths = self.config.get("path")
        if isinstance(input_data, dict) and "path" in input_data:
            paths = input_data["path"]
            assert "path" not in self.config, "FileDirectoryNode `path` cannot be in input data if it is in the configuration."
            if isinstance(paths, list):
                assert all(isinstance(path, str) for path in paths), "FileDirectoryNode `path` in input data must be a list of strings."
        if isinstance(paths, str):
            paths = [paths]
        
        paths = [os.path.expanduser(path) for path in paths]

        file_contents = []
        for path in paths:
            for file_path in glob.glob(path, recursive=True):
                if file_path.lower().endswith(('.txt', '.md', '.html')):
                    with open(os.path.expanduser(file_path), 'r', encoding='utf-8') as file:
                        content = file.read()
                        file_contents.append({
                            'file_path': file_path,
                            'content': content,
                            'file_type': os.path.splitext(file_path)[1][1:]  # Get file extension without the dot
                        })
        
        return pd.DataFrame(file_contents)

    def validate_config(self) -> bool:
        if "path" in self.config:
            paths = self.config["path"]
            assert isinstance(paths, str) or isinstance(paths, list), "FileDirectoryNode 'path' must be a string or a list of strings."
        return True


class SQLNode(Node):
    description = """Node to execute a SQL query."""

    def execute(self, input_data: Any) -> pd.DataFrame:
        connection_string = self.secrets["database_url"]
        read_sql_fn = self.functions.get("read_sql", read_sql)
        return read_sql_fn(self.config["query"], connection_string)

    def validate_config(self) -> bool:
        assert (
            "query" in self.config and "database_url" in self.secrets
        ), "SQLNode must have a 'query' configuration and a 'database_url' secret."
        return True


class StringTemplateNode(Node):
    description = """Node to apply a string template to input data. The
    `header`, if given, is placed at the top. The `template` is filled in
    at least once, and potentially many times depending on the input."""

    def execute(self, input_data: Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame]) -> Union[str, List[str]]:
        template = Template(self.config["template"])
        mode = self.config.get("mode", "single")

        return_string = self.config.get("header", "")
        if return_string:
            return_string = return_string + "\n\n"

        if isinstance(input_data, (list, pd.DataFrame)):
            records = input_data
            if isinstance(input_data, pd.DataFrame):
                records = input_data.to_dict(orient="records")

            if mode == "single":  # Single string output.
                for item in records:
                    if isinstance(item, dict):
                        return_string = return_string + template.safe_substitute(item) + "\n\n"
                    elif isinstance(item, str):
                        return_string = return_string + template.safe_substitute(input=item) + "\n\n"
            elif mode == "multiple":  # List of strings output.
                return_items = [
                    template.safe_substitute(item)
                    if isinstance(item, dict)
                    else template.safe_substitute(input=item)
                    for item in records
                ]
                return return_items
        elif isinstance(input_data, dict):
            return_string = return_string + template.safe_substitute(input_data)
        elif isinstance(input_data, str):
            return_string = return_string + template.safe_substitute(input=input_data)
        elif input_data is None:
            return_string = return_string + template.safe_substitute()
    
        return return_string

    def validate_config(self) -> bool:
        assert "template" in self.config, "StringTemplateNode must have a 'template' configuration."
        if "mode" in self.config:
            modes_are_valid = self.config["mode"] in ["single", "multiple"]
            assert modes_are_valid, "StringTemplateNode `mode` must be `single` or `multiple`."
        return True


class LLMNode(Node):
    description = """Node to query an LLM API with a template. Template must
    contain "{input}" in order to substitute something in. If given a string,
    it will fill in the template and return the result. If given a DataFrame
    or list of dicts, it will turn the column / field denoted by `input_field`
    into a list of strings, fill in the template for each string."""

    def _try_to_load_as_json(self, text: str) -> Union[str, dict]:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text

    def execute(
        self, input_data: Union[str, List[str], pd.DataFrame]
    ) -> Union[str, List[str]]:
        llm_call = self.functions["llm_call"]
        num_threads = self.config.get("num_threads", 1)
        assert num_threads >= 1, "Number of threads must be at least 1."
        model = self.config.get("model", "gpt-4o-mini")
        if num_threads > 1:
            log.info(f"Using {num_threads} threads for LLM Node.")

        # If it's a single string, just call the LLM once.
        if isinstance(input_data, str):
            response = llm_call(
                self.config["template"].format(input=input_data),
                model=model
            )
            return self._try_to_load_as_json(response)
        # If it is a single dict, call the LLM once.
        elif isinstance(input_data, dict):
            assert "input_field" in self.config, "LLMNode must have a `input_field` configuration if given a dict."
            response = llm_call(
                self.config["template"].format(input=input_data[self.config["input_field"]]),
                model=model
            )
            return self._try_to_load_as_json(response)
        
        # Turn input_data into a list of strings.
        if isinstance(input_data, pd.DataFrame):
            # Assert that `input_field` is in the configuration.
            assert "input_field" in self.config, "LLMNode must have a `input_field` configuration for DataFrame or list of dicts."
            # Turn it into a list of strings.
            texts = input_data[self.config["input_field"]].tolist()
        elif isinstance(input_data, pd.Series):
            texts = input_data.tolist()
        elif isinstance(input_data, list):
            is_all_strings = all(isinstance(item, str) for item in input_data)
            is_all_dicts = all(isinstance(item, dict) for item in input_data)
            is_all_dataframes = all(isinstance(item, pd.DataFrame) for item in input_data)

            assert is_all_strings or is_all_dicts or is_all_dataframes, "LLMNode input must be a list of strings, dicts, or DataFrames."
            if is_all_strings:
                texts = input_data
            elif is_all_dicts:
                texts = [item[self.config["input_field"]] for item in input_data]
            else:
                assert "input_field" in self.config, "LLMNode must have a `input_field` configuration for list of DataFrames."
                texts = []
                for df in input_data:
                    texts.extend(df[self.config["input_field"]].tolist())

        # Call the LLM in parallel.
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=num_threads
        ) as executor:
            results = list(
                executor.map(
                    lambda text: llm_call(
                        self.config["template"].format(input=text),
                        model=model
                    ),
                    texts,
                )
            )
        results = [self._try_to_load_as_json(result) for result in results]

        # Now reshape the output to be compatible with the input.
        if isinstance(input_data, pd.DataFrame):
            new_df = input_data.copy()
            new_df.loc[:, self.config["output_field"]] = results
            return new_df
        elif isinstance(input_data, pd.Series):
            return results  # Just keep it as a list.
        elif isinstance(input_data, list):
            if is_all_strings:  # Case: list of strings.
                return results
            elif is_all_dicts:  # Case: list of dicts.
                return [
                    {**item, self.config["output_field"]: result}
                    for item, result in zip(input_data, results)
                ]
            else:  # Case: list of DataFrames.
                dfs = []
                for df in input_data:
                    new_df = df.copy()
                    new_df.loc[:, self.config["output_field"]] = results[: len(df)]
                    results = results[len(df) :]
                    dfs.append(df)
                return dfs
        
    def validate_config(self) -> bool:
        assert (
            "template" in self.config
        ), "LLMNode must have a 'template' configuration."
        assert (
            "input" in self.config
        ), "LLMNode must have an 'input' configuration."
        assert "{input}" in self.config["template"], "LLMNode template must contain '{input}'."

        # If `input_field` in the configuration, then there must also be an `output_field`.
        if "input_field" in self.config:
            assert "output_field" in self.config, "LLMNode must have an `output_field` configuration if `field` is present."
        return True


class LLMFilterNode(Node):
    description = """Node to use LLMs to filter a dataframe."""

    def execute(self, input_data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        is_dataframe = isinstance(input_data, pd.DataFrame)
        assert is_dataframe, "LLMFilterNode must have a DataFrame as input."
        model = self.config.get("model", "gpt-4o-mini")
        num_threads = self.config.get("num_threads", 1)
        assert num_threads >= 1, "Number of threads must be at least 1."
        if num_threads > 1:
            log.info(f"Using {num_threads} threads for LLMFilter Node.")

        llm_filter_fn = self.functions.get("llm_filter", llm_filter)
        return llm_filter_fn(
            input_data,
            self.config["column"],
            self.config["prompt"],
            model=model,
            num_threads=num_threads,
        )

    def validate_config(self) -> bool:
        assert (
            "column" in self.config and "prompt" in self.config
        ), "LLMFilterNode must have 'column' and 'prompt' configuration."
        assert (
            "input" in self.config
        ), "LLMFilterNode must have an 'input' configuration."
        assert isinstance(
            self.config["input"], str
        ), "LLMFilter node 'input' configuration must have only one input."
        return True


class FileWriterNode(Node):
    description = """Node to write data to a file."""

    def execute(self, input_data: str) -> str:
        mode = self.config.get("mode", "single")
        if mode == "single":
            path = os.path.expanduser(self.config["path"])
            with open(path, "w") as file:
                file.write(str(input_data))
        elif mode == "multiple":
            for i, data in enumerate(input_data):
                path = os.path.expanduser(self.config["path"].format(i=i))
                with open(path, "w") as file:
                    file.write(str(data))

    def validate_config(self) -> bool:
        assert (
            "path" in self.config
        ), "FileWriterNode must have a 'path' configuration."
        if "mode" in self.config:
            modes_are_allowed = self.config["mode"] in ["single", "multiple"]
            assert modes_are_allowed, "FileWriterNode `mode` must be `single` or `multiple`."

            if self.config["mode"] == "multiple":
                assert "{i}" in self.config["path"], "FileWriterNode `path` must contain \"{i}\" to format for `multiple` mode."
        return True


class DataFrameOperationNode(Node):
    description = """Node to apply a DataFrame operation to the input data."""

    def execute(
        self, input_data: Union[pd.DataFrame, List]
    ) -> pd.DataFrame:
        if not isinstance(input_data, list):
            input_data = [input_data]
        return self._apply_operation(input_data)
        
    def __handle_dataframe_method(self, df: pd.DataFrame, operation: str, *args, **kwargs) -> pd.DataFrame:
        """Handle DataFrame methods."""
        if hasattr(df, operation):
            method = getattr(df, operation)
            if callable(method):
                try:
                    return method(*args, **kwargs)
                except TypeError:
                    raise TypeError(f"Error calling method {operation} with args {args} and kwargs {kwargs}.")
            else:
                return method
        else:
            raise ValueError(f"Unknown DataFrame operation: {operation}")

    def _apply_operation(self, input_array: List) -> pd.DataFrame:
        operation = self.config["operation"]
        args = self.config.get("args", [])
        kwargs = self.config.get("kwargs", {})

        every_element_is_dataframe = all(isinstance(df, pd.DataFrame) for df in input_array)

        if operation == "concat":
            assert every_element_is_dataframe, "All inputs must be DataFrames."
            return pd.concat(input_array, **kwargs)
        elif operation == "merge":
            assert len(input_array) == 2, "Merge operation requires exactly two DataFrames."
            assert every_element_is_dataframe, "All inputs must be DataFrames."
            return pd.merge(input_array[0], input_array[1], **kwargs)
        elif operation == "set_column":
            # Assert that one input is a DataFrame and the other is a string, list, or Series.
            assert len(input_array) == 2, "`set_column` operation requires exactly two inputs."
            assert isinstance(input_array[0], pd.DataFrame), "First input must be a DataFrame."
            assert isinstance(input_array[1], (str, list, pd.Series)), "Second input must be a string, list, or Series."
            df = input_array[0]
            df.loc[:, self.config["column_name"]] = input_array[1]
            return df
        else:
            # Apply the operation to each DataFrame.
            completed = [
                self.__handle_dataframe_method(df, operation, *args, **kwargs)
                for df in input_array
            ]
            if len(completed) == 1:
                return completed[0]
            return completed

    def validate_config(self) -> bool:
        assert (
            "operation" in self.config
        ), "DataFrameOperationNode must have an `operation` configuration."
        if self.config["operation"] == "set_column":
            assert (
                "column_name" in self.config
            ), "DataFrameOperationNode with `set_column` operation must have a `column_name` configuration."
        return True


class PythonScriptNode(Node):
    description = """Node to execute a Python script using subprocess."""

    def execute(self, input_data: Any) -> Any:
        if input_data is not None:
            if isinstance(input_data, pd.Series):
                input_data = input_data.tolist()
            elif isinstance(input_data, (str, int, float, complex, bool)):
                input_data = [input_data]
            elif isinstance(input_data, list):
                input_data = input_data
            else:
                raise ValueError("Input data must be a primitive, list, or pandas Series.")
        else:
            # None if no input data, so that it runs once and appends
            # nothing to the command.
            input_data = [None]

        # Prepare the command.
        script_path = shlex.quote(self.config["script_path"])
        args = self.config.get("args", [])
        command = ["python", script_path]
        if args:
            # Add args from YAML config.
            for arg in args:
                if isinstance(arg, dict):
                    for k, v in arg.items():
                        command.extend([f"--{shlex.quote(k)}", shlex.quote(str(v))])
                else:
                    command.append(shlex.quote(str(arg)))

        # Prepare the results list and execute the script.
        results = []
        for item in input_data:
            cmd = command.copy()
            if item is not None:
                cmd.append(shlex.quote(str(item)))

            # Execute the script.
            log.info("Attempting to execute command: " + shlex.join(cmd))
            try:
                subprocess_result = subprocess.run(cmd, shell=False, stdout=subprocess.PIPE)
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Script execution failed: {e.stderr}")
        
            # Handle output.
            output_file_key = "return_output_file"
            return_stdout_key = "return_stdout"
            if output_file_key in self.config:
                assert self.config[output_file_key].endswith(
                    (".csv", ".json")
                ), f"{output_file_key} must be a csv or json file."

                # Read the file and return the result.
                if self.config[output_file_key].endswith(".csv"):
                    result = pd.read_csv(self.config[output_file_key])
                    results.append(result)
                elif self.config[output_file_key].endswith(".json"):
                    with open(self.config[output_file_key], "r") as file:
                        result = json.load(file)
                        results.append(result)
            elif return_stdout_key in self.config and self.config[return_stdout_key]:
                result = subprocess_result.stdout.decode("utf-8").strip()
                try:
                    result = ast.literal_eval(result)
                except ValueError:
                    raise ValueError(f"Could not properly eval the `stdout` of the PythonScriptNode output for command: {cmd}. Please make sure you are printing out well-formed JSON.")
                results.append(result)

        if len(results) == 1:
            return results[0]

        return results

    def validate_config(self) -> bool:
        assert "script_path" in self.config, "PythonScript node must have a `script_path` configuration."
        assert os.path.exists(self.config["script_path"]), "PythonScript node `script_path` does not exist."
        # Assert that only one of output_file or return_stdout is present, but not both.
        has_output_file = "return_output_file" in self.config
        has_return_stdout = "return_stdout" in self.config
        not_both = has_output_file != has_return_stdout
        neither = not has_output_file and not has_return_stdout
        assert not_both or neither, "PythonScript node must have either `output_file` or `return_stdout` configuration, or neither."
        return True


class PythonFunctionNode(Node):
    description = """Node to execute a specific Python function from a module or script. 
    Input handling modes:
    - 'arg': Pass input as first argument (default)
    - 'kwarg': Pass input as a named argument (requires input_kwarg)
    - 'ignore': Don't pass input to function
    """

    def execute(self, input_data: Any) -> Any:
        # Import the module and get the function
        module_path = os.path.expanduser(self.config["module_path"])
        function_name = self.config["function_name"]
        
        # If it's a .py file, load it as a module
        if module_path.endswith('.py'):
            module_name = os.path.splitext(os.path.basename(module_path))[0]
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        else:
            # Otherwise, assume it's a module path (e.g., 'os.path')
            module = importlib.import_module(module_path)
        
        # Get the function from the module
        if not hasattr(module, function_name):
            raise ValueError(f"Function '{function_name}' not found in module '{module_path}'")
        function = getattr(module, function_name)
        
        # Prepare arguments
        args = self.config.get("args", [])
        kwargs = self.config.get("kwargs", {})
        
        # Handle input data if provided
        if input_data is not None:
            mode = self.config.get("mode", "arg")
            if mode == "arg":
                args = [input_data] + args
            elif mode == "kwarg":
                input_kwarg = self.config["input_kwarg"]
                kwargs[input_kwarg] = input_data
            elif mode == "ignore":
                pass
            else:
                raise ValueError(f"Unknown mode: {mode}")
        
        # Execute the function
        return function(*args, **kwargs)

    def validate_config(self) -> bool:
        assert "module_path" in self.config, "PythonFunctionNode must have a 'module_path' configuration."
        assert "function_name" in self.config, "PythonFunctionNode must have a 'function_name' configuration."
        
        if "mode" in self.config:
            valid_modes = ["arg", "kwarg", "ignore"]
            assert self.config["mode"] in valid_modes, \
                f"mode must be one of: {valid_modes}"
            
            if self.config["mode"] == "kwarg":
                assert "input_kwarg" in self.config, \
                    "`input_kwarg` must be specified when mode is `kwarg`"
        
        return True


class DAGNode(Node):
    description = """Node to execute a sub-DAG defined in a YAML file."""

    def execute(self, input_data: Dict[str, Any]) -> Any:
        from .dag import WordcelDAG
        # Ensure that the custom_functions don't override the default
        # functions. Subtract out the ones that do.
        self.functions = {
            key: value
            for key, value in self.functions.items()
            if key not in WordcelDAG.default_functions
        }

        # Construct the runtime_config_params. Check for conflicts between
        # the DAG instantiation, the YAML definition.
        runtime_config_params = {}
        if self.runtime_config_params:  # From DAG instantiation in code.
            runtime_config_params = self.runtime_config_params.copy()
        if "runtime_config_params" in self.config:  # From the user.
            # Check if there are any conflicts and raise an error if so.
            conflicts = set(self.config["runtime_config_params"].keys()).intersection(
                runtime_config_params.keys()
            )
            assert not conflicts, f"Runtime config params conflict between DAG instantiation and YAML definition: {conflicts}."

            # If `from_input_data` is in the runtime_config_params, then
            # we need to get the value from the input_data.
            if "from_input_data" in self.config["runtime_config_params"]:
                input_data_key = self.config["runtime_config_params"]["from_input_data"]
                # If it's not present in the input_data, then assume that the user 
                # wants to pass the entire input_data.
                if input_data_key not in input_data:
                    runtime_config_params[input_data_key] = input_data
                else:
                    runtime_config_params[input_data_key] = input_data[input_data_key]

            runtime_config_params.update(self.config["runtime_config_params"])

        # We do not need to give the custom_backends or custom_nodes to the
        # sub-DAG, as they are already in the registry.
        sub_dag = WordcelDAG(
            yaml_file=os.path.expanduser(self.config["path"]),
            secrets_file=self.config.get("secrets_path"),
            custom_functions=self.functions,
            runtime_config_params=runtime_config_params,
        )
        dag_results = sub_dag.execute(input_data=input_data)

        # The dag_results are a dict. If self.config has param `output_key`,
        # then return the value of that key.
        if self.config.get("output_key"):
            output_keys = self.config["output_key"]
            log.info(f"Returning results for output key in DAGNode: {output_keys}")
            if isinstance(output_keys, list):
                return {key: dag_results[key] for key in output_keys} 
            elif isinstance(output_keys, str):
                return dag_results[output_keys]
            else:
                raise ValueError(f"DAGNOde `output_key` must be a string or a list of strings.")
        return dag_results

    def validate_config(self) -> bool:
        assert "path" in self.config, "DAGNode must have a 'path' configuration."
        return True


NODE_TYPES: Dict[str, Type[Node]] = {
    "csv": CSVNode,
    "yaml": YAMLNode,
    "json": JSONNode,
    "json_dataframe": JSONDataFrameNode,
    "file_directory": FileDirectoryNode,
    "sql": SQLNode,
    "string_template": StringTemplateNode,
    "llm": LLMNode,
    "llm_filter": LLMFilterNode,
    "file_writer": FileWriterNode,
    "dataframe_operation": DataFrameOperationNode,
    "python_script": PythonScriptNode,
    "python_function": PythonFunctionNode,
    "dag": DAGNode,
}


class NodeRegistry:
    _registry = {}

    @classmethod
    def register(cls, node_type: str, node_class: Type[Node]):
        cls._registry[node_type] = node_class

    @classmethod
    def get(cls, node_type: str) -> Type[Node]:
        return cls._registry.get(node_type)

    @classmethod
    def register_default_nodes(cls):
        for node_type, node_class in NODE_TYPES.items():
            cls.register(node_type, node_class)
