"""Node definitions."""

import os
import sys
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
from typing import Dict, Any, Type, Callable, List, Union, Optional, Tuple
from .default_functions import read_sql, llm_filter
from ..config import DEFAULT_MODEL
from ..logging_config import get_logger

log = get_logger("dag.nodes")


class Node(ABC):
    """Abstract base class for all DAG nodes.

    Attributes:
        input_spec: Specification for expected input types and description.
        description: Human-readable description of what this node does.
        config: Node configuration from YAML definition.
        secrets: Dictionary of secret values (API keys, connection strings, etc.).
        functions: Dictionary of callable functions available to this node.
        runtime_config_params: Parameters substituted at runtime.
    """

    input_spec: Dict[str, Any] = {
        "type": None,
        "description": "This node does not take any input.",
    }
    description: str = "Base node class."

    def __init__(
        self,
        config: Dict[str, Any],
        secrets: Dict[str, str],
        runtime_config_params: Optional[Dict[str, str]] = None,
        custom_functions: Optional[Dict[str, Callable[..., Any]]] = None,
    ) -> None:
        self.config = config
        self.secrets = secrets
        self.functions: Dict[str, Callable[..., Any]] = {}
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
    """Node to read a CSV file into a pandas DataFrame."""

    description = """Node to read a CSV file."""
    input_spec: Dict[str, Any] = {"type": None, "description": "This node does not take any input."}

    def execute(self, input_data: Any) -> pd.DataFrame:
        path = os.path.expanduser(self.config["path"])
        if not os.path.exists(path) and not path.startswith(("http://", "https://")):
            raise FileNotFoundError(
                f"CSV file not found: '{path}'. "
                f"Please check that the file exists and the path is correct."
            )
        return pd.read_csv(path)

    def validate_config(self) -> bool:
        if "path" not in self.config:
            raise ValueError(
                f"CSVNode requires a 'path' configuration. "
                f"Got config keys: {list(self.config.keys())}"
            )
        return True


class YAMLNode(Node):
    """Node to read a YAML file into a dictionary."""

    description = """Node to read a YAML file."""
    input_spec: Dict[str, Any] = {"type": None, "description": "This node does not take any input."}

    def execute(self, input_data: Any) -> Dict[str, Any]:
        path = os.path.expanduser(self.config["path"])
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"YAML file not found: '{path}'. "
                f"Please check that the file exists and the path is correct."
            )
        with open(path, "r") as file:
            return yaml.safe_load(file)

    def validate_config(self) -> bool:
        if "path" not in self.config:
            raise ValueError(
                f"YAMLNode requires a 'path' configuration. "
                f"Got config keys: {list(self.config.keys())}"
            )
        return True


class JSONNode(Node):
    """Node to read a JSON file into a dictionary."""

    description = """Node to read a JSON file."""
    input_spec: Dict[str, Any] = {"type": None, "description": "This node does not take any input."}

    def execute(self, input_data: Any) -> Dict[str, Any]:
        path = os.path.expanduser(self.config["path"])
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"JSON file not found: '{path}'. "
                f"Please check that the file exists and the path is correct."
            )
        with open(path, "r") as file:
            return json.load(file)

    def validate_config(self) -> bool:
        if "path" not in self.config:
            raise ValueError(
                f"JSONNode requires a 'path' configuration. "
                f"Got config keys: {list(self.config.keys())}"
            )
        return True


class JSONDataFrameNode(Node):
    """Node to read a JSON file into a pandas DataFrame."""

    description = """Node to read a JSON file into a pandas DataFrame."""
    input_spec: Dict[str, Any] = {"type": None, "description": "This node does not take any input."}

    def execute(self, input_data: Any) -> pd.DataFrame:
        path = os.path.expanduser(self.config["path"])
        if not os.path.exists(path) and not path.startswith(("http://", "https://")):
            raise FileNotFoundError(
                f"JSON file not found: '{path}'. "
                f"Please check that the file exists and the path is correct."
            )
        return pd.read_json(path, **self.config.get("read_json_kwargs", {}))

    def validate_config(self) -> bool:
        if "path" not in self.config:
            raise ValueError(
                f"JSONDataFrameNode requires a 'path' configuration. "
                f"Got config keys: {list(self.config.keys())}"
            )
        return True


class FileDirectoryNode(Node):
    """Node to read text and markdown files from a directory or list of directories."""

    description = """Node to read text and markdown files from a directory or list of directories, supporting regex patterns."""
    input_spec: Dict[str, Any] = {
        "type": (dict, type(None), list, pd.DataFrame, pd.Series),
        "description": "Optionally accepts a dictionary with a 'path' key (string or list of strings) to override the path in the config. Other input types are ignored, allowing this node to be used as a dependency.",
    }

    def execute(self, input_data: Any) -> pd.DataFrame:
        paths = self.config.get("path")
        if isinstance(input_data, dict) and "path" in input_data:
            paths = input_data["path"]
            if "path" in self.config:
                raise ValueError(
                    "FileDirectoryNode configuration conflict: 'path' cannot be specified "
                    "in both the config and input data. Remove one of them."
                )
            if isinstance(paths, list) and not all(isinstance(path, str) for path in paths):
                raise TypeError(
                    f"FileDirectoryNode 'path' in input data must be a list of strings. "
                    f"Got types: {[type(p).__name__ for p in paths]}"
                )
        elif input_data is not None and not isinstance(input_data, dict):
            log.info(
                f"FileDirectoryNode received input of type {type(input_data).__name__} but expected a dict. "
                "Ignoring input and using the path from config. This is okay if you are just using the `input` for ordering."
            )

        if isinstance(paths, str):
            paths = [paths]

        paths = [os.path.expanduser(path) for path in paths]

        file_contents: List[Dict[str, str]] = []
        for path in paths:
            for file_path in glob.glob(path, recursive=True):
                if file_path.lower().endswith((".txt", ".md", ".html")):
                    with open(
                        os.path.expanduser(file_path), "r", encoding="utf-8"
                    ) as file:
                        content = file.read()
                        file_contents.append(
                            {
                                "file_path": file_path,
                                "content": content,
                                "file_type": os.path.splitext(file_path)[1][
                                    1:
                                ],  # Get file extension without the dot
                            }
                        )

        return pd.DataFrame(file_contents)

    def validate_config(self) -> bool:
        if "path" in self.config:
            paths = self.config["path"]
            if not isinstance(paths, (str, list)):
                raise TypeError(
                    f"FileDirectoryNode 'path' must be a string or a list of strings. "
                    f"Got type: {type(paths).__name__}"
                )
        return True


class SQLNode(Node):
    """Node to execute a SQL query and return results as a DataFrame."""

    description = """Node to execute a SQL query."""
    input_spec: Dict[str, Any] = {"type": None, "description": "This node does not take any input."}

    def execute(self, input_data: Any) -> pd.DataFrame:
        connection_string = self.secrets["database_url"]
        read_sql_fn = self.functions.get("read_sql", read_sql)
        return read_sql_fn(self.config["query"], connection_string)

    def validate_config(self) -> bool:
        missing = []
        if "query" not in self.config:
            missing.append("'query' in config")
        if "database_url" not in self.secrets:
            missing.append("'database_url' in secrets")
        if missing:
            raise ValueError(
                f"SQLNode is missing required configuration: {', '.join(missing)}. "
                f"Config keys: {list(self.config.keys())}, Secret keys: {list(self.secrets.keys())}"
            )
        return True


class StringTemplateNode(Node):
    """Node to apply a string template to input data."""

    description = """Node to apply a string template to input data. The
    `header`, if given, is placed at the top. The `template` is filled in
    at least once, and potentially many times depending on the input."""
    input_spec: Dict[str, Any] = {
        "type": (dict, list, pd.DataFrame, str, type(None)),
        "description": "Accepts a dictionary for direct template substitution, or a list/DataFrame to iterate over. If a string is provided, it's substituted for '{input}'. Can also be run with no input.",
    }

    def execute(
        self, input_data: Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame, str, None]
    ) -> Union[str, List[str]]:
        template = Template(self.config["template"])
        mode = self.config.get("mode", "single")

        # If multiple inputs are defined in the DAG, map them to a dict by node ID.
        if isinstance(self.config.get("input"), list) and isinstance(input_data, list):
            input_data = dict(zip(self.config["input"], input_data))

        return_string = self.config.get("header", "")
        if return_string:
            return_string = return_string + "\n\n"

        if isinstance(input_data, (list, pd.DataFrame)):
            records: Union[List[Any], pd.DataFrame] = input_data
            if isinstance(input_data, pd.DataFrame):
                records = input_data.to_dict(orient="records")

            if mode == "single":  # Single string output.
                for item in records:
                    if isinstance(item, dict):
                        return_string = (
                            return_string + template.safe_substitute(item) + "\n\n"
                        )
                    elif isinstance(item, str):
                        return_string = (
                            return_string
                            + template.safe_substitute(input=item)
                            + "\n\n"
                        )
            elif mode == "multiple":  # List of strings output.
                return_items: List[str] = [
                    (
                        template.safe_substitute(item)
                        if isinstance(item, dict)
                        else template.safe_substitute(input=item)
                    )
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
        if "template" not in self.config:
            raise ValueError(
                f"StringTemplateNode requires a 'template' configuration. "
                f"Got config keys: {list(self.config.keys())}"
            )
        if "mode" in self.config:
            valid_modes = ["single", "multiple"]
            if self.config["mode"] not in valid_modes:
                raise ValueError(
                    f"StringTemplateNode 'mode' must be one of {valid_modes}. "
                    f"Got: '{self.config['mode']}'"
                )
        return True


class StringConcatNode(Node):
    """Node to concatenate strings from multiple inputs with optional separator and prefix/suffix."""

    description = """Node to concatenate strings from multiple inputs with optional separator and prefix/suffix."""
    input_spec: Dict[str, Any] = {
        "type": (str, list, pd.DataFrame, pd.Series),
        "description": "Accepts a string, a list of strings, or a pandas DataFrame/Series to concatenate.",
    }

    def execute(
        self, input_data: Union[str, List[str], pd.DataFrame, pd.Series]
    ) -> str:
        # Get configuration parameters with defaults
        separator: str = self.config.get("separator", " ")
        prefix: str = self.config.get("prefix", "")
        suffix: str = self.config.get("suffix", "")

        # Handle different input types
        strings: List[str]
        if isinstance(input_data, str):
            strings = [input_data]
        elif isinstance(input_data, list):
            if not all(isinstance(x, str) for x in input_data):
                non_strings = [(i, type(x).__name__) for i, x in enumerate(input_data) if not isinstance(x, str)]
                raise TypeError(
                    f"StringConcatNode requires all elements in input list to be strings. "
                    f"Found non-string elements at indices: {non_strings[:5]}{'...' if len(non_strings) > 5 else ''}"
                )
            strings = input_data
        elif isinstance(input_data, pd.DataFrame):
            if "column" not in self.config:
                raise ValueError(
                    f"StringConcatNode requires 'column' in config when input is a DataFrame. "
                    f"Available columns: {list(input_data.columns)}"
                )
            strings = input_data[self.config["column"]].tolist()
        elif isinstance(input_data, pd.Series):
            strings = input_data.tolist()
        else:
            raise TypeError(
                f"StringConcatNode received unsupported input type: {type(input_data).__name__}. "
                f"Expected: str, list, DataFrame, or Series."
            )

        # Filter out None values and convert all elements to strings
        strings = [str(s) for s in strings if s is not None]

        # Perform the concatenation
        result = prefix + separator.join(strings) + suffix

        return result

    def validate_config(self) -> bool:
        # All config parameters are optional but must be correct type if present
        for param in ["separator", "prefix", "suffix"]:
            if param in self.config and not isinstance(self.config[param], str):
                raise TypeError(
                    f"StringConcatNode '{param}' must be a string. "
                    f"Got type: {type(self.config[param]).__name__}"
                )
        return True


class LLMNode(Node):
    """Node to query an LLM API with a template."""

    description = """Node to query an LLM API with a template. Template must
    contain "{input}" in order to substitute something in. If given a string,
    it will fill in the template and return the result. If given a DataFrame
    or list of dicts, it will turn the column / field denoted by `input_field`
    into a list of strings, fill in the template for each string."""
    input_spec: Dict[str, Any] = {
        "type": (str, list, pd.DataFrame, pd.Series, dict),
        "description": "Accepts a string, list of strings, dict, or a pandas DataFrame/Series. The 'input_field' config is required for structured data (dict, DataFrame).",
    }

    def _try_to_load_as_json(self, text: str) -> Union[str, Dict[str, Any]]:
        """Attempt to parse text as JSON, returning original text if parsing fails.

        Args:
            text: The text to attempt JSON parsing on

        Returns:
            Parsed JSON dict if successful, original text string if not
        """
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text

    def execute(
        self, input_data: Union[str, List[str], pd.DataFrame, pd.Series, Dict[str, Any]]
    ) -> Union[str, List[str], pd.DataFrame, List[Dict[str, Any]]]:
        llm_call = self.functions["llm_call"]
        num_threads: int = self.config.get("num_threads", 1)
        if num_threads < 1:
            raise ValueError(
                f"LLMNode 'num_threads' must be at least 1. Got: {num_threads}"
            )
        model: str = self.config.get("model", DEFAULT_MODEL)
        web_search_options: Optional[Dict[str, Any]] = self.config.get("web_search_options")

        if num_threads > 1:
            log.info(f"Using {num_threads} threads for LLM Node.")

        # Prepare llm_call parameters
        llm_params: Dict[str, Any] = {"model": model}
        if web_search_options:
            llm_params["web_search_options"] = web_search_options

        # If it's a single string, just call the LLM once.
        if isinstance(input_data, str):
            response = llm_call(
                self.config["template"].format(input=input_data), **llm_params
            )
            return self._try_to_load_as_json(response)
        # If it is a single dict, call the LLM once.
        elif isinstance(input_data, dict):
            if "input_field" not in self.config:
                raise ValueError(
                    f"LLMNode requires 'input_field' configuration when input is a dict. "
                    f"Available keys in input: {list(input_data.keys())}"
                )
            response = llm_call(
                self.config["template"].format(
                    input=input_data[self.config["input_field"]]
                ),
                **llm_params,
            )
            return self._try_to_load_as_json(response)

        # Turn input_data into a list of strings.
        texts: List[str]
        is_all_strings = False
        is_all_dicts = False

        if isinstance(input_data, pd.DataFrame):
            if "input_field" not in self.config:
                raise ValueError(
                    f"LLMNode requires 'input_field' configuration for DataFrame input. "
                    f"Available columns: {list(input_data.columns)}"
                )
            texts = input_data[self.config["input_field"]].tolist()
        elif isinstance(input_data, pd.Series):
            texts = input_data.tolist()
        elif isinstance(input_data, list):
            is_all_strings = all(isinstance(item, str) for item in input_data)
            is_all_dicts = all(isinstance(item, dict) for item in input_data)
            is_all_dataframes = all(
                isinstance(item, pd.DataFrame) for item in input_data
            )

            if not (is_all_strings or is_all_dicts or is_all_dataframes):
                types_found = set(type(item).__name__ for item in input_data)
                raise TypeError(
                    f"LLMNode input list must contain all strings, all dicts, or all DataFrames. "
                    f"Found mixed types: {types_found}"
                )
            if is_all_strings:
                texts = input_data
            elif is_all_dicts:
                texts = [item[self.config["input_field"]] for item in input_data]
            else:
                if "input_field" not in self.config:
                    raise ValueError(
                        "LLMNode requires 'input_field' configuration for list of DataFrames."
                    )
                texts = []
                for df in input_data:
                    texts.extend(df[self.config["input_field"]].tolist())
        else:
            raise TypeError(
                f"LLMNode received unexpected input type: {type(input_data).__name__}"
            )

        # Call the LLM in parallel.
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            results = list(
                executor.map(
                    lambda text: llm_call(
                        self.config["template"].format(input=text), **llm_params
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
                dfs: List[pd.DataFrame] = []
                for df in input_data:
                    new_df = df.copy()
                    new_df.loc[:, self.config["output_field"]] = results[: len(df)]
                    results = results[len(df) :]
                    dfs.append(df)
                return dfs

        return results

    def validate_config(self) -> bool:
        missing = []
        if "template" not in self.config:
            missing.append("'template'")
        if "input" not in self.config:
            missing.append("'input'")
        if missing:
            raise ValueError(
                f"LLMNode is missing required configuration: {', '.join(missing)}. "
                f"Got config keys: {list(self.config.keys())}"
            )

        if "{input}" not in self.config["template"]:
            raise ValueError(
                f"LLMNode template must contain '{{input}}' placeholder. "
                f"Template preview: '{self.config['template'][:100]}...'"
            )

        # If `input_field` in the configuration, then there must also be an `output_field`.
        if "input_field" in self.config and "output_field" not in self.config:
            raise ValueError(
                "LLMNode requires 'output_field' configuration when 'input_field' is specified. "
                "The output_field specifies where to store LLM results in the output DataFrame/dict."
            )

        if "web_search_options" in self.config:
            if not isinstance(self.config["web_search_options"], dict):
                raise TypeError(
                    f"LLMNode 'web_search_options' must be a dictionary. "
                    f"Got type: {type(self.config['web_search_options']).__name__}"
                )
        return True


class LLMFilterNode(Node):
    """Node to use LLMs to filter a DataFrame based on a prompt."""

    description = """Node to use LLMs to filter a dataframe."""
    input_spec: Dict[str, Any] = {
        "type": pd.DataFrame,
        "description": "Requires a pandas DataFrame as input.",
    }

    def execute(self, input_data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        model: str = self.config.get("model", DEFAULT_MODEL)
        num_threads: int = self.config.get("num_threads", 1)
        if num_threads < 1:
            raise ValueError(
                f"LLMFilterNode 'num_threads' must be at least 1. Got: {num_threads}"
            )
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
        missing = []
        if "column" not in self.config:
            missing.append("'column'")
        if "prompt" not in self.config:
            missing.append("'prompt'")
        if "input" not in self.config:
            missing.append("'input'")
        if missing:
            raise ValueError(
                f"LLMFilterNode is missing required configuration: {', '.join(missing)}. "
                f"Got config keys: {list(self.config.keys())}"
            )

        if not isinstance(self.config["input"], str):
            raise TypeError(
                f"LLMFilterNode 'input' must be a single string (node ID), not a list. "
                f"Got type: {type(self.config['input']).__name__}"
            )
        return True


class FileWriterNode(Node):
    """Node to write data to a file."""

    description = """Node to write data to a file."""
    input_spec: Dict[str, Any] = {
        "type": (str, list),
        "description": "Accepts a string to write to a single file, or a list of strings to write to multiple files (in 'multiple' mode).",
    }

    def execute(self, input_data: Union[str, List[str]]) -> Optional[str]:
        mode: str = self.config.get("mode", "single")
        if mode == "single":
            path = os.path.expanduser(self.config["path"])
            with open(path, "w") as file:
                file.write(str(input_data))
            return path
        elif mode == "multiple":
            if not isinstance(input_data, list):
                raise TypeError(
                    f"FileWriterNode in 'multiple' mode requires a list input. "
                    f"Got type: {type(input_data).__name__}"
                )
            paths_written: List[str] = []
            for i, data in enumerate(input_data):
                path = os.path.expanduser(self.config["path"].format(i=i))
                with open(path, "w") as file:
                    file.write(str(data))
                paths_written.append(path)
            return None
        return None

    def validate_config(self) -> bool:
        if "path" not in self.config:
            raise ValueError(
                f"FileWriterNode requires a 'path' configuration. "
                f"Got config keys: {list(self.config.keys())}"
            )
        if "mode" in self.config:
            valid_modes = ["single", "multiple"]
            if self.config["mode"] not in valid_modes:
                raise ValueError(
                    f"FileWriterNode 'mode' must be one of {valid_modes}. "
                    f"Got: '{self.config['mode']}'"
                )
            if self.config["mode"] == "multiple" and "{i}" not in self.config["path"]:
                raise ValueError(
                    f"FileWriterNode in 'multiple' mode requires '{{i}}' placeholder in path. "
                    f"Got path: '{self.config['path']}'"
                )
        return True


class DataFrameOperationNode(Node):
    """Node to apply a DataFrame operation to the input data."""

    description = """Node to apply a DataFrame operation to the input data."""
    input_spec: Dict[str, Any] = {
        "type": (pd.DataFrame, list),
        "description": "Accepts a single DataFrame or a list of DataFrames for operations like 'concat' or 'merge'.",
    }

    def execute(self, input_data: Union[pd.DataFrame, List[Any]]) -> Union[pd.DataFrame, List[pd.DataFrame]]:
        if not isinstance(input_data, list):
            input_data = [input_data]
        return self._apply_operation(input_data)

    def _handle_dataframe_method(
        self, df: pd.DataFrame, operation: str, *args: Any, **kwargs: Any
    ) -> Any:
        """Handle DataFrame methods."""
        if hasattr(df, operation):
            method = getattr(df, operation)
            if callable(method):
                try:
                    return method(*args, **kwargs)
                except TypeError as e:
                    raise TypeError(
                        f"Error calling DataFrame method '{operation}' with args={args}, kwargs={kwargs}. "
                        f"Original error: {e}"
                    )
            else:
                return method
        else:
            raise ValueError(
                f"Unknown DataFrame operation: '{operation}'. "
                f"This is not a valid pandas DataFrame method."
            )

    def _apply_operation(self, input_array: List[Any]) -> Union[pd.DataFrame, List[pd.DataFrame]]:
        operation: str = self.config["operation"]
        args: List[Any] = self.config.get("args", [])
        kwargs: Dict[str, Any] = self.config.get("kwargs", {})

        every_element_is_dataframe = all(
            isinstance(df, pd.DataFrame) for df in input_array
        )

        if operation == "concat":
            if not every_element_is_dataframe:
                types_found = [type(x).__name__ for x in input_array]
                raise TypeError(
                    f"DataFrameOperationNode 'concat' requires all inputs to be DataFrames. "
                    f"Got types: {types_found}"
                )
            return pd.concat(input_array, **kwargs)
        elif operation == "merge":
            if len(input_array) != 2:
                raise ValueError(
                    f"DataFrameOperationNode 'merge' requires exactly 2 DataFrames. "
                    f"Got {len(input_array)} inputs."
                )
            if not every_element_is_dataframe:
                types_found = [type(x).__name__ for x in input_array]
                raise TypeError(
                    f"DataFrameOperationNode 'merge' requires all inputs to be DataFrames. "
                    f"Got types: {types_found}"
                )
            return pd.merge(input_array[0], input_array[1], **kwargs)
        elif operation == "set_column":
            if len(input_array) != 2:
                raise ValueError(
                    f"DataFrameOperationNode 'set_column' requires exactly 2 inputs "
                    f"(DataFrame and column values). Got {len(input_array)} inputs."
                )
            if not isinstance(input_array[0], pd.DataFrame):
                raise TypeError(
                    f"DataFrameOperationNode 'set_column' requires first input to be a DataFrame. "
                    f"Got type: {type(input_array[0]).__name__}"
                )
            if not isinstance(input_array[1], (str, list, pd.Series)):
                raise TypeError(
                    f"DataFrameOperationNode 'set_column' requires second input to be a string, list, or Series. "
                    f"Got type: {type(input_array[1]).__name__}"
                )
            df = input_array[0]
            df.loc[:, self.config["column_name"]] = input_array[1]
            return df
        else:
            # Apply the operation to each DataFrame.
            completed = [
                self._handle_dataframe_method(df, operation, *args, **kwargs)
                for df in input_array
            ]
            if len(completed) == 1:
                return completed[0]
            return completed

    def validate_config(self) -> bool:
        if "operation" not in self.config:
            raise ValueError(
                f"DataFrameOperationNode requires an 'operation' configuration. "
                f"Got config keys: {list(self.config.keys())}"
            )
        if self.config["operation"] == "set_column" and "column_name" not in self.config:
            raise ValueError(
                "DataFrameOperationNode with 'set_column' operation requires a 'column_name' configuration."
            )
        return True


class PythonScriptNode(Node):
    """Node to execute a Python script using subprocess."""

    description = """Node to execute a Python script using subprocess."""
    input_spec: Dict[str, Any] = {
        "type": (str, int, float, complex, bool, list, pd.Series, pd.DataFrame, type(None)),
        "description": "Accepts a primitive type, a list, a pandas Series, or a pandas DataFrame to be passed as a command-line argument to the script. Can also be run with no input.",
    }

    def execute(self, input_data: Any) -> Any:
        items_to_process: List[Any]
        if input_data is not None:
            if isinstance(input_data, pd.DataFrame):
                items_to_process = input_data.to_records(index=False).tolist()
            elif isinstance(input_data, pd.Series):
                items_to_process = input_data.tolist()
            elif isinstance(input_data, (str, int, float, complex, bool)):
                items_to_process = [input_data]
            elif isinstance(input_data, list):
                items_to_process = input_data
            else:
                raise TypeError(
                    f"PythonScriptNode received unsupported input type: {type(input_data).__name__}. "
                    f"Expected: primitive, list, pandas Series, or pandas DataFrame."
                )
        else:
            # None if no input data, so that it runs once and appends
            # nothing to the command.
            items_to_process = [None]

        # Prepare the command.
        script_path = shlex.quote(self.config["script_path"])
        args: List[Any] = self.config.get("args", [])
        command: List[str] = ["python", script_path]
        if args:
            # Add args from YAML config.
            for arg in args:
                if isinstance(arg, dict):
                    for k, v in arg.items():
                        command.extend([f"--{shlex.quote(k)}", shlex.quote(str(v))])
                else:
                    command.append(shlex.quote(str(arg)))

        # Prepare the results list and execute the script.
        results: List[Any] = []
        for item in items_to_process:
            cmd = command.copy()
            if item is not None:
                cmd.append(shlex.quote(str(item)))

            # Execute the script.
            log.info("Attempting to execute command: " + shlex.join(cmd))
            try:
                subprocess_result = subprocess.run(
                    cmd, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )
            except subprocess.CalledProcessError as e:
                raise RuntimeError(
                    f"PythonScriptNode failed to execute script '{self.config['script_path']}'. "
                    f"Error: {e.stderr}"
                )

            # Handle output.
            output_file_key = "return_output_file"
            return_stdout_key = "return_stdout"
            if output_file_key in self.config:
                output_file = self.config[output_file_key]
                if not output_file.endswith((".csv", ".json")):
                    raise ValueError(
                        f"PythonScriptNode 'return_output_file' must be a .csv or .json file. "
                        f"Got: '{output_file}'"
                    )

                # Read the file and return the result.
                if output_file.endswith(".csv"):
                    result = pd.read_csv(output_file)
                    results.append(result)
                elif output_file.endswith(".json"):
                    with open(output_file, "r") as file:
                        result = json.load(file)
                        results.append(result)
            elif return_stdout_key in self.config and self.config[return_stdout_key]:
                result = subprocess_result.stdout.decode("utf-8").strip()
                try:
                    result = ast.literal_eval(result)
                except (ValueError, SyntaxError) as e:
                    raise ValueError(
                        f"PythonScriptNode could not parse stdout as Python literal. "
                        f"Command: {shlex.join(cmd)}. "
                        f"stdout: '{result[:200]}...'. "
                        f"Error: {e}"
                    )
                results.append(result)

        if len(results) == 1:
            return results[0]

        return results

    def validate_config(self) -> bool:
        if "script_path" not in self.config:
            raise ValueError(
                f"PythonScriptNode requires a 'script_path' configuration. "
                f"Got config keys: {list(self.config.keys())}"
            )
        if not os.path.exists(self.config["script_path"]):
            raise FileNotFoundError(
                f"PythonScriptNode script not found: '{self.config['script_path']}'. "
                f"Please check the path is correct."
            )
        # Validate that only one of output_file or return_stdout is present, but not both.
        has_output_file = "return_output_file" in self.config
        has_return_stdout = "return_stdout" in self.config
        if has_output_file and has_return_stdout:
            raise ValueError(
                "PythonScriptNode cannot have both 'return_output_file' and 'return_stdout'. "
                "Choose one method to capture output."
            )
        return True


class PythonFunctionNode(Node):
    """Node to execute a specific Python function using a dotted path."""

    description = """Node to execute a specific Python function using a dotted path.
    Supports two modes:
    - 'single': passes the entire input as a single argument
    - 'multiple': iterates through input data (list, Series, DataFrame column) and calls function for each item

    Function path should be in the format: package.module.function_name or local.module.function_name."""
    input_spec: Dict[str, Any] = {
        "type": (str, int, float, complex, bool, list, dict, pd.DataFrame, pd.Series, type(None)),
        "description": "Accepts various data types. The data is passed as an argument to the specified function. Behavior is controlled by 'mode' and 'input_kwarg'.",
    }

    def execute(self, input_data: Any) -> Any:
        # Get the function using the dotted path
        function_path: str = self.config["function_path"]
        mode: str = self.config.get("mode", "single")

        # Split the path into module path and function name
        try:
            module_path, function_name = function_path.rsplit(".", 1)
        except ValueError:
            raise ValueError(
                f"PythonFunctionNode invalid function path: '{function_path}'. "
                f"Must be in format 'package.module.function_name' (e.g., 'json.dumps' or 'mymodule.process_data')."
            )

        # Temporarily add current directory to Python path to support local imports
        cwd = os.getcwd()
        sys.path.insert(0, cwd)

        try:
            # Import the module
            try:
                module = importlib.import_module(module_path)
            except ModuleNotFoundError as e:
                raise ModuleNotFoundError(
                    f"PythonFunctionNode could not import module '{module_path}'. "
                    f"Make sure the module is installed or available in the current directory. "
                    f"Original error: {e}"
                )

            # Get the function from the module
            if not hasattr(module, function_name):
                available = [attr for attr in dir(module) if not attr.startswith("_")]
                raise AttributeError(
                    f"PythonFunctionNode: function '{function_name}' not found in module '{module_path}'. "
                    f"Available attributes: {available[:10]}{'...' if len(available) > 10 else ''}"
                )
            function: Callable[..., Any] = getattr(module, function_name)

            # Prepare arguments
            args: List[Any] = list(self.config.get("args", []))
            kwargs: Dict[str, Any] = dict(self.config.get("kwargs", {}))

            if mode == "single":
                # Handle input data if provided (single mode)
                if input_data is not None:
                    if self.config.get("input_kwarg"):
                        kwargs[self.config["input_kwarg"]] = input_data
                    else:
                        args = [input_data] + args
                return function(*args, **kwargs)

            elif mode == "multiple":
                # Handle different types of input data for multiple mode
                if input_data is None:
                    raise ValueError(
                        "PythonFunctionNode in 'multiple' mode requires non-None input data."
                    )

                # Convert input_data to a list of items to process
                items: List[Any]
                if isinstance(input_data, pd.DataFrame):
                    if "input_field" not in self.config:
                        raise ValueError(
                            f"PythonFunctionNode requires 'input_field' in config when using DataFrame input in 'multiple' mode. "
                            f"Available columns: {list(input_data.columns)}"
                        )
                    items = input_data[self.config["input_field"]].tolist()
                elif isinstance(input_data, pd.Series):
                    items = input_data.tolist()
                elif isinstance(input_data, (list, tuple)):
                    items = list(input_data)
                else:
                    raise TypeError(
                        f"PythonFunctionNode in 'multiple' mode received unsupported input type: {type(input_data).__name__}. "
                        f"Expected: DataFrame, Series, list, or tuple."
                    )

                # Process each item
                results: List[Any] = []
                for item in items:
                    item_args = args.copy()
                    item_kwargs = kwargs.copy()

                    if self.config.get("input_kwarg"):
                        item_kwargs[self.config["input_kwarg"]] = item
                    else:
                        item_args = [item] + item_args

                    results.append(function(*item_args, **item_kwargs))

                # If input was a DataFrame, return results in the same format
                if isinstance(input_data, pd.DataFrame):
                    input_data = input_data.copy()
                    output_column = self.config.get("output_field", "result")
                    input_data[output_column] = results
                    return input_data

                return results

            else:
                raise ValueError(
                    f"PythonFunctionNode 'mode' must be 'single' or 'multiple'. Got: '{mode}'"
                )

        finally:
            # Remove the temporarily added path
            sys.path.remove(cwd)

    def validate_config(self) -> bool:
        if "function_path" not in self.config:
            raise ValueError(
                f"PythonFunctionNode requires a 'function_path' configuration. "
                f"Got config keys: {list(self.config.keys())}"
            )

        if "mode" in self.config:
            valid_modes = ["single", "multiple"]
            if self.config["mode"] not in valid_modes:
                raise ValueError(
                    f"PythonFunctionNode 'mode' must be one of {valid_modes}. "
                    f"Got: '{self.config['mode']}'"
                )

        return True


class DAGNode(Node):
    """Node to execute a sub-DAG defined in a YAML file."""

    description = """Node to execute a sub-DAG defined in a YAML file."""
    input_spec: Dict[str, Any] = {
        "type": (dict, pd.DataFrame, type(None)),
        "description": "Accepts a dictionary where keys are node IDs in the sub-DAG to receive input, or a DataFrame. Can also be run with no input.",
    }

    def execute(self, input_data: Union[Dict[str, Any], pd.DataFrame, None]) -> Any:
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
        runtime_config_params: Dict[str, Any] = {}
        if self.runtime_config_params:  # From DAG instantiation in code.
            runtime_config_params = self.runtime_config_params.copy()
        if "runtime_config_params" in self.config:  # From the user.
            # Check if there are any conflicts and raise an error if so.
            conflicts = set(self.config["runtime_config_params"].keys()).intersection(
                runtime_config_params.keys()
            )
            if conflicts:
                raise ValueError(
                    f"DAGNode runtime_config_params conflict between DAG instantiation and YAML definition. "
                    f"Conflicting keys: {conflicts}"
                )

            # If `from_input_data` is in the runtime_config_params, then
            # we need to get the value from the input_data.
            if "from_input_data" in self.config["runtime_config_params"]:
                input_data_key = self.config["runtime_config_params"]["from_input_data"]
                # If it's not present in the input_data, then assume that the user
                # wants to pass the entire input_data.
                if input_data is None or (isinstance(input_data, dict) and input_data_key not in input_data):
                    runtime_config_params[input_data_key] = input_data
                elif isinstance(input_data, dict):
                    runtime_config_params[input_data_key] = input_data[input_data_key]

            runtime_config_params.update(self.config["runtime_config_params"])

        # Check for `input_nodes` in the configuration.
        # It should be a list of `node_id`s where the input data should be
        # passed to the sub-DAG.
        mapped_input: Union[Dict[str, Any], pd.DataFrame, None]
        if "input_nodes" in self.config:
            input_nodes: List[str] = self.config["input_nodes"]
            mapped_input = {}
            for node_id in input_nodes:
                mapped_input[node_id] = input_data
        else:
            mapped_input = input_data

        # We do not need to give the custom_backends or custom_nodes to the
        # sub-DAG, as they are already in the registry.
        sub_dag = WordcelDAG(
            dag_definition=os.path.expanduser(self.config["path"]),
            secrets=self.config.get("secrets_path"),
            custom_functions=self.functions,
            runtime_config_params=runtime_config_params,
        )
        dag_results = sub_dag.execute(input_data=mapped_input)

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
                raise TypeError(
                    f"DAGNode 'output_key' must be a string or a list of strings. "
                    f"Got type: {type(output_keys).__name__}"
                )
        return dag_results

    def validate_config(self) -> bool:
        if "path" not in self.config:
            raise ValueError(
                f"DAGNode requires a 'path' configuration pointing to the sub-DAG YAML file. "
                f"Got config keys: {list(self.config.keys())}"
            )
        if "input_nodes" in self.config:
            if not isinstance(self.config["input_nodes"], list):
                raise TypeError(
                    f"DAGNode 'input_nodes' must be a list of node IDs. "
                    f"Got type: {type(self.config['input_nodes']).__name__}"
                )
        return True


NODE_TYPES: Dict[str, Type[Node]] = {
    "csv": CSVNode,
    "yaml": YAMLNode,
    "json": JSONNode,
    "json_dataframe": JSONDataFrameNode,
    "file_directory": FileDirectoryNode,
    "sql": SQLNode,
    "string_template": StringTemplateNode,
    "string_concat": StringConcatNode,
    "llm": LLMNode,
    "llm_filter": LLMFilterNode,
    "file_writer": FileWriterNode,
    "dataframe_operation": DataFrameOperationNode,
    "python_script": PythonScriptNode,
    "python_function": PythonFunctionNode,
    "dag": DAGNode,
}


class NodeRegistry:
    """Registry for node types, supporting custom node registration."""

    _registry: Dict[str, Type[Node]] = {}

    @classmethod
    def register(cls, node_type: str, node_class: Type[Node]) -> None:
        """Register a node type with its class."""
        cls._registry[node_type] = node_class

    @classmethod
    def get(cls, node_type: str) -> Optional[Type[Node]]:
        """Get a node class by its type name."""
        return cls._registry.get(node_type)

    @classmethod
    def register_default_nodes(cls) -> None:
        """Register all built-in node types."""
        for node_type, node_class in NODE_TYPES.items():
            cls.register(node_type, node_class)

    @classmethod
    def list_node_types(cls) -> List[str]:
        """List all registered node types."""
        return list(cls._registry.keys())
