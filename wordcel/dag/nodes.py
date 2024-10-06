"""Node definitions."""
import subprocess
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
        custom_functions: Dict[str, Callable] = None,
    ):
        self.config = config
        self.secrets = secrets
        self.functions = {}
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
    """Node to read a CSV file."""

    def execute(self, input_data: Any) -> pd.DataFrame:
        return pd.read_csv(self.config["path"])

    def validate_config(self) -> bool:
        assert "path" in self.config, "CSV node must have a 'path' configuration."
        return True


class SQLNode(Node):
    """Node to execute a SQL query."""

    def execute(self, input_data: Any) -> pd.DataFrame:
        connection_string = f"postgresql://{self.secrets['db_user']}:{self.secrets['db_password']}@{self.secrets['db_host']}/{self.secrets['db_name']}"
        read_sql_fn = self.functions.get("read_sql", read_sql)
        return read_sql_fn(self.config["query"], connection_string)

    def validate_config(self) -> bool:
        assert (
            "query" in self.config and "database_url" in self.secrets
        ), "SQL node must have a 'query' configuration and a 'database_url' secret."
        return True


class LLMNode(Node):
    """Node to query an LLM API with a template. If given a string, it
    will fill in the template and return the result. If given a DataFrame,
    it will turn the `input_column` into a list of strings, fill in the
    template for each string, and return a list of results."""

    def execute(
        self, input_data: Union[str, List[str], pd.DataFrame]
    ) -> Union[str, List[str]]:
        llm_call = self.functions["llm_call"]
        num_threads = self.config.get("num_threads", 1)
        assert num_threads >= 1, "Number of threads must be at least 1."
        if num_threads > 1:
            log.info(f"Using {num_threads} threads for LLM Node.")

        if isinstance(input_data, pd.DataFrame):
            if "input_column" not in self.config:
                raise ValueError(
                    "LLM node must have an 'input_column' configuration for DataFrames."
                )
            texts = input_data[self.config["input_column"]].tolist()
            log.info(f"Using column {self.config['input_column']} as input.")

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=num_threads
            ) as executor:
                results = list(
                    executor.map(
                        lambda text: llm_call(
                            self.config["template"].format(input=text)
                        ),
                        texts,
                    )
                )

            return results
        elif isinstance(input_data, list):
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=num_threads
            ) as executor:
                results = list(
                    executor.map(
                        lambda text: llm_call(
                            self.config["template"].format(input=text)
                        ),
                        input_data,
                    )
                )
            return results
        else:
            return llm_call(self.config["template"].format(input=input_data))

    def validate_config(self) -> bool:
        assert (
            "template" in self.config
        ), "LLM node must have a 'template' configuration."
        return True


class LLMFilterNode(Node):
    """Node to use LLMs to filter a dataframe."""

    def execute(self, input_data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        is_dataframe = isinstance(input_data, pd.DataFrame)
        assert is_dataframe, "LLMFilter node must have a DataFrame as input."
        num_threads = self.config.get("num_threads", 1)
        assert num_threads >= 1, "Number of threads must be at least 1."
        if num_threads > 1:
            log.info(f"Using {num_threads} threads for LLMFilter Node.")

        llm_filter_fn = self.functions.get("llm_filter", llm_filter)
        return llm_filter_fn(
            input_data,
            self.config["column"],
            self.config["prompt"],
            num_threads=num_threads,
        )

    def validate_config(self) -> bool:
        assert (
            "column" in self.config and "prompt" in self.config
        ), "LLMFilter node must have 'column' and 'prompt' configuration."
        assert (
            "input" in self.config
        ), "LLMFilter node must have an 'input' configuration."
        assert isinstance(
            self.config["input"], str
        ), "LLMFilter node 'input' configuration must have only one input."
        return True


class FileWriterNode(Node):
    """Node to write data to a file."""

    def execute(self, input_data: str) -> str:
        with open(self.config["path"], "w") as file:
            file.write(str(input_data))

        return input_data

    def validate_config(self) -> bool:
        assert (
            "path" in self.config
        ), "FileWriter node must have a 'path' configuration."
        return True


class DataFrameOperationNode(Node):
    """Node to apply a DataFrame operation to the input data."""

    def execute(
        self, input_data: Union[pd.DataFrame, List[pd.DataFrame]]
    ) -> pd.DataFrame:
        if not isinstance(input_data, list):
            input_data = [input_data]

        if not all(isinstance(df, pd.DataFrame) for df in input_data):
            raise ValueError("All inputs must be DataFrames")

        return self._apply_operation(input_data)

    def _apply_operation(self, dataframes: List[pd.DataFrame]) -> pd.DataFrame:
        operation = self.config["operation"]
        args = self.config.get("args", [])
        kwargs = self.config.get("kwargs", {})

        if operation in ["concat", "merge"]:
            # For concat and merge, use pd.concat or pd.merge directly
            if operation == "concat":
                return pd.concat(dataframes, *args, **kwargs)
            elif operation == "merge":
                if len(dataframes) != 2:
                    raise ValueError("Merge operation requires exactly two DataFrames")
                return pd.merge(dataframes[0], dataframes[1], *args, **kwargs)
        else:
            # For other operations, apply to the first DataFrame
            df = dataframes[0]
            if hasattr(df, operation):
                method = getattr(df, operation)
                if callable(method):
                    return method(*args, **kwargs)
                else:
                    return method
            else:
                raise ValueError(f"Unknown DataFrame operation: {operation}")

    def validate_config(self) -> bool:
        assert (
            "operation" in self.config
        ), "DataFrameOperation node must have an 'operation' configuration."
        return True


class PythonScriptNode(Node):
    """Node to execute a Python script using subprocess."""

    def execute(self, input_data: Any) -> Any:
        script_path = self.config["script_path"]
        args = self.config.get("args", [])

        # If input_data is a DataFrame, convert it to a CSV string
        if isinstance(input_data, pd.DataFrame):
            input_data = input_data.to_csv(index=False)

        # Prepare the command.
        command = "python " + script_path
        if args:
            command += " " + " ".join(args)

        # Execute the script.
        try:
            result = subprocess.run(command, shell=True)
            return result.stdout
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Script execution failed: {e.stderr}")

        # Simply pass through the input data.
        return input_data

    def validate_config(self) -> bool:
        assert (
            "script_path" in self.config
        ), "PythonScript node must have a 'script_path' configuration."
        return True


class DAGNode(Node):
    """Node to execute a sub-DAG defined in a YAML file."""

    def execute(self, input_data: Any) -> Any:
        from .dag import WordcelDAG
        # Ensure that the custom_functions don't override the default
        # functions. Subtract out the ones that do.
        self.functions = {
            key: value
            for key, value in self.functions.items()
            if key not in WordcelDAG.default_functions
        }

        sub_dag = WordcelDAG(
            yaml_file=self.config["path"],
            secrets_file=self.config.get("secrets_path"),
            custom_functions=self.functions,
        )
        return sub_dag.execute()

    def validate_config(self) -> bool:
        assert "path" in self.config, "DAG node must have a 'path' configuration."
        return True


NODE_TYPES: Dict[str, Type[Node]] = {
    "csv": CSVNode,
    "sql": SQLNode,
    "llm": LLMNode,
    "llm_filter": LLMFilterNode,
    "file_writer": FileWriterNode,
    "dataframe_operation": DataFrameOperationNode,
    "python_script": PythonScriptNode,
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