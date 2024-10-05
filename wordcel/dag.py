"""DAG definition and node implementations."""

import subprocess
import logging
import yaml
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, Any, Type, Callable, List, Union
from abc import ABC, abstractmethod
from sqlalchemy import create_engine
from .llm_providers import openai_call
import concurrent.futures
from .featurize import apply_io_bound_function

log: logging.Logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


"""Node definitions."""


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

            with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
                results = list(executor.map(lambda text: llm_call(self.config["template"].format(input=text)), texts))

            return results
        elif isinstance(input_data, list):
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
                results = list(executor.map(lambda text: llm_call(self.config["template"].format(input=text)), input_data))
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

        llm_filter = self.functions["llm_filter"]
        return llm_filter(
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


"""Helper functions."""


def read_sql(query: str, connection_string: str) -> pd.DataFrame:
    """Helper function to execute a read-only SQL query."""
    engine = create_engine(connection_string)
    results = pd.read_sql(query, connection_string)
    engine.dispose()
    return results


def llm_filter(df: pd.DataFrame, column: str, prompt: str, num_threads: int = 1) -> pd.DataFrame:
    """Helper function to filter a DataFrame using an LLM yes/no question."""
    if num_threads == 1:
        results = df[column].apply(
            lambda value: openai_call(prompt + "\n\n----\n\n" + value)
        )
    else:
        results = apply_io_bound_function(
            df,
            lambda value: openai_call(prompt + "\n\n----\n\n" + value),
            text_column=column,
            num_threads=num_threads
        )
    return df[results.str.lower() == "yes"]


def create_node(
    node_config: Dict[str, Any],
    secrets: Dict[str, str],
    custom_functions: Dict[str, Callable] = None,
) -> Node:
    node_type = node_config.get("type")
    node_class = NodeRegistry.get(node_type)
    if node_class is None:
        raise ValueError(f"Unknown node type: {node_type}.")

    node = node_class(node_config, secrets, custom_functions=custom_functions)
    node.validate_config()
    return node


"""DAG definition."""


class WordcelDAG:
    """DAG class to define and execute a directed acyclic graph."""

    def __init__(
        self,
        yaml_file: str,
        secrets_file: str = None,
        custom_functions: Dict[str, Callable] = None,
        custom_nodes: Dict[str, Type[Node]] = None,
    ):
        """
        Initialize the DAG from a YAML file.
        @param yaml_file: The path to the YAML file containing the DAG configuration.
        @param secrets_file: The path to the YAML file containing the secrets.
        """
        log.warning("This class is still experimental: use at your own risk.")
        self.config = WordcelDAG.load_yaml(yaml_file)
        self.name = self.config["dag"]["name"]

        self.secrets = {}
        if secrets_file is not None:
            self.secrets = WordcelDAG.load_secrets(secrets_file)

        # Register default nodes.
        NodeRegistry.register_default_nodes()

        # Register custom nodes if provided.
        if custom_nodes:
            for node_type, node_class in custom_nodes.items():
                NodeRegistry.register(node_type, node_class)

        # Register custom functions.
        self.functions = self.default_functions
        if custom_functions:
            # Assert that the custom functions do not override the defaults.
            for key, value in custom_functions.items():
                if key in self.default_functions:
                    raise ValueError(f"Custom function {key} overrides default function.")

            self.functions.update(custom_functions)
        
        self.graph = self.create_graph()
        self.nodes = self.create_nodes()

    @property
    def default_functions(self) -> Dict[str, Callable]:
        return {
            "read_sql": read_sql,
            "llm_call": openai_call,
            "llm_filter": llm_filter,
        }

    @staticmethod
    def load_yaml(yaml_file: str) -> Dict[str, Any]:
        """Load a YAML file."""
        with open(yaml_file, "r") as file:
            return yaml.safe_load(file)

    @staticmethod
    def load_secrets(secrets_file: str) -> Dict[str, str]:
        """Load a secrets file."""
        return WordcelDAG.load_yaml(secrets_file)

    def save_image(self, path: str) -> None:
        """Save an image of the DAG using graph.draw."""
        nx.draw(self.graph, with_labels=True, font_weight="bold")
        plt.savefig(path)
        plt.close

    def create_graph(self) -> nx.DiGraph:
        G = nx.DiGraph()
        for node in self.config["nodes"]:
            G.add_node(node["id"], **node)
            if "input" in node:
                if isinstance(node["input"], list):
                    for input_node in node["input"]:
                        G.add_edge(input_node, node["id"])
                else:
                    G.add_edge(node["input"], node["id"])
        return G

    def create_nodes(self) -> Dict[str, Node]:
        """Create node instances from the graph configuration."""
        nodes = {}
        for node_id, node_config in self.graph.nodes(data=True):
            try:
                nodes[node_id] = create_node(
                    node_config, self.secrets, custom_functions=self.default_functions
                )
            except ValueError as e:
                raise ValueError(f"Error creating node {node_id}: {str(e)}")
        return nodes

    def execute(self) -> Dict[str, Any]:
        """Execute the DAG."""
        # Sort and execute the nodes.
        results = {}
        for node_id in nx.topological_sort(self.graph):
            log.info(f"Executing node `{node_id}` from DAG `{self.name}`.")
            node = self.nodes[node_id]
            input_config = self.graph.nodes[node_id].get("input")
            if isinstance(input_config, list):
                output = [results[input_id] for input_id in input_config]
            else:
                output = results.get(input_config)

            results[node_id] = node.execute(output)
        return results
