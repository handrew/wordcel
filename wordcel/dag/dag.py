"""DAG definition and node implementations."""
import os
import json
import logging
from string import Template
import yaml
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from rich import print
from typing import Dict, Any, Type, Callable
from .nodes import Node, NodeRegistry
from .backends import Backend, BackendRegistry
from .default_functions import read_sql, llm_filter, llm_call

log: logging.Logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def create_node(
    node_config: Dict[str, Any],
    secrets: Dict[str, str],
    runtime_config_params: Dict[str, str] = None,
    custom_functions: Dict[str, Callable] = None,
) -> Node:
    node_type = node_config.get("type")
    node_class = NodeRegistry.get(node_type)
    if node_class is None:
        raise ValueError(
            f"Unknown node type: `{node_type}`. You likely forgot to define its `type`, have it listed as an input somewhere without first creating the node, or haven't given a custom node."
        )

    node = node_class(
        node_config,
        secrets,
        runtime_config_params=runtime_config_params,
        custom_functions=custom_functions
    )
    node.validate_config()
    return node


def _is_json_serializable(data: Any) -> bool:
    """Check if the data is JSON serializable, or is a DataFrame which
    is JSON serializable."""
    try:
        if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
            data.to_json(orient="records")
        elif hasattr(data, "to_json"):
            data.to_json()
        elif isinstance(data, dict):
            # Check keys for DataFrames.
            for key, value in data.items():
                if isinstance(value, pd.DataFrame):
                    value.to_json(orient="records")
                elif not _is_json_serializable(value):
                    return False
        elif isinstance(data, list):
            is_all_dataframes_or_dicts = all(
                isinstance(item, (pd.DataFrame, dict)) for item in data
            )
            if is_all_dataframes_or_dicts:
                for item in data:
                    if isinstance(item, pd.DataFrame):
                        item.to_json(orient="records")
                    else:
                        json.dumps(item)
        else:
            json.dumps(data)
        return True
    except TypeError:
        return False


"""DAG definition."""


class WordcelDAG:
    """DAG class to define and execute a directed acyclic graph."""

    default_functions = {
        "read_sql": read_sql,
        "llm_call": llm_call,
        "llm_filter": llm_filter,
    }

    def __init__(
        self,
        yaml_file: str,
        secrets_file: str = None,
        runtime_config_params: Dict[str, str] = None,
        custom_functions: Dict[str, Callable] = None,
        custom_nodes: Dict[str, Type[Node]] = None,
        custom_backends: Dict[str, Type[Backend]] = None,
    ):
        """
        Initialize the DAG from a YAML file.
        @param yaml_file: The path to the YAML file containing the DAG
        configuration.
        @param secrets_file: The path to the YAML file containing the
        secrets.
        @param runtime_config_params: A dictionary of parameters to substitute
        in the YAML file at runtime.
        @param custom_functions: A dictionary of custom functions to register.
        @param custom_nodes: A dictionary of custom nodes to register.
        @param custom_backends: A dictionary of custom backends to register.
        """
        log.warning("This class is still experimental: use at your own risk.")

        # First load the configuration and secrets.
        self.runtime_config_params = runtime_config_params
        if self.runtime_config_params:
            # If we are provided with config params, substitute them in the YAML file.
            with open(yaml_file, "r") as f:
                pipeline_content = f.read()
            yaml_file = Template(pipeline_content).safe_substitute(
                self.runtime_config_params
            )
        self.config = WordcelDAG.load_yaml(yaml_file)
        self.name = self.config["dag"]["name"]
        self.backend_config = self.config["dag"].get("backend", {})
        self.secrets = {}
        if secrets_file is not None:
            self.secrets = WordcelDAG.load_secrets(secrets_file)

        # Register functions, nodes, backend.
        self._register_default_and_custom_functions(custom_functions)
        self._register_default_and_custom_nodes(custom_nodes)
        self._register_default_and_custom_backends(custom_backends)

        # Create the backend.
        backend_type = self.backend_config.get("type") if self.backend_config else None
        self.backend = None
        if backend_type:
            assert (
                BackendRegistry.get(backend_type) is not None
            ), f"Unknown backend: {backend_type}"
            self.backend = BackendRegistry.get(backend_type)(self.backend_config)

        # Create the graph and nodes.
        self.graph = self.create_graph()
        self.nodes = self.create_nodes()

    @staticmethod
    def load_yaml(yaml_file: str) -> Dict[str, Any]:
        """Load a YAML (or JSON) file."""
        yaml_file = os.path.expanduser(yaml_file)
        if yaml_file.endswith(".json"):
            with open(yaml_file, "r") as file:
                return json.load(file)
        elif yaml_file.endswith(".yaml"):
            with open(yaml_file, "r") as file:
                return yaml.safe_load(file)
        elif isinstance(yaml_file, str):
            # Attempt to read the string as a yaml file.
            result = yaml.safe_load(yaml_file)
            if not isinstance(result, dict):
                result = json.loads(yaml_file)
                if not isinstance(result, dict):
                    raise ValueError(f"Unknown file type: {yaml_file}")
            return result
        else:
            raise ValueError(f"Unknown file type: {yaml_file}")
    

    @staticmethod
    def load_secrets(secrets_file: str) -> Dict[str, str]:
        """Load a secrets file."""
        return WordcelDAG.load_yaml(secrets_file)

    def _register_default_and_custom_functions(
        self, custom_functions: Dict[str, Callable] = None
    ) -> None:
        """Register custom functions."""
        self.functions = self.default_functions
        if custom_functions:
            # Assert that the custom functions do not override the defaults.
            for key, value in custom_functions.items():
                if key in self.default_functions:
                    raise ValueError(
                        f"Custom function {key} overrides default function."
                    )

            self.functions.update(custom_functions)

    def _register_default_and_custom_nodes(
        self, custom_nodes: Dict[str, Type[Node]] = None
    ) -> None:
        """Register custom nodes."""
        NodeRegistry.register_default_nodes()
        if custom_nodes:
            for node_type, node_class in custom_nodes.items():
                NodeRegistry.register(node_type, node_class)

    def _register_default_and_custom_backends(
        self, custom_backends: Dict[str, Type[Backend]] = None
    ) -> None:
        """Register custom backends."""
        BackendRegistry.register_default_backends()
        if custom_backends:
            for backend_type, backend_class in custom_backends.items():
                BackendRegistry.register(backend_type, backend_class)

    def save_image(self, path: str) -> None:
        """Save an image of the DAG using graph.draw."""
        subset_key = "__wordcel_dag_layer__"
        # `multipartite_layout` expects the layer as a node attribute, so add the
        # numeric layer value as a node attribute
        for layer, node in enumerate(nx.topological_generations(self.graph)):
            for n in node:
                self.graph.nodes[n][subset_key] = layer

        pos = nx.multipartite_layout(self.graph, subset_key=subset_key)
        nx.draw_networkx(self.graph, with_labels=True, pos=pos)
        # Make the image LARGE!
        plt.gcf().set_size_inches(18.5, 10.5)
        plt.savefig(path)
        plt.close()

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
                    node_config,
                    self.secrets,
                    runtime_config_params=self.runtime_config_params,
                    custom_functions=self.default_functions
                )
            except ValueError as e:
                raise ValueError(f"Error creating node {node_id}: {str(e)}")
        return nodes


    def __prepare_incoming_input(self, input_data, results, node_id: str):
        """Prepare the incoming input for a node."""
        incoming_edges = self.graph.nodes[node_id].get("input")
        incoming_input = None
        if input_data and isinstance(input_data, dict) and node_id in input_data:
            # First check if the input data is given at runtime.
            # If so, we don't need to look at the incoming edges.
            assert (
                incoming_edges is None
            ), "Node cannot have both `input` and input data given at runtime."
            incoming_input = input_data[node_id]
        elif isinstance(incoming_edges, list):
            incoming_input = [results[input_id] for input_id in incoming_edges]
        else:
            incoming_input = results.get(incoming_edges)

        return incoming_input
    
    def __check_result_is_json_serializable(self, results, node_id: str):
        if not _is_json_serializable(results[node_id]):
            result_type = type(results[node_id]).__name__
            raise ValueError(
                f"Node `{node_id}` returned type `{result_type}`, which either is not serializable or contains something, like a DataFrame, that is not serializable."
            )

    def execute(self, input_data: Dict[str, Any] = None, verbose=False) -> Dict[str, Any]:
        """Execute the DAG.

        @param input_data: A dictionary of input data for the nodes. The key
        is the node ID that the input data is for.
        """
        # Sort and execute the nodes.
        results = {}

        for node_id in nx.topological_sort(self.graph):
            log.info(f"Executing node `{node_id}` from DAG `{self.name}`.")
            node = self.nodes[node_id]

            # Get the incoming edges and their inputs.
            incoming_input = self.__prepare_incoming_input(input_data, results, node_id)
            
            # Check the cache, if we have a backend.
            if self.backend and self.backend.exists(node_id, incoming_input):
                log.info(f"Loading node `{node_id}` from cache.")
                results[node_id] = self.backend.load(node_id, incoming_input)
            else:
                results[node_id] = node.execute(incoming_input)

                # If the node is not a DAG node, check if the result is JSON serializable.
                is_dag_node = isinstance(node, NodeRegistry.get("dag"))
                if not is_dag_node:
                    self.__check_result_is_json_serializable(results, node_id)

                # Don't save DAG nodes to cache, since they may have their own cache.
                if self.backend and not is_dag_node:
                    log.info(f"Saving node `{node_id}` to cache.")
                    self.backend.save(node_id, incoming_input, results[node_id])

            if verbose:
                print(f"Result for node {node_id}:")
                print(results[node_id])

        return results
