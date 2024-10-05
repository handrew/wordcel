"""DAG definition and node implementations."""
import logging
import yaml
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, Any, Type, Callable
from ..llm_providers import openai_call
from .nodes import Node, NodeRegistry
from .default_functions import read_sql, llm_filter

log: logging.Logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


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

    default_functions = {
        "read_sql": read_sql,
        "llm_call": openai_call,
        "llm_filter": llm_filter,
    }

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
                    raise ValueError(
                        f"Custom function {key} overrides default function."
                    )

            self.functions.update(custom_functions)

        self.graph = self.create_graph()
        self.nodes = self.create_nodes()

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
