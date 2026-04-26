"""DAG definition and node implementations."""

import os
import json
from string import Template
from datetime import datetime
from typing import Dict, Any, Type, Callable, Union, Optional, List
import yaml
import pandas as pd
import networkx as nx
from rich.console import Console
from .nodes import Node, NodeRegistry
from .backends import Backend, BackendRegistry
from .default_functions import read_sql, llm_filter, llm_call
from .executors import ExecutorRegistry
from .schema import WordcelConfig
from ..logging_config import get_logger

log = get_logger("dag.dag")

console = Console()


class DAGConfig:
    """Handles loading, substitution, and validation of DAG configurations."""

    def __init__(
        self,
        dag_definition: Union[str, Dict[str, Any]],
        runtime_config_params: Optional[Dict[str, str]] = None,
    ):
        self.runtime_config_params = runtime_config_params
        self.raw_config = self._load_and_substitute(dag_definition)
        self.validated_config = WordcelConfig.model_validate(self.raw_config)

    def _load_and_substitute(self, dag_definition: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Load the configuration and substitute runtime parameters."""
        if isinstance(dag_definition, str):
            # If we are provided with config params, substitute them in the YAML content.
            if self.runtime_config_params:
                with open(os.path.expanduser(dag_definition), "r") as f:
                    content = f.read()
                dag_definition = Template(content).safe_substitute(self.runtime_config_params)
            
            # Load the substituted or original string
            config = self._load_yaml_or_json(dag_definition)
        else:
            if self.runtime_config_params:
                log.warning(
                    "Runtime config params provided, but DAG definition is a dict. "
                    "Substitution skipped for dictionary configuration."
                )
            config = dag_definition

        return config

    @staticmethod
    def _load_yaml_or_json(content: str) -> Dict[str, Any]:
        """Parse YAML or JSON from a file path or raw string."""
        # Try as a file path first if it looks like one and exists
        if len(content) < 1024 and (content.endswith((".yaml", ".json")) or os.path.exists(os.path.expanduser(content))):
            path = os.path.expanduser(content)
            if os.path.exists(path):
                with open(path, "r") as f:
                    if path.endswith(".json"):
                        return json.load(f)
                    return yaml.safe_load(f)

        # Try parsing as raw content
        try:
            return yaml.safe_load(content)
        except yaml.YAMLError:
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                raise ValueError(f"Could not parse DAG definition as YAML or JSON.")

    @property
    def dag_section(self):
        return self.validated_config.dag

    @property
    def nodes_section(self):
        return self.validated_config.nodes


class NodeFactory:
    """Factory for creating node instances."""

    @staticmethod
    def create_node(
        node_config: Dict[str, Any],
        secrets: Dict[str, str],
        runtime_config_params: Optional[Dict[str, str]] = None,
        custom_functions: Optional[Dict[str, Callable]] = None,
    ) -> Node:
        """Create a node instance from configuration."""
        node_type = node_config.get("type")
        node_class = NodeRegistry.get(node_type)
        if node_class is None:
            raise ValueError(
                f"Unknown node type: `{node_type}`. Ensure the type is defined and registered."
            )

        node = node_class(
            node_config,
            secrets,
            runtime_config_params=runtime_config_params,
            custom_functions=custom_functions,
        )
        node.validate_config()
        return node


def _is_json_serializable(data: Any) -> bool:
    """Check if the data is JSON serializable."""
    try:
        if isinstance(data, (pd.DataFrame, pd.Series)):
            data.to_json(orient="records")
        elif hasattr(data, "to_json"):
            data.to_json()
        elif isinstance(data, dict):
            for value in data.values():
                if not _is_json_serializable(value):
                    return False
        elif isinstance(data, list):
            for item in data:
                if not _is_json_serializable(item):
                    return False
        else:
            json.dumps(data)
        return True
    except (TypeError, OverflowError, ValueError):
        return False


class WordcelDAG:
    """DAG class to define and execute a directed acyclic graph."""

    default_functions: Dict[str, Callable] = {
        "read_sql": read_sql,
        "llm_call": llm_call,
        "llm_filter": llm_filter,
    }

    def __init__(
        self,
        dag_definition: Union[str, Dict[str, Any]] = None,
        secrets: Union[str, Dict[str, Any]] = None,
        runtime_config_params: Optional[Dict[str, str]] = None,
        custom_functions: Optional[Dict[str, Callable]] = None,
        custom_nodes: Optional[Dict[str, Type[Node]]] = None,
        custom_backends: Optional[Dict[str, Type[Backend]]] = None,
    ):
        log.warning("This class is still experimental: use at your own risk.")
        
        # Load and validate configuration
        self.runtime_config_params = runtime_config_params
        self.dag_config = DAGConfig(dag_definition, runtime_config_params)
        self.config = self.dag_config.raw_config # Keep for legacy compatibility
        
        dag_info = self.dag_config.dag_section
        self.name = dag_info.name
        self.backend_config = dag_info.backend.model_dump() if dag_info.backend else {}

        # Executor configuration
        if dag_info.executor:
            self.executor_type = dag_info.executor.type
            self.max_workers = dag_info.executor.max_workers
        
        # Legacy support/overrides
        if dag_info.max_workers is not None:
            self.max_workers = dag_info.max_workers
        if dag_info.enable_parallel is not None:
            self.executor_type = "parallel" if dag_info.enable_parallel else "sequential"

        # Load secrets
        self.secrets = self._load_secrets(secrets)

        # Register components
        self._register_components(custom_functions, custom_nodes, custom_backends)

        # Create the backend
        self.backend = self._create_backend()

        # Create the graph and nodes
        self.graph = self.create_graph()
        self.nodes = self._create_node_instances()

    def _load_secrets(self, secrets: Union[str, Dict[str, Any]]) -> Dict[str, str]:
        if secrets is None:
            return {}
        if isinstance(secrets, dict):
            return secrets
        
        path = os.path.expanduser(secrets)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Secrets file `{secrets}` does not exist.")
        
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def _register_components(self, custom_functions, custom_nodes, custom_backends):
        # Functions
        self.functions = self.default_functions.copy()
        if custom_functions:
            for key in custom_functions:
                if key in self.default_functions:
                    raise ValueError(f"Custom function {key} overrides default function.")
            self.functions.update(custom_functions)

        # Nodes
        NodeRegistry.register_default_nodes()
        if custom_nodes:
            for node_type, node_class in custom_nodes.items():
                NodeRegistry.register(node_type, node_class)

        # Backends
        BackendRegistry.register_default_backends()
        if custom_backends:
            for backend_type, backend_class in custom_backends.items():
                BackendRegistry.register(backend_type, backend_class)

    def _create_backend(self) -> Optional[Backend]:
        if not self.backend_config:
            return None
        
        backend_type = self.backend_config.get("type")
        backend_class = BackendRegistry.get(backend_type)
        if not backend_class:
            raise ValueError(f"Unknown backend: {backend_type}")
        
        return backend_class(self.backend_config)

    def _create_node_instances(self) -> Dict[str, Node]:
        nodes = {}
        for node_id, node_config in self.graph.nodes(data=True):
            try:
                nodes[node_id] = NodeFactory.create_node(
                    node_config,
                    self.secrets,
                    runtime_config_params=self.runtime_config_params,
                    custom_functions=self.functions,
                )
            except Exception as e:
                raise ValueError(f"Error creating node {node_id}: {str(e)}")
        return nodes

    def create_graph(self) -> nx.DiGraph:
        G = nx.DiGraph()
        for node in self.dag_config.nodes_section:
            G.add_node(node["id"], **node)
            inputs = node.get("input")
            if inputs:
                if isinstance(inputs, list):
                    for input_id in inputs:
                        G.add_edge(input_id, node["id"])
                else:
                    G.add_edge(inputs, node["id"])
        return G

    def save_image(self, path: str) -> None:
        """Save an image of the DAG using graphviz."""
        try:
            import graphviz
        except ImportError:
            print("Graphviz is not installed. Please install it with 'pip install graphviz'")
            return

        dot = graphviz.Digraph(self.name)
        dot.attr(rankdir='TB', splines='ortho', ranksep='1.5', nodesep='0.5')
        dot.attr('node', shape='box', style='rounded,filled', fillcolor='lightblue', fontname='Helvetica')
        dot.attr('edge', fontname='Helvetica', fontsize='10')

        for i, generation in enumerate(nx.topological_generations(self.graph)):
            with dot.subgraph() as s:
                s.attr(rank='same')
                for node_id in generation:
                    node = self.nodes[node_id]
                    node_type = node.__class__.__name__
                    label = f"{node_id}\n({node_type})"
                    s.node(node_id, label=label)

        for u, v in self.graph.edges():
            dot.edge(u, v)

        render_path, file_format = os.path.splitext(path)
        file_format = file_format.lstrip('.')
        dot.render(render_path, format=file_format, view=False, cleanup=True)

    def get_node_info(self) -> List[Dict[str, Any]]:
        """Return a list of information about each node in the DAG."""
        info = []
        for node_id in nx.topological_sort(self.graph):
            node = self.nodes[node_id]
            inputs = node.config.get("input")
            if isinstance(inputs, str):
                inputs = [inputs]
            
            info.append({
                "id": node_id,
                "type": node.__class__.__name__,
                "description": node.description,
                "config_keys": list(node.config.keys()),
                "inputs": inputs
            })
        return info

    def get_execution_order(self) -> List[str]:
        """Return the topological sort of the graph."""
        return list(nx.topological_sort(self.graph))

    def dry_run(self) -> bool:
        """Perform a dry run of the DAG to validate the configuration."""
        return True

    def _validate_node_input(self, node_id: str, incoming_input: Any) -> None:
        """Validate the input for a node against its spec."""
        node = self.nodes[node_id]
        spec = node.input_spec
        if not spec:
            return

        expected_type = spec.get("type")
        if expected_type is object or expected_type is None:
            if expected_type is None and incoming_input is not None:
                raise TypeError(f"Node '{node_id}' expected no input, but got {type(incoming_input).__name__}")
            return

        if not isinstance(incoming_input, expected_type):
            raise TypeError(
                f"Node '{node_id}' received an invalid input type. "
                f"Expected {expected_type}, but got {type(incoming_input).__name__}."
            )

    def execute(
        self,
        input_data: Dict[str, Any] = None,
        verbose=False,
        executor_type: str = None,
        console=None,
        **executor_kwargs,
    ) -> Dict[str, Any]:
        exec_type = executor_type or self.executor_type
        executor_args = {"verbose": verbose, "console": console}
        if exec_type == "parallel":
            executor_args["max_workers"] = executor_kwargs.get("max_workers", self.max_workers)
        
        executor_args.update(executor_kwargs)
        executor = ExecutorRegistry.create(exec_type, **executor_args)
        return executor.execute(self, input_data)

    @staticmethod
    def load_yaml(yaml_file: str) -> Dict[str, Any]:
        return DAGConfig._load_yaml_or_json(yaml_file)

    @staticmethod
    def load_secrets(secrets) -> Dict[str, str]:
        # Legacy support for static load_secrets
        temp_dag = WordcelDAG.__new__(WordcelDAG)
        return temp_dag._load_secrets(secrets)
