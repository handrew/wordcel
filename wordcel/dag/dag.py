"""DAG definition and node implementations."""

import os
import json
from string import Template
from datetime import datetime
import yaml
import pandas as pd
import networkx as nx
from rich import print
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
)
from typing import Dict, Any, Type, Callable, Union, Optional
from .nodes import Node, NodeRegistry
from .backends import Backend, BackendRegistry
from .default_functions import read_sql, llm_filter, llm_call
from .executors import ExecutorRegistry
from ..logging_config import get_logger

log = get_logger("dag.dag")

console = Console()


def create_node(
    node_config: Dict[str, Any],
    secrets: Dict[str, str],
    runtime_config_params: Optional[Dict[str, str]] = None,
    custom_functions: Optional[Dict[str, Callable]] = None,
) -> Node:
    """Create a node instance from configuration.

    Args:
        node_config: Configuration dictionary for the node
        secrets: Dictionary of secret values
        runtime_config_params: Runtime configuration parameters
        custom_functions: Dictionary of custom functions to inject

    Returns:
        Instantiated and validated node

    Raises:
        ValueError: If node type is unknown or invalid
    """
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
        custom_functions=custom_functions,
    )
    node.validate_config()
    return node


def _is_json_serializable(data: Any) -> bool:
    """Check if the data is JSON serializable, or is a DataFrame which is JSON serializable.

    Args:
        data: The data to check for JSON serializability

    Returns:
        True if data can be serialized to JSON, False otherwise
    """
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


def _assert_dag_param_is_valid(dag_definition):
    """Assert that the DAG definition is valid."""
    assert isinstance(
        dag_definition, (str, dict)
    ), "DAG definition must be a string or dictionary."
    if isinstance(dag_definition, str):
        assert os.path.exists(
            os.path.expanduser(dag_definition)
        ), f"DAG definition file `{dag_definition}` does not exist."
        assert dag_definition.endswith(
            ".yaml"
        ), "DAG definition file must be a YAML file."


class WordcelDAG:
    """DAG class to define and execute a directed acyclic graph."""

    default_functions = {
        "read_sql": read_sql,
        "llm_call": llm_call,
        "llm_filter": llm_filter,
    }

    def __init__(
        self,
        dag_definition: Union[str, Dict[str, Any]] = None,
        secrets: Union[str, Dict[str, Any]] = None,
        runtime_config_params: Dict[str, str] = None,
        custom_functions: Dict[str, Callable] = None,
        custom_nodes: Dict[str, Type[Node]] = None,
        custom_backends: Dict[str, Type[Backend]] = None,
    ):
        """
        Initialize the DAG from a YAML file.
        @param dag_definition: The path to the YAML file containing the DAG
        configuration, or a dict.
        @param secrets: The path to the YAML file containing the
        secrets.
        @param runtime_config_params: A dictionary of parameters to substitute
        in the YAML file at runtime.
        @param custom_functions: A dictionary of custom functions to register.
        @param custom_nodes: A dictionary of custom nodes to register.
        @param custom_backends: A dictionary of custom backends to register.
        """
        log.warning("This class is still experimental: use at your own risk.")
        _assert_dag_param_is_valid(dag_definition)

        # First load the configuration.
        self.runtime_config_params = runtime_config_params
        dag_definition = self.__substitute_runtime_config_params_in_dag_definition(
            dag_definition
        )
        if isinstance(dag_definition, str):
            self.config = WordcelDAG.load_yaml(dag_definition)
        else:
            self.config = dag_definition
        self.name = self.config["dag"]["name"]
        self.backend_config = self.config["dag"].get("backend", {})

        # Executor configuration
        self.executor_config = self.config["dag"].get("executor", {})
        self.executor_type = self.executor_config.get("type", "parallel")
        self.max_workers = self.executor_config.get("max_workers", 4)

        # Legacy support for old configuration format
        if "max_workers" in self.config["dag"]:
            self.max_workers = self.config["dag"]["max_workers"]
        if "enable_parallel" in self.config["dag"]:
            self.executor_type = (
                "parallel" if self.config["dag"]["enable_parallel"] else "sequential"
            )

        # Then load the secrets.
        self.secrets = {}
        if secrets is not None:
            self.secrets = WordcelDAG.load_secrets(secrets)

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

    def __substitute_runtime_config_params_in_dag_definition(self, dag_definition):
        """Substitute runtime config params in the DAG definition."""
        if self.runtime_config_params and isinstance(dag_definition, str):
            # If we are provided with config params, substitute them in the YAML file.
            with open(dag_definition, "r") as f:
                pipeline_content = f.read()
            dag_definition = Template(pipeline_content).safe_substitute(
                self.runtime_config_params
            )
        elif self.runtime_config_params and isinstance(dag_definition, dict):
            log.warning(
                "Runtime config params are provided, but the DAG definition "
                "is not a file. Some variables may not be substituted."
            )
        return dag_definition

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
    def load_secrets(secrets) -> Dict[str, str]:
        """Load a secrets file."""
        assert isinstance(secrets, (str, dict)), "Secrets must be a string or dict."

        # Check exists and is a YAML file.
        if isinstance(secrets, str):
            assert os.path.exists(
                os.path.expanduser(secrets)
            ), f"Secrets file `{secrets}` does not exist."
            assert secrets.endswith(".yaml"), "Secrets file must be a YAML file."

        # If it's a dict, return it.
        if isinstance(secrets, dict):
            return secrets

        # Load the YAML file.
        return WordcelDAG.load_yaml(secrets)

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

        # Use topological sort to create layers
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

        try:
            # Infer format from path extension
            render_path, file_format = os.path.splitext(path)
            file_format = file_format.lstrip('.')
            
            dot.render(render_path, format=file_format, view=False, cleanup=True)
            print(f"DAG visualization saved to {path}")
        except graphviz.backend.ExecutableNotFound:
            print("Graphviz executable not found. Please install it from https://graphviz.org/download/")


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
                    custom_functions=self.default_functions,
                )
            except ValueError as e:
                raise ValueError(f"Error creating node {node_id}: {str(e)}")
        return nodes

    def __prepare_incoming_input(self, input_data, results, node_id: str):
        """Prepare the incoming input for a node."""
        incoming_edges = self.graph.nodes[node_id].get("input")
        incoming_input = None
        if (
            input_data is not None
            and isinstance(input_data, dict)
            and node_id in input_data
        ):
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

    def _validate_node_input(self, node_id: str, incoming_input: Any):
        """Validate the input for a node against its spec."""
        node = self.nodes[node_id]
        spec = node.input_spec

        # If there's no spec, there's nothing to validate.
        if not spec:
            return

        expected_type = spec.get("type")
        # `object` is a wildcard that accepts any type.
        if expected_type == object:
            return

        if expected_type is None:
            if incoming_input is not None:
                raise TypeError(
                    f"Node '{node_id}' expected no input, but received an input of type {type(incoming_input).__name__}. "
                    f"Node description: {spec.get('description')}"
                )
            return

        if not isinstance(incoming_input, expected_type):
            raise TypeError(
                f"Node '{node_id}' received an invalid input type. "
                f"Expected {expected_type}, but got {type(incoming_input).__name__}. "
                f"Node description: {spec.get('description')}"
            )

    def __check_result_is_json_serializable(self, results, node_id: str):
        if not _is_json_serializable(results[node_id]):
            result_type = type(results[node_id]).__name__
            raise ValueError(
                f"Node `{node_id}` returned type `{result_type}`, which either is not serializable or contains something, like a DataFrame, that is not serializable."
            )

    def execute(
        self,
        input_data: Dict[str, Any] = None,
        verbose=False,
        executor_type: str = None,
        console=None,
        **executor_kwargs,
    ) -> Dict[str, Any]:
        """Execute the DAG using the specified executor.

        @param input_data: A dictionary of input data for the nodes. The key
            is the node ID that the input data is for.
        @param verbose: Whether to print verbose output during execution.
        @param executor_type: Override the executor type (e.g., 'parallel', 'sequential').
        @param console: Console instance to use for output. If None, uses default console.
        @param executor_kwargs: Additional arguments to pass to the executor.
        """
        # Determine executor type to use
        exec_type = executor_type or self.executor_type

        # Create executor with configuration
        executor_args = {"verbose": verbose}
        if console:
            executor_args["console"] = console
        if exec_type == "parallel":
            executor_args["max_workers"] = executor_kwargs.get(
                "max_workers", self.max_workers
            )

        # Override with any additional kwargs
        executor_args.update(executor_kwargs)

        # Create and run executor
        executor = ExecutorRegistry.create(exec_type, **executor_args)
        return executor.execute(self, input_data)

    def get_node_info(self):
        """Get summary info about all nodes in the DAG."""
        info = []
        for node_id in nx.topological_sort(self.graph):
            node = self.nodes[node_id]
            predecessors = list(self.graph.predecessors(node_id))
            info.append(
                {
                    "id": node_id,
                    "type": node.__class__.__name__,
                    "config_keys": list(node.config.keys()),
                    "inputs": predecessors if predecessors else None,
                }
            )
        return info

    def get_execution_order(self):
        """Get the execution order of nodes in the DAG."""
        return list(nx.topological_sort(self.graph))

    def dry_run(self):
        """Validate DAG configuration without executing nodes."""
        start_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        console.print(
            f"\n🔍 [bold blue]Running DAG validation:[/bold blue] [bold]{self.name}[/bold] [dim]({start_timestamp})[/dim]"
        )
        console.print(f"📊 [dim]Total nodes: {len(self.nodes)}[/dim]\n")

        issues = []
        nodes_list = list(nx.topological_sort(self.graph))

        for i, node_id in enumerate(nodes_list, 1):
            node = self.nodes[node_id]
            node_type = node.__class__.__name__

            # Add model info if available
            model_info = ""
            if hasattr(node, "config") and "model" in node.config:
                model_info = f" | Model: {node.config['model']}"

            node_timestamp = datetime.now().strftime("%H:%M:%S")
            console.print(
                f"[dim][{node_timestamp}][/dim] [bold cyan]\[{i}/{len(nodes_list)}][/bold cyan] [bold]{node_id}[/bold] [dim]({node_type}{model_info})[/dim]",
                end=" ",
            )

            try:
                node.validate_config()
                console.print("[bold green]✓[/bold green]")
            except Exception as e:
                issues.append(f"{node_id}: {e}")
                console.print(f"[bold red]✗[/bold red] [red]{e}[/red]")

        console.print()
        if issues:
            console.print(f"[red]❌ Found {len(issues)} validation issues:[/red]")
            for issue in issues:
                console.print(f"   [red]• {issue}[/red]")
            return False
        else:
            console.print(
                "[bold green]✅ DAG validation passed! All nodes configured correctly.[/bold green]"
            )
            return True
