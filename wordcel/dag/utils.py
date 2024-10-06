"""Utility functions for the DAG module."""
import importlib.util
from typing import Dict, Type
from .nodes import Node
from .backends import Backend


def load_module(file_path, module_name):
    """Load a Python module from a file."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def create_custom_nodes_from_file(custom_nodes_file: str) -> Dict[str, Type[Node]]:
    """Create custom nodes from a Python file."""
    nodes_module = load_module(custom_nodes_file, "custom_nodes")
    custom_nodes = {name: cls for name, cls in nodes_module.__dict__.items() if isinstance(cls, type) and issubclass(cls, Node) and cls is not Node}
    return custom_nodes


def create_custom_functions_from_file(custom_functions_file: str) -> Dict[str, callable]:
    """Create custom functions from a Python file."""
    functions_module = load_module(custom_functions, "custom_functions")
    custom_functions = {name: func for name, func in functions_module.__dict__.items() if callable(func)}


def create_custom_backends_from_file(custom_backends_file: str) -> Dict[str, Type[Backend]]:
    """Create custom backends from a Python file."""
    backends_module = load_module(custom_backends, "custom_backends")
    custom_backends = {name: cls for name, cls in backends_module.__dict__.items() if isinstance(cls, type) and issubclass(cls, Backend) and cls is not Backend}
    return custom_backends
