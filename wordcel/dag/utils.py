"""Utility functions for the DAG module."""
import importlib.util
from typing import Dict, Type, Union, List
from rich import print
from .nodes import Node
from .backends import Backend


def load_module(file_path, module_name):
    """Load a Python module from a file."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def create_custom_nodes_from_files(custom_nodes_files: Union[str, List]) -> Dict[str, Type[Node]]:
    """Create custom nodes from a Python file."""
    if isinstance(custom_nodes_files, str):
        custom_nodes_files = [custom_nodes_files]

    custom_nodes = {}
    for custom_nodes_file in custom_nodes_files:
        nodes_module = load_module(custom_nodes_file, custom_nodes_file)
        
        found_nodes = {name: cls for name, cls in nodes_module.__dict__.items() if isinstance(cls, type) and issubclass(cls, Node) and cls is not Node}
        # Check for duplicates.
        duplicates = set(custom_nodes.keys()) & set(found_nodes.keys())
        if duplicates:
            raise ValueError(f"Duplicate node(s) found in {custom_nodes_file}: {duplicates}.")

        custom_nodes.update(found_nodes)
    
    print("Created custom nodes: ", custom_nodes)
    return custom_nodes


def create_custom_functions_from_files(custom_functions_files: Union[str, List]) -> Dict[str, callable]:
    """Create custom functions from a Python file."""
    if isinstance(custom_functions_files, str):
        custom_functions_files = [custom_functions_files]

    custom_functions = {}
    for custom_functions_file in custom_functions_files:
        functions_module = load_module(custom_functions, "custom_functions")
        found_functions = {name: func for name, func in functions_module.__dict__.items() if callable(func)}
        # Check for duplicates.
        duplicates = set(custom_functions.keys()) & set(found_functions.keys())
        if duplicates:
            raise ValueError(f"Duplicate function(s) found in {custom_functions_file}: {duplicates}.")
        
        custom_functions.update(found_functions)

    print("Created custom functions: ", custom_functions)
    return custom_functions


def create_custom_backends_from_files(custom_backends_files: Union[str, List]) -> Dict[str, Type[Backend]]:
    """Create custom backends from a Python file."""
    if isinstance(custom_backends_files, str):
        custom_backends_files = [custom_backends_files]

    custom_backends = {}
    for custom_backends_file in custom_backends_files:
        backends_module = load_module(custom_backends, "custom_backends")
        found_backends = {name: cls for name, cls in backends_module.__dict__.items() if isinstance(cls, type) and issubclass(cls, Backend) and cls is not Backend}
        # Check for duplicates.
        duplicates = set(custom_backends.keys()) & set(found_backends.keys())
        if duplicates:
            raise ValueError(f"Duplicate backend(s) found in {custom_backends_file}: {duplicates}.")
        
        custom_backends.update(found_backends)

    print("Created custom backends: ", custom_backends)
    return custom_backends
