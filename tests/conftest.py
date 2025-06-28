"""Pytest configuration and fixtures for the test suite."""

import os
import glob
import pytest


@pytest.fixture(scope="session", autouse=True)
def cleanup_test_files():
    """Clean up any files created during testing."""
    yield  # This allows tests to run first

    # List of specific files that might be created by tests
    test_files = [
        # DAG test files
        "test_dag.yaml",
        "test_output.txt",
        "test_dag.png",
        "test_combined_output.csv",
        # Python function test files
        "test_functions.py",
        # JSON node test files
        "test_config.yaml",
        # Contextual retrieval test files
        "retriever.pkl",
        # Subdag test files
        "output.txt",
        # Backend test files
        "dag.yaml",
        "test.csv",
        "test_backend.csv",
        "test_backend_dag.yaml",
    ]

    # Also clean up any files matching test patterns
    patterns = [
        "test_*.yaml",
        "test_*.txt",
        "test_*.csv",
        "test_*.png",
        "test_*.pkl",
        "*.pkl",
    ]

    files_to_remove = set(test_files)

    # Add pattern matches
    for pattern in patterns:
        files_to_remove.update(glob.glob(pattern))

    # Remove the files
    for file_path in files_to_remove:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"Cleaned up test file: {file_path}")
            except OSError as e:
                print(f"Warning: Could not remove {file_path}: {e}")

    # Clean up any cache directories and __pycache__ directories
    directories_to_remove = ["__pycache__", "cache", "test_backend_cache"]

    for dir_name in directories_to_remove:
        if os.path.exists(dir_name):
            import shutil

            try:
                shutil.rmtree(dir_name)
                print(f"Cleaned up {dir_name} directory")
            except OSError as e:
                print(f"Warning: Could not remove {dir_name}: {e}")
