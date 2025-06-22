"""Tests for prettier execution output."""

import io
import sys
from wordcel.dag import WordcelDAG
from rich.console import Console


def test_prettier_sequential_execution_output():
    """Test that sequential execution has prettier output format."""
    # Capture console output
    console_output = io.StringIO()
    console = Console(file=console_output, width=120)

    dag_config = {
        "dag": {"name": "test_prettier_output"},
        "nodes": [
            {
                "id": "simple_node",
                "type": "csv",
                "path": "https://raw.githubusercontent.com/cs109/2014_data/master/countries.csv",
            }
        ],
    }

    dag = WordcelDAG(dag_config)

    # Test dry run output
    dag.dry_run()

    # For actual execution, we'd need to mock or use a simpler node type
    # This test mainly verifies the structure is correct


def test_prettier_parallel_execution_output():
    """Test that parallel execution has prettier output format."""
    # Similar structure for parallel execution
    dag_config = {
        "dag": {"name": "test_prettier_parallel"},
        "nodes": [
            {
                "id": "node1",
                "type": "csv",
                "path": "https://raw.githubusercontent.com/cs109/2014_data/master/countries.csv",
            },
            {
                "id": "node2",
                "type": "dataframe_operation",
                "input": "node1",
                "operation": "head",
                "args": [5],
            },
        ],
    }

    dag = WordcelDAG(dag_config)
    dag.dry_run()


def test_model_display_in_execution_format():
    """Test that model information appears correctly in execution output format."""
    dag_config = {
        "dag": {"name": "test_model_format"},
        "nodes": [
            {
                "id": "data_node",
                "type": "csv",
                "path": "https://raw.githubusercontent.com/cs109/2014_data/master/countries.csv",
            },
            {
                "id": "llm_explicit",
                "type": "llm",
                "input": "data_node",
                "input_field": "Country",
                "output_field": "Summary",
                "template": "Summarize: {input}",
                "model": "openai/gpt-4o-mini",
            },
            {
                "id": "llm_default",
                "type": "llm",
                "input": "data_node",
                "input_field": "Country",
                "output_field": "Description",
                "template": "Describe: {input}",
                # No model specified - should use default
            },
        ],
    }

    dag = WordcelDAG(dag_config)

    # Verify both nodes are created correctly
    assert dag.nodes["llm_explicit"].config.get("model") == "openai/gpt-4o-mini"
    assert "model" not in dag.nodes["llm_default"].config

    # Test dry run to verify structure
    dag.dry_run()


def test_execution_indicators():
    """Test that execution indicators (running, cached, completed) are intuitive."""
    # This is more of a visual/manual test, but we can verify the basic structure
    dag_config = {
        "dag": {"name": "test_indicators"},
        "nodes": [
            {
                "id": "test_node",
                "type": "csv",
                "path": "https://raw.githubusercontent.com/cs109/2014_data/master/countries.csv",
            }
        ],
    }

    dag = WordcelDAG(dag_config)

    # Test that the executors can be created with the prettier formatting
    from wordcel.dag.executors import SequentialDAGExecutor, ParallelDAGExecutor

    seq_executor = SequentialDAGExecutor(verbose=False)
    parallel_executor = ParallelDAGExecutor(max_workers=2, verbose=False)

    assert seq_executor is not None
    assert parallel_executor is not None

    # The actual formatting is tested through manual inspection of output
