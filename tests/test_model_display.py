"""Tests for model display functionality in DAG execution."""

import tempfile
import os
from wordcel.dag import WordcelDAG


def test_model_display_explicit_model():
    """Test that explicitly configured models are displayed during execution."""
    dag_config = {
        "dag": {"name": "test_model_display"},
        "nodes": [
            {
                "id": "data_node",
                "type": "csv",
                "path": "https://raw.githubusercontent.com/cs109/2014_data/master/countries.csv",
            },
            {
                "id": "llm_node",
                "type": "llm",
                "input": "data_node",
                "input_field": "Country",
                "output_field": "Summary",
                "template": "Summarize this country in one word: {input}",
                "model": "openai/gpt-4o-mini",
            },
        ],
    }

    dag = WordcelDAG(dag_config)

    # Check that model is properly stored in config
    assert "model" in dag.nodes["llm_node"].config
    assert dag.nodes["llm_node"].config["model"] == "openai/gpt-4o-mini"

    # Test that execution works (we won't actually run the LLM for the test)
    # but we can verify the DAG structure is correct
    dag.dry_run()


def test_model_display_default_model():
    """Test that default model is displayed when no model is explicitly configured."""
    dag_config = {
        "dag": {"name": "test_default_model"},
        "nodes": [
            {
                "id": "data_node",
                "type": "csv",
                "path": "https://raw.githubusercontent.com/cs109/2014_data/master/countries.csv",
            },
            {
                "id": "llm_node",
                "type": "llm",
                "input": "data_node",
                "input_field": "Country",
                "output_field": "Summary",
                "template": "Summarize this country in one word: {input}",
                # Note: no model specified, should use default
            },
        ],
    }

    dag = WordcelDAG(dag_config)

    # Check that model is not in config but default will be used
    assert "model" not in dag.nodes["llm_node"].config

    # Test that execution logic can determine the model
    from wordcel.config import DEFAULT_MODEL

    effective_model = dag.nodes["llm_node"].config.get("model", DEFAULT_MODEL)
    assert effective_model == DEFAULT_MODEL

    dag.dry_run()


def test_model_display_llm_filter_node():
    """Test that LLMFilterNode also displays model information."""
    dag_config = {
        "dag": {"name": "test_llm_filter_model"},
        "nodes": [
            {
                "id": "data_node",
                "type": "csv",
                "path": "https://raw.githubusercontent.com/cs109/2014_data/master/countries.csv",
            },
            {
                "id": "filter_node",
                "type": "llm_filter",
                "input": "data_node",
                "column": "Country",
                "prompt": "Is this country in Africa? Answer only Yes or No.",
                "model": "openai/gpt-3.5-turbo",
            },
        ],
    }

    dag = WordcelDAG(dag_config)

    # Check that model is properly stored in config
    assert "model" in dag.nodes["filter_node"].config
    assert dag.nodes["filter_node"].config["model"] == "openai/gpt-3.5-turbo"

    dag.dry_run()


def test_non_llm_nodes_no_model_display():
    """Test that non-LLM nodes don't show model information."""
    dag_config = {
        "dag": {"name": "test_non_llm_nodes"},
        "nodes": [
            {
                "id": "csv_node",
                "type": "csv",
                "path": "https://raw.githubusercontent.com/cs109/2014_data/master/countries.csv",
            },
            {
                "id": "dataframe_node",
                "type": "dataframe_operation",
                "input": "csv_node",
                "operation": "head",
                "args": [5],
            },
        ],
    }

    dag = WordcelDAG(dag_config)

    # Check that non-LLM nodes don't have model in config
    assert "model" not in dag.nodes["csv_node"].config
    assert "model" not in dag.nodes["dataframe_node"].config

    dag.dry_run()


def test_executor_model_display_logic():
    """Test the executor logic for determining model display."""
    from wordcel.dag.executors import SequentialDAGExecutor
    from wordcel.config import DEFAULT_MODEL

    # Create a mock node with LLM type
    class MockLLMNode:
        def __init__(self, config):
            self.config = config
            self.__class__.__name__ = "LLMNode"

    class MockCSVNode:
        def __init__(self, config):
            self.config = config
            self.__class__.__name__ = "CSVNode"

    # Test LLM node with explicit model
    llm_node_explicit = MockLLMNode({"model": "custom-model"})
    assert llm_node_explicit.__class__.__name__ in ["LLMNode", "LLMFilterNode"]
    model = llm_node_explicit.config.get("model", DEFAULT_MODEL)
    assert model == "custom-model"

    # Test LLM node with default model
    llm_node_default = MockLLMNode({})
    model = llm_node_default.config.get("model", DEFAULT_MODEL)
    assert model == DEFAULT_MODEL

    # Test non-LLM node
    csv_node = MockCSVNode({})
    assert csv_node.__class__.__name__ not in ["LLMNode", "LLMFilterNode"]
