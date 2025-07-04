import os
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from wordcel.dag import WordcelDAG


class TestWordcelDAG:
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        # Setup
        self.test_yaml_path = "test_dag.yaml"
        self.test_output_path = "test_output.txt"
        self.test_image_path = "test_dag.png"
        self.test_combined_output_path = "test_combined_output.csv"

        yield

        # Teardown
        for file_path in [
            self.test_yaml_path,
            self.test_output_path,
            self.test_image_path,
            self.test_combined_output_path,
        ]:
            if os.path.exists(file_path):
                os.remove(file_path)

    def create_test_yaml(self, dag_config):
        with open(self.test_yaml_path, "w") as f:
            f.write(dag_config)

    def test_simple_pipeline_with_llm_filtering(self):
        dag_config = """
dag:
  name: simple_llm_filtering_pipeline

nodes:
  - id: get_data
    type: csv
    path: "https://raw.githubusercontent.com/cs109/2014_data/master/countries.csv"

  - id: df_filter
    type: dataframe_operation
    input: get_data
    operation: "head"
    args: [10]

  - id: llm_filter
    input: df_filter
    type: llm_filter
    column: "Country"
    prompt: "Is this country in Africa? Answer only Yes or No."
    num_threads: 2

  - id: process_filtered
    type: llm
    template: "What cuisine is this country known for? {input}"
    input: llm_filter
    input_field: "Country"
    output_field: "Cuisine"
    num_threads: 2

  - id: format_output
    type: string_template
    input: process_filtered
    template: "${Cuisine}"

  - id: save_results
    type: file_writer
    path: "test_output.txt"
    input: format_output
        """
        self.create_test_yaml(dag_config)

        dag = WordcelDAG(self.test_yaml_path)
        dag.save_image(self.test_image_path)
        results = dag.execute()

        assert os.path.exists(self.test_image_path)
        assert os.path.exists(self.test_output_path)
        assert "save_results" in results

    def test_pipeline_with_dataframe_operation(self):
        dag_config = """
dag:
  name: dataframe_operation_pipeline

nodes:
  - id: countries_data
    type: csv
    path: https://raw.githubusercontent.com/cs109/2014_data/master/countries.csv

  - id: mtcars_data
    type: csv
    path: https://raw.githubusercontent.com/cs109/2014_data/master/mtcars.csv

  - id: combine_data
    type: dataframe_operation
    operation: concat
    kwargs:
      axis: 1
    input:
      - countries_data
      - mtcars_data

  - id: format_output
    type: string_template
    input: combine_data
    template: "${__str__}"

  - id: output
    type: file_writer
    path: test_combined_output.csv
    input: format_output
        """
        self.create_test_yaml(dag_config)

        dag = WordcelDAG(self.test_yaml_path)
        results = dag.execute()

        assert os.path.exists(self.test_combined_output_path)
        assert "output" in results

    def test_pipeline_with_multiple_inputs_outputs(self):
        dag_config = """
dag:
  name: multiple_inputs_outputs_pipeline

nodes:
  - id: node1
    type: csv
    path: https://raw.githubusercontent.com/cs109/2014_data/master/countries.csv

  - id: node2
    type: dataframe_operation
    input: node1
    operation: head
    args: [5]

  - id: node3
    type: dataframe_operation
    input: node1
    operation: tail
    args: [5]

  - id: node4
    type: llm
    input: [node2, node3]
    template: "What is this country known for?: {input}"
    input_field: "Country"
    output_field: "Known For"

  - id: node5
    input: node4
    type: file_writer
    path: test_output.txt
        """
        self.create_test_yaml(dag_config)

        dag = WordcelDAG(self.test_yaml_path)
        results = dag.execute()

        assert os.path.exists(self.test_output_path)
        assert "node5" in results
        assert isinstance(results["node1"], pd.DataFrame)
        assert isinstance(results["node2"], pd.DataFrame)
        assert isinstance(results["node3"], pd.DataFrame)

    def test_file_not_found_error(self):
        """Test that file nodes give clear error messages for missing files."""
        dag_config = """
dag:
  name: test_file_not_found

nodes:
  - id: missing_csv
    type: csv
    path: /nonexistent/file.csv
        """
        self.create_test_yaml(dag_config)

        dag = WordcelDAG(self.test_yaml_path)

        with pytest.raises(RuntimeError) as excinfo:
            dag.execute()

        # The RuntimeError should contain the id of the failing node.
        assert "missing_csv" in str(excinfo.value)
        assert "failed" in str(excinfo.value)

    def test_get_node_info(self):
        """Test the get_node_info() method."""
        dag_config = """
dag:
  name: test_node_info

nodes:
  - id: load_data
    type: csv
    path: "https://raw.githubusercontent.com/cs109/2014_data/master/countries.csv"

  - id: process_data
    type: dataframe_operation
    input: load_data
    operation: head
    args: [5]
        """
        self.create_test_yaml(dag_config)

        dag = WordcelDAG(self.test_yaml_path)
        info = dag.get_node_info()

        assert len(info) == 2

        # Check first node
        assert info[0]["id"] == "load_data"
        assert info[0]["type"] == "CSVNode"
        assert "path" in info[0]["config_keys"]
        assert info[0]["inputs"] is None

        # Check second node
        assert info[1]["id"] == "process_data"
        assert info[1]["type"] == "DataFrameOperationNode"
        assert "operation" in info[1]["config_keys"]
        assert info[1]["inputs"] == ["load_data"]

    def test_get_execution_order(self):
        """Test the get_execution_order() method returns correct topological order."""
        dag_config = """
dag:
  name: test_execution_order

nodes:
  - id: node_a
    type: csv
    path: "https://raw.githubusercontent.com/cs109/2014_data/master/countries.csv"

  - id: node_b
    type: dataframe_operation
    input: node_a
    operation: head
    args: [5]

  - id: node_c
    type: dataframe_operation
    input: node_b
    operation: tail
    args: [3]

  - id: node_d
    type: dataframe_operation
    input: node_a
    operation: describe
        """
        self.create_test_yaml(dag_config)

        dag = WordcelDAG(self.test_yaml_path)
        execution_order = dag.get_execution_order()

        # Check that we get all nodes
        assert len(execution_order) == 4
        assert set(execution_order) == {"node_a", "node_b", "node_c", "node_d"}

        # Check topological ordering constraints
        # node_a must come before node_b and node_d
        assert execution_order.index("node_a") < execution_order.index("node_b")
        assert execution_order.index("node_a") < execution_order.index("node_d")
        # node_b must come before node_c
        assert execution_order.index("node_b") < execution_order.index("node_c")

    def test_dry_run_success(self):
        """Test dry_run() with valid configuration."""
        dag_config = """
dag:
  name: test_dry_run_success

nodes:
  - id: load_data
    type: csv
    path: "https://raw.githubusercontent.com/cs109/2014_data/master/countries.csv"

  - id: process_data
    type: dataframe_operation
    input: load_data
    operation: head
    args: [5]
        """
        self.create_test_yaml(dag_config)

        dag = WordcelDAG(self.test_yaml_path)
        result = dag.dry_run()

        assert result is True

    def test_dry_run_failure(self):
        """Test that invalid configuration is caught during DAG construction."""
        dag_config = """
dag:
  name: test_dry_run_failure

nodes:
  - id: bad_csv
    type: csv
    # Missing required 'path' field
        """
        self.create_test_yaml(dag_config)

        # The DAG constructor should fail due to validation
        with pytest.raises(AssertionError) as excinfo:
            dag = WordcelDAG(self.test_yaml_path)

        assert "must have a 'path' configuration" in str(excinfo.value)

    def test_http_urls_bypass_file_check(self):
        """Test that HTTP URLs don't trigger file existence checks."""
        dag_config = """
dag:
  name: test_http_urls

nodes:
  - id: load_remote_csv
    type: csv
    path: "https://raw.githubusercontent.com/cs109/2014_data/master/countries.csv"
        """
        self.create_test_yaml(dag_config)

        dag = WordcelDAG(self.test_yaml_path)

        # Should not raise FileNotFoundError for HTTP URLs
        results = dag.execute()
        assert "load_remote_csv" in results
        assert isinstance(results["load_remote_csv"], pd.DataFrame)

    def test_rich_logging_and_timing(self):
        """Test that DAG execution includes timing and rich formatting."""
        dag_config = """
dag:
  name: test_timing

nodes:
  - id: load_data
    type: csv
    path: "https://raw.githubusercontent.com/cs109/2014_data/master/countries.csv"

  - id: quick_op
    type: dataframe_operation
    input: load_data
    operation: head
    args: [3]
        """
        self.create_test_yaml(dag_config)

        # Capture console output
        from io import StringIO
        import sys
        from rich.console import Console

        output_buffer = StringIO()
        test_console = Console(file=output_buffer, width=80)

        # Temporarily replace the console
        import wordcel.dag.dag as dag_module

        original_console = dag_module.console
        dag_module.console = test_console

        try:
            dag = WordcelDAG(self.test_yaml_path)
            results = dag.execute(console=test_console)

            # Get the captured output
            output = output_buffer.getvalue()

            # Check for expected timing and progress indicators
            assert (
                "Executing DAG" in output
            )  # Works for both "Executing DAG:" and "Executing DAG (Parallel):"
            assert "test_timing" in output
            assert "load_data" in output
            assert "quick_op" in output
            assert "✅" in output  # Success checkmarks
            assert "completed successfully!" in output

            # Verify results
            assert len(results) == 2
            assert "load_data" in results
            assert "quick_op" in results

        finally:
            # Restore original console
            dag_module.console = original_console

    def test_llm_with_web_search(self):
        """Test that web_search_options are passed to the llm_call function."""
        dag_config = """
dag:
  name: test_web_search

nodes:
  - id: search_query
    type: string_template
    template: "What's the weather in London?"

  - id: web_search_llm
    type: llm
    input: search_query
    template: "{input}"
    model: "unsupported/unsupported"
    web_search_options:
      search_context_size: "medium"
        """
        self.create_test_yaml(dag_config)

        dag = WordcelDAG(self.test_yaml_path)
        with pytest.raises(RuntimeError) as excinfo:
            dag.execute()

        # The RuntimeError should wrap a RetryError, which in turn wraps the AssertionError
        assert "Provider `unsupported` not supported" in str(excinfo.value.__cause__.__cause__)

    def test_string_template_node_with_multiple_inputs(self):
        """Test that StringTemplateNode can handle multiple named inputs."""
        dag_config = """
dag:
  name: test_string_template_multiple_inputs

nodes:
  - id: get_name
    type: yaml
    path: "name.yaml"

  - id: get_place
    type: yaml
    path: "place.yaml"

  - id: make_greeting
    type: string_template
    input:
      - get_name
      - get_place
    template: "Hello ${get_name}, welcome to ${get_place}."
"""
        self.create_test_yaml(dag_config)
        with open("name.yaml", "w") as f:
            f.write('"World"')
        with open("place.yaml", "w") as f:
            f.write('"the Machine"')

        dag = WordcelDAG(self.test_yaml_path)
        results = dag.execute()

        assert "make_greeting" in results
        assert results["make_greeting"] == "Hello World, welcome to the Machine."

        # Clean up the extra files
        if os.path.exists("name.yaml"):
            os.remove("name.yaml")
        if os.path.exists("place.yaml"):
            os.remove("place.yaml")

    def test_input_spec_validation(self):
        """Test that the DAG validates node inputs against their spec."""
        dag_config = """
dag:
  name: test_input_spec_validation

nodes:
  - id: string_producer
    type: yaml
    path: "input.yaml"

  - id: dataframe_consumer
    type: llm_filter
    input: string_producer
    column: "some_column"
    prompt: "some_prompt"
        """
        self.create_test_yaml(dag_config)
        with open("input.yaml", "w") as f:
            f.write('"this is just a string"')

        dag = WordcelDAG(self.test_yaml_path)

        with pytest.raises(RuntimeError) as excinfo:
            dag.execute()

        # Check that the error message is from our new validation logic
        assert "failed" in str(excinfo.value)
        assert "dataframe_consumer" in str(excinfo.value)
        assert "invalid input type" in str(excinfo.value)
        assert "Expected <class 'pandas.core.frame.DataFrame'>" in str(excinfo.value)
        assert "got str" in str(excinfo.value)

        # Clean up the extra file
        if os.path.exists("input.yaml"):
            os.remove("input.yaml")
