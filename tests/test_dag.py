import os
import pandas as pd
import pytest
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

  - id: save_results
    type: file_writer
    path: "test_output.txt"
    input: process_filtered
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

  - id: output
    type: file_writer
    path: test_combined_output.csv
    input: combine_data
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
        
        # The RuntimeError should contain the original FileNotFoundError message
        assert "CSV file not found" in str(excinfo.value)
        assert "/nonexistent/file.csv" in str(excinfo.value)

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
        assert info[0]['id'] == 'load_data'
        assert info[0]['type'] == 'CSVNode'
        assert 'path' in info[0]['config_keys']
        assert info[0]['inputs'] is None
        
        # Check second node
        assert info[1]['id'] == 'process_data'
        assert info[1]['type'] == 'DataFrameOperationNode'
        assert 'operation' in info[1]['config_keys']
        assert info[1]['inputs'] == ['load_data']

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
            assert "Executing DAG" in output  # Works for both "Executing DAG:" and "Executing DAG (Parallel):"
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
