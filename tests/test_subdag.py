import tempfile
import os
import pytest
from unittest.mock import patch
from wordcel.dag import WordcelDAG


class TestDAGNode:

    @pytest.fixture(autouse=True)
    def setup(self):
        # URL of the CSV file

        # Sub-DAG YAML configuration
        self.sub_dag_yaml = """
dag:
    name: sub_dag
nodes:
  - id: csv_node
    type: csv
    path: https://raw.githubusercontent.com/cs109/2014_data/master/countries.csv
  - id: head_node
    type: dataframe_operation
    input: csv_node
    operation: head
    args: [5]
  - id: llm_node
    type: llm
    input: head_node
    input_field: Country
    output_field: Summary
    template: "Summarize this data: {input}"
"""

        # Main DAG YAML configuration
        self.main_dag_yaml = """
dag:
    name: main_dag
nodes:
  - id: sub_dag
    type: dag
    path: {sub_dag_path}
  - id: format_output
    type: string_template
    input: sub_dag
    template: "${{llm_node}}"
  - id: file_writer
    type: file_writer
    input: format_output
    path: output.txt
"""

    @patch("wordcel.llms.llm_call")
    def test_dag_node(self, mock_llm_call):
        # Mock the LLM call to return a fixed response
        mock_llm_call.return_value = "Mocked LLM response"

        # Create temporary sub-DAG YAML file
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".yaml"
        ) as sub_dag_file:
            sub_dag_file.write(self.sub_dag_yaml)
            sub_dag_path = sub_dag_file.name

        # Create temporary main DAG YAML file
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".yaml"
        ) as main_dag_file:
            main_dag_file.write(self.main_dag_yaml.format(sub_dag_path=sub_dag_path))
            main_dag_path = main_dag_file.name

        try:
            # Initialize and execute the main DAG
            dag = WordcelDAG(main_dag_path)
            results = dag.execute()

            # Check if the sub_dag was executed
            assert "sub_dag" in results
            assert isinstance(results["sub_dag"], dict)
            assert "csv_node" in results["sub_dag"]
            assert "head_node" in results["sub_dag"]
            assert "llm_node" in results["sub_dag"]

            # Check if the head operation was applied correctly
            assert len(results["sub_dag"]["head_node"]) == 5

            # Check if the file_writer node was executed
            assert "file_writer" in results
            assert os.path.exists("output.txt")

        finally:
            # Clean up temporary files
            os.unlink(sub_dag_path)
            os.unlink(main_dag_path)
            if os.path.exists("output.txt"):
                os.unlink("output.txt")
