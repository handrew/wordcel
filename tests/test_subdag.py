import unittest
import tempfile
import os
from unittest.mock import patch
from wordcel.dag import WordcelDAG


class TestDAGNode(unittest.TestCase):

    def setUp(self):
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
    key: Country
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
  - id: file_writer
    type: file_writer
    input: sub_dag
    path: output.txt
"""

    @patch("wordcel.llms.openai_call")
    def test_dag_node(self, mock_openai_call):
        # Mock the LLM call to return a fixed response
        mock_openai_call.return_value = "Mocked LLM response"

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
            self.assertIn("sub_dag", results)
            self.assertIsInstance(results["sub_dag"], dict)
            self.assertIn("csv_node", results["sub_dag"])
            self.assertIn("head_node", results["sub_dag"])
            self.assertIn("llm_node", results["sub_dag"])

            # Check if the head operation was applied correctly
            self.assertEqual(len(results["sub_dag"]["head_node"]), 5)

            # Check if the file_writer node was executed
            self.assertIn("file_writer", results)
            self.assertTrue(os.path.exists("output.txt"))

        finally:
            # Clean up temporary files
            os.unlink(sub_dag_path)
            os.unlink(main_dag_path)
            if os.path.exists("output.txt"):
                os.unlink("output.txt")


if __name__ == "__main__":
    unittest.main()
