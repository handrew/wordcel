import unittest
import os
import pandas as pd
from wordcel.dag import WordcelDAG


class TestWordcelDAG(unittest.TestCase):
    def setUp(self):
        self.test_yaml_path = "test_dag.yaml"
        self.test_output_path = "test_output.txt"
        self.test_image_path = "test_dag.png"
        self.test_combined_output_path = "test_combined_output.csv"

    def tearDown(self):
        # Clean up any files created during tests
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
    input_column: "Country"
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

        self.assertTrue(os.path.exists(self.test_image_path))
        self.assertTrue(os.path.exists(self.test_output_path))
        self.assertIn("save_results", results)

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

        self.assertTrue(os.path.exists(self.test_combined_output_path))
        self.assertIn("output", results)

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
    args: [10]

  - id: node3
    type: dataframe_operation
    input: node1
    operation: tail
    args: [10]

  - id: node4
    type: llm
    input: [node2, node3]
    template: "Summarize this data: {input}"

  - id: node5
    input: node4
    type: file_writer
    path: test_output.txt
        """
        self.create_test_yaml(dag_config)

        dag = WordcelDAG(self.test_yaml_path)
        results = dag.execute()

        self.assertTrue(os.path.exists(self.test_output_path))
        self.assertIn("node5", results)
        self.assertIsInstance(results["node1"], pd.DataFrame)
        self.assertIsInstance(results["node2"], pd.DataFrame)
        self.assertIsInstance(results["node3"], pd.DataFrame)


if __name__ == "__main__":
    unittest.main()
