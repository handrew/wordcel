from wordcel.dag import Node
from typing import Union


class MultiplyNode(Node):
    def execute(self, input_data: Union[int, float]) -> Union[int, float]:
        factor = self.config.get("factor", 1)
        return input_data * factor

    def validate_config(self) -> bool:
        if "factor" not in self.config:
            raise ValueError("MultiplyNode must have a 'factor' configuration.")
        if not isinstance(self.config["factor"], (int, float)):
            raise ValueError("MultiplyNode 'factor' must be a number.")
        return True


import unittest
from unittest.mock import patch
from wordcel.dag import WordcelDAG, NodeRegistry
import pandas as pd
import yaml
import tempfile
import os


class TestMultiplyNodeAndDAG(unittest.TestCase):

    def setUp(self):
        self.config = {"type": "multiply", "factor": 2}
        self.secrets = {}
        self.node = MultiplyNode(self.config, self.secrets)

    def test_execute(self):
        # Test with integer input
        result = self.node.execute(5)
        self.assertEqual(result, 10)

        # Test with float input
        result = self.node.execute(3.5)
        self.assertEqual(result, 7.0)

    def test_validate_config_success(self):
        self.assertTrue(self.node.validate_config())

    def test_validate_config_missing_factor(self):
        node = MultiplyNode({"type": "multiply"}, self.secrets)
        with self.assertRaises(ValueError):
            node.validate_config()

    def test_validate_config_invalid_factor(self):
        node = MultiplyNode({"type": "multiply", "factor": "invalid"}, self.secrets)
        with self.assertRaises(ValueError):
            node.validate_config()

    def test_node_registration(self):
        NodeRegistry.register("multiply", MultiplyNode)
        self.assertEqual(NodeRegistry.get("multiply"), MultiplyNode)

    def test_create_node(self):
        from wordcel.dag import create_node

        NodeRegistry.register("multiply", MultiplyNode)
        node = create_node(self.config, self.secrets)
        self.assertIsInstance(node, MultiplyNode)

    def test_dag_with_custom_node(self):
        # Create a temporary YAML file for the DAG configuration
        dag_config = {
            "dag": {"name": "test_dag"},
            "nodes": [
                {"id": "csv_node", "type": "csv", "path": "test.csv"},
                {
                    "id": "multiply_node",
                    "type": "multiply",
                    "factor": 2,
                    "input": "csv_node",
                },
                {
                    "id": "llm_node",
                    "type": "llm",
                    "template": "Multiply result: {input}",
                    "input": "multiply_node",
                    "input_column": "value",
                },
            ],
        }

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
            yaml.dump(dag_config, temp_file)
            temp_file_path = temp_file.name

        # Mock the necessary components
        mock_csv_data = pd.DataFrame({"value": [1, 2, 3]})
        mock_llm_output = [
            "Multiply result: 2",
            "Multiply result: 4",
            "Multiply result: 6",
        ]

        with patch("pandas.read_csv", return_value=mock_csv_data), patch(
            "wordcel.dag.openai_call", side_effect=mock_llm_output
        ):

            # Create and execute the DAG
            dag = WordcelDAG(temp_file_path, custom_nodes={"multiply": MultiplyNode})
            results = dag.execute()

        # Clean up the temporary file
        os.unlink(temp_file_path)

        # Assert the results
        self.assertIn("csv_node", results)
        self.assertIn("multiply_node", results)
        self.assertIn("llm_node", results)

        pd.testing.assert_frame_equal(results["csv_node"], mock_csv_data)
        pd.testing.assert_frame_equal(results["multiply_node"], mock_csv_data * 2)
        self.assertEqual(results["llm_node"], mock_llm_output)


if __name__ == "__main__":
    unittest.main()
