import unittest
import os
import json
import pandas as pd
from wordcel.dag import WordcelDAG

class TestJSONNodes(unittest.TestCase):
    def setUp(self):
        # Create temporary JSON files for testing
        self.json_data = {
            "name": "John Doe",
            "age": 30,
            "city": "New York"
        }
        self.json_list_data = [
            {"name": "John Doe", "age": 30, "city": "New York"},
            {"name": "Jane Smith", "age": 25, "city": "Los Angeles"}
        ]
        
        with open('test_data.json', 'w') as f:
            json.dump(self.json_data, f)
        
        with open('test_list_data.json', 'w') as f:
            json.dump(self.json_list_data, f)

        # Create YAML configuration
        self.yaml_config = """
dag:
  name: json_test_dag

nodes:
  - id: read_json
    type: json
    path: test_data.json

  - id: read_json_df
    type: json_dataframe
    path: test_list_data.json
    read_json_kwargs:
      orient: records
"""
        with open('test_config.yaml', 'w') as f:
            f.write(self.yaml_config)


    def tearDown(self):
        # Clean up temporary files
        os.remove('test_data.json')
        os.remove('test_list_data.json')
        os.remove('test_config.yaml')

    def test_json_and_json_dataframe_nodes(self):
        dag = WordcelDAG('test_config.yaml')
        results = dag.execute()

        # Test JSON node
        self.assertIsInstance(results['read_json'], dict)
        self.assertEqual(results['read_json'], self.json_data)

        # Test JSON DataFrame node
        self.assertIsInstance(results['read_json_df'], pd.DataFrame)
        self.assertEqual(len(results['read_json_df']), 2)
        self.assertListEqual(list(results['read_json_df'].columns), ['name', 'age', 'city'])


if __name__ == '__main__':
    unittest.main()
