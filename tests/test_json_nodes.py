import os
import json
import pandas as pd
import pytest
from wordcel.dag import WordcelDAG

class TestJSONNodes:
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        # Clean up any existing files first
        test_files = ['test_data.json', 'test_list_data.json', 'test_config.yaml']
        
        for file_path in test_files:
            if os.path.exists(file_path):
                os.remove(file_path)
        
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

        yield
        
        # Clean up temporary files
        for file_path in test_files:
            if os.path.exists(file_path):
                os.remove(file_path)

    def test_json_and_json_dataframe_nodes(self):
        dag = WordcelDAG('test_config.yaml')
        results = dag.execute()

        # Test JSON node
        assert isinstance(results['read_json'], dict)
        assert results['read_json'] == self.json_data

        # Test JSON DataFrame node
        assert isinstance(results['read_json_df'], pd.DataFrame)
        assert len(results['read_json_df']) == 2
        assert list(results['read_json_df'].columns) == ['name', 'age', 'city']
