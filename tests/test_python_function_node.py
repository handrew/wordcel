import os
import unittest
import tempfile
import yaml
from wordcel.dag import WordcelDAG

class TestPythonFunctionNode(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for our test files
        self.test_dir = tempfile.mkdtemp()
        
        # Create a test functions file
        self.functions_path = os.path.join(self.test_dir, 'test_functions.py')
        with open(self.functions_path, 'w') as f:
            f.write("""
def add_numbers(a, b):
    return a + b

def process_data(data, prefix='', suffix=''):
    if isinstance(data, list):
        return [f"{prefix}{item}{suffix}" for item in data]
    return f"{prefix}{data}{suffix}"

def process_with_kwarg(prefix='', suffix='', data=None):
    if isinstance(data, list):
        return [f"{prefix}{item}{suffix}" for item in data]
    return f"{prefix}{data}{suffix}"

def standalone_function():
    return "standalone result"
""")

    def create_dag_config(self, nodes):
        """Helper to create a DAG config file"""
        config = {
            "dag": {
                "name": "test_dag"
            },
            "nodes": nodes
        }
        
        config_path = os.path.join(self.test_dir, 'test_config.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        return config_path

    def test_simple_function_call(self):
        """Test calling a simple function with direct arguments"""
        nodes = [{
            "id": "add_numbers",
            "type": "python_function",
            "module_path": self.functions_path,
            "function_name": "add_numbers",
            "mode": "ignore",  # Don't use input
            "args": [5, 3]
        }]
        
        config_path = self.create_dag_config(nodes)
        dag = WordcelDAG(config_path)
        results = dag.execute()
        
        self.assertEqual(results["add_numbers"], 8)

    def test_function_with_arg_input(self):
        """Test calling a function with input data as first argument (default mode)"""
        nodes = [
            {
                "id": "input_data",
                "type": "python_function",
                "module_path": self.functions_path,
                "function_name": "process_data",
                "mode": "ignore",
                "args": [["apple", "banana", "cherry"]],
                "kwargs": {"prefix": "fruit_"}
            },
            {
                "id": "process_fruits",
                "type": "python_function",
                "module_path": self.functions_path,
                "function_name": "process_data",
                "input": "input_data",
                # mode: "arg" is default
                "kwargs": {"suffix": "_processed"}
            }
        ]
        
        config_path = self.create_dag_config(nodes)
        dag = WordcelDAG(config_path)
        results = dag.execute()
        
        expected = [
            "fruit_apple_processed",
            "fruit_banana_processed",
            "fruit_cherry_processed"
        ]
        self.assertEqual(results["process_fruits"], expected)

    def test_function_with_kwarg_input(self):
        """Test calling a function with input data as keyword argument"""
        nodes = [
            {
                "id": "input_data",
                "type": "python_function",
                "module_path": self.functions_path,
                "function_name": "process_data",
                "mode": "ignore",
                "args": [["apple", "banana", "cherry"]],
                "kwargs": {"prefix": "fruit_"}
            },
            {
                "id": "process_fruits",
                "type": "python_function",
                "module_path": self.functions_path,
                "function_name": "process_with_kwarg",
                "input": "input_data",
                "mode": "kwarg",
                "input_kwarg": "data",
                "kwargs": {"prefix": "processed_", "suffix": "_done"}
            }
        ]
        
        config_path = self.create_dag_config(nodes)
        dag = WordcelDAG(config_path)
        results = dag.execute()
        
        expected = [
            "processed_fruit_apple_done",
            "processed_fruit_banana_done",
            "processed_fruit_cherry_done"
        ]
        self.assertEqual(results["process_fruits"], expected)

    def test_function_ignore_input(self):
        """Test calling a function while ignoring input data"""
        nodes = [
            {
                "id": "input_data",
                "type": "python_function",
                "module_path": self.functions_path,
                "function_name": "process_data",
                "mode": "ignore",
                "args": [["apple", "banana", "cherry"]]
            },
            {
                "id": "standalone",
                "type": "python_function",
                "module_path": self.functions_path,
                "function_name": "standalone_function",
                "input": "input_data",
                "mode": "ignore"
            }
        ]
        
        config_path = self.create_dag_config(nodes)
        dag = WordcelDAG(config_path)
        results = dag.execute()
        
        self.assertEqual(results["standalone"], "standalone result")

    def test_invalid_mode(self):
        """Test error handling for invalid mode"""
        nodes = [{
            "id": "invalid_mode",
            "type": "python_function",
            "module_path": self.functions_path,
            "function_name": "process_data",
            "mode": "invalid_mode"
        }]
        
        config_path = self.create_dag_config(nodes)
        
        with self.assertRaises(AssertionError):
            dag = WordcelDAG(config_path)
            dag.execute()

    def test_missing_input_kwarg(self):
        """Test error handling for missing input_kwarg in kwarg mode"""
        nodes = [{
            "id": "missing_input_kwarg",
            "type": "python_function",
            "module_path": self.functions_path,
            "function_name": "process_data",
            "mode": "kwarg"  # Missing input_kwarg
        }]
        
        config_path = self.create_dag_config(nodes)
        
        with self.assertRaises(AssertionError):
            dag = WordcelDAG(config_path)
            dag.execute()

    def test_invalid_function(self):
        """Test error handling for invalid function"""
        nodes = [{
            "id": "invalid_function",
            "type": "python_function",
            "module_path": self.functions_path,
            "function_name": "nonexistent_function"
        }]
        
        config_path = self.create_dag_config(nodes)
        
        with self.assertRaises(ValueError):
            dag = WordcelDAG(config_path)
            dag.execute()

    def tearDown(self):
        # Clean up temporary files
        import shutil
        shutil.rmtree(self.test_dir)

if __name__ == '__main__':
    unittest.main()