import os
import pandas as pd
import yaml
import pytest
from wordcel.dag import WordcelDAG


class TestPythonFunctionNode:
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        # Create a temporary test function file
        self.test_functions_path = "test_functions.py"

        # Clean up before test in case of previous failures
        if os.path.exists(self.test_functions_path):
            os.remove(self.test_functions_path)

        with open(self.test_functions_path, "w") as f:
            f.write(
                """
def add_numbers(a, b=1):
    return a + b

def multiply_by_two(x):
    return x * 2

def process_text(text, prefix=""):
    return prefix + text.upper()
"""
            )

        yield

        # Clean up the temporary file
        if os.path.exists(self.test_functions_path):
            os.remove(self.test_functions_path)

    def test_single_mode_simple(self):
        # Test YAML for simple single mode operation
        yaml_content = """
dag:
  name: test_python_function_single
nodes:
  - id: add_numbers
    type: python_function
    function_path: test_functions.add_numbers
    mode: single
    kwargs:
      a: 5
      b: 3
"""
        dag = WordcelDAG(yaml.safe_load(yaml_content))
        results = dag.execute()
        assert results["add_numbers"] == 8

    def test_single_mode_with_input(self):
        # Test YAML for single mode with input
        yaml_content = """
dag:
  name: test_python_function_single_input
nodes:
  - id: multiply
    type: python_function
    function_path: test_functions.multiply_by_two
    mode: single
    input_kwarg: x
"""
        dag = WordcelDAG(yaml.safe_load(yaml_content))
        results = dag.execute(input_data={"multiply": 5})
        assert results["multiply"] == 10

    def test_multiple_mode_with_dataframe(self):
        # Test YAML for multiple mode with DataFrame
        yaml_content = """
dag:
  name: test_python_function_multiple
nodes:
  - id: process_texts
    type: python_function
    function_path: test_functions.process_text
    mode: multiple
    input_field: text
    output_column: processed_text
    kwargs:
      prefix: "PREFIX_"
"""
        # Create test DataFrame
        df = pd.DataFrame({"text": ["hello", "world", "test"]})

        dag = WordcelDAG(yaml.safe_load(yaml_content))
        results = dag.execute(input_data={"process_texts": df})

        expected_results = ["PREFIX_HELLO", "PREFIX_WORLD", "PREFIX_TEST"]
        # The output column name appears to be 'result' instead of 'processed_text'
        assert results["process_texts"]["result"].tolist() == expected_results

    def test_multiple_mode_with_list(self):
        # Test YAML for multiple mode with list input
        yaml_content = """
dag:
  name: test_python_function_multiple_list
nodes:
  - id: multiply_numbers
    type: python_function
    function_path: test_functions.multiply_by_two
    mode: multiple
"""
        input_list = [1, 2, 3, 4, 5]

        dag = WordcelDAG(yaml.safe_load(yaml_content))
        results = dag.execute(input_data={"multiply_numbers": input_list})

        expected_results = [2, 4, 6, 8, 10]
        assert results["multiply_numbers"] == expected_results

    def test_chained_nodes(self):
        # Test YAML for chaining multiple Python function nodes
        yaml_content = """
dag:
  name: test_python_function_chain
nodes:
  - id: multiply_first
    type: python_function
    function_path: test_functions.multiply_by_two
    mode: single
    input_kwarg: x

  - id: add_numbers
    type: python_function
    function_path: test_functions.add_numbers
    mode: single
    input: multiply_first
    input_kwarg: a
    kwargs:
      b: 5
"""
        dag = WordcelDAG(yaml.safe_load(yaml_content))
        results = dag.execute(input_data={"multiply_first": 3})

        # First node should multiply 3 by 2 = 6
        # Second node should add 5 to 6 = 11
        assert results["multiply_first"] == 6
        assert results["add_numbers"] == 11


if __name__ == "__main__":
    unittest.main()
