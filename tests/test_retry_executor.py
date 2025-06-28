import os
import pytest
from unittest.mock import MagicMock, call
from wordcel.dag import WordcelDAG
from wordcel.dag.nodes import Node

class FailingNode(Node):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attempts = 0

    def execute(self, input_data: any) -> any:
        self.attempts += 1
        if self.attempts < 3:
            raise ValueError("Intentional failure")
        return "Success"

    def validate_config(self) -> bool:
        return True

@pytest.fixture
def dag_with_failing_node():
    dag_config = """
dag:
  name: retry_test_dag
nodes:
  - id: failing_node
    type: failing_node
"""
    # Create a temporary YAML file for the DAG
    test_yaml_path = "test_retry_dag.yaml"
    with open(test_yaml_path, "w") as f:
        f.write(dag_config)

    # Register the custom node
    from wordcel.dag.nodes import NodeRegistry
    NodeRegistry.register("failing_node", FailingNode)

    # Create the DAG
    dag = WordcelDAG(dag_definition=test_yaml_path)
    
    yield dag

    # Teardown: remove the temporary file
    if os.path.exists(test_yaml_path):
        os.remove(test_yaml_path)

def test_sequential_executor_retry(dag_with_failing_node):
    dag = dag_with_failing_node
    
    # Execute the DAG using the sequential executor
    results = dag.execute(executor_type="sequential")

    # Assert that the failing_node was attempted 3 times
    failing_node_instance = dag.nodes['failing_node']
    assert failing_node_instance.attempts == 3
    assert results['failing_node'] == "Success"

def test_parallel_executor_retry(dag_with_failing_node):
    dag = dag_with_failing_node
    
    # Execute the DAG using the parallel executor
    results = dag.execute(executor_type="parallel")

    # Assert that the failing_node was attempted 3 times
    failing_node_instance = dag.nodes['failing_node']
    assert failing_node_instance.attempts == 3
    assert results['failing_node'] == "Success"
