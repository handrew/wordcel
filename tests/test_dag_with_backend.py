import os
import yaml
from wordcel.dag import WordcelDAG

# Create a temporary directory for our tests
test_dir = "."
cache_dir = os.path.join(test_dir, "cache")

# Create a simple DAG configuration
dag_config = {
    'dag': {
        'name': 'test_dag',
        'backend': {
            'type': 'local',
            'cache_dir': cache_dir
        }
    },
    'nodes': [
        {
            'id': 'csv_node',
            'type': 'csv',
            'path': os.path.join(test_dir, 'test.csv')
        },
        {
            'id': 'operation_node',
            'type': 'dataframe_operation',
            'input': 'csv_node',
            'operation': 'head',
            'kwargs': {'n': 2}
        }
    ]
}

# Create a test CSV file
with open(os.path.join(test_dir, 'test.csv'), 'w') as f:
    f.write('col1,col2\n1,a\n2,b\n3,c')

# Write DAG configuration to a YAML file
dag_yaml_path = os.path.join(test_dir, 'dag.yaml')
with open(dag_yaml_path, 'w') as f:
    yaml.dump(dag_config, f)

# Test DAG with backend
print("\nTesting DAG with backend...")
dag = WordcelDAG(dag_yaml_path)
results = dag.execute()

# Test DAG node update
print("\nTesting DAG node update...")
first_results = dag.execute()

# Modify the CSV file
with open(os.path.join(test_dir, 'test.csv'), 'w') as f:
    f.write('col1,col2\n4,d\n5,e\n6,f')

new_results = dag.execute()

print("First 'csv_node' results:")
print(first_results['csv_node'].head())
print("\nNew 'csv_node' results:")
print(new_results['csv_node'].head())

print("\nFirst 'operation_node' results:")
print(first_results['operation_node'])
print("\nNew 'operation_node' results:")
print(new_results['operation_node'])
