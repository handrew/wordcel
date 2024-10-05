# WordcelDAG Documentation

## Overview

WordcelDAG is a flexible and extensible framework for defining and executing Directed Acyclic Graphs (DAGs) of data processing tasks, particularly involving LLMs and dataframes. 

There are plenty of great Pythonic DAG execution frameworks out there. Metaflow is great for data science pipelines; Prefect, Luigi, and Dagster for data pipelines; I really liked ControlFlow for agentic workflows. Rivet by Ironclad was probably the closest thing to what I wanted, but it didn't have great support for Python, and you had to use a visual canvas (as with Flowise, LangFlow, etc).

While great projects in their own right, none of the above quite provided what I was looking for. The underlying motivation for WordcelDAG was to create a DAG framework with a few things in mind: 
1. YAML as a first-class citizen. I didn't want to be writing and maintaining Python, or drawing on a visual canvas.
2. Making it easy to call chain LLMs.
3. Support for working with dataframes.

## Key Features

- Define DAGs using YAML configuration files
- Support for various built-in node types (CSV, SQL, LLM, DataFrame operations, etc.)
- Extensibility through custom node types and functions
- Secrets management for sensitive information
- Visualization of DAG structure

## Basic Usage

1. Define your DAG in a YAML file.
2. Create a WordcelDAG instance.
3. Execute the DAG.

```python
from wordceldag import WordcelDAG

dag = WordcelDAG("path/to/your/dag.yaml", "path/to/your/secrets.yaml")
results = dag.execute()
```

## DAG Configuration (YAML)

The DAG is defined in a YAML file with the following structure:

```yaml
dag:
  name: "Your DAG Name"

nodes:
  - id: "node1"
    type: "csv"
    path: "path/to/your/file.csv"

  - id: "node2"
    type: "sql"
    input: "node1"
    query: "SELECT * FROM table"

  - id: "node3"
    type: "llm"
    input: "node2"
    template: "Summarize this: {input}"

  # ... more nodes ...
```

## Built-in Node Types

1. CSVNode: Read CSV files
2. SQLNode: Execute SQL queries
3. LLMNode: Query Language Models
4. LLMFilterNode: Filter data using Language Models
5. FileWriterNode: Write data to files
6. DataFrameOperationNode: Perform operations on DataFrames
7. PythonScriptNode: Execute Python scripts
8. DAGNode: Execute sub-DAGs

## Defining Custom Nodes

To create a custom node type:

1. Create a new class that inherits from the `Node` base class
2. Implement the `execute` and `validate_config` methods
3. Register the custom node type

Example:

```python
from wordceldag import Node, NodeRegistry

class MyCustomNode(Node):
    def execute(self, input_data):
        # Your custom logic here
        return processed_data

    def validate_config(self):
        # Validate your node's configuration
        return True

# Use in your DAG creation
dag = WordcelDAG("path/to/your/dag.yaml", custom_nodes={"my_custom_node": MyCustomNode})
```

## YAML Examples

### LLM-based Text Processing

```yaml
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
    args: [2]

  - id: llm_filter
    input: df_filter
    type: llm_filter
    column: "Country"
    prompt: "Is this country in Africa? Answer only Yes or No."

  - id: process_filtered
    type: llm
    template: "What cuisine is this country known for? {input}"
    input: llm_filter
    input_column: "Country"

  - id: save_results
    type: file_writer
    path: "test_output.txt"
    input: process_filtered
```

## Other Features

- Secrets Management: Use a separate YAML file for sensitive information
- Custom Functions: Pass custom functions to be used in nodes
- DAG Visualization: Use `dag.save_image("path/to/image.png")` to visualize your DAG
