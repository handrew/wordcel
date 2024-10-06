<p align="center">
  <img src="../assets/tree_of_life.jpg" height="400" />
</p>

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

### Running in Python

1. Define your DAG in a YAML file.
2. Create a WordcelDAG instance.
3. Execute the DAG.

```python
from wordcel.dag import WordcelDAG

dag = WordcelDAG("path/to/your/dag.yaml", "path/to/your/secrets.yaml")
results = dag.execute()
```

### Running with the CLI

There is a CLI! `wordcel dag --help`:

```
Usage: wordcel dag [OPTIONS] COMMAND [ARGS]...

  WordcelDAG commands.

Options:
  --help  Show this message and exit.

Commands:
  execute  Execute a pipeline.
  new      Create a new pipeline.
```

## DAG Configuration (YAML)

The DAG is defined in a YAML file with the following structure:

```yaml
dag:
  name: "Your DAG Name"
  backend:
    type: local
    cache_dir: path/to/your/cache/folder

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

Some examples can be found in the `tests` folder.

General optional parameters for all nodes:
- id: A unique identifier for the node (if not provided, it will be auto-generated).
- inputs: List of input node ids (for nodes that accept input from other nodes).

### `csv` CSVNode

Required:
- `path`: The file path to the CSV file.

Optional:
- `None` specific to this node.

### `sql` SQLNode

Required:
- `query`: The SQL query to execute.

Optional:
- `None` specific to this node, but requires database connection details in secrets.

### `llm` LLMNode

Required:
- `template`: The prompt template for the LLM.

Optional:
- `input_column`: The column name to use as input when processing a DataFrame.
- num_threads: Number of threads for parallel processing (default: 1).

### `llm_filter` LLMFilterNode

Required:
- `column`: The column to apply the filter on.
- prompt: The prompt to use for filtering.
- input: The input node (must be a single input).

Optional:
- `num_threads`: Number of threads for parallel processing (default: 1).

### `file_writer` FileWriterNode

Required:
- `path`: The file path to write the output.

Optional:
- `None` specific to this node.

### `dataframe_operation` DataFrameOperationNode

Required:
- `operation`: The DataFrame operation to perform.

Optional:
- `args`: List of positional arguments for the operation.
- kwargs: Dictionary of keyword arguments for the operation.

### `python_script` PythonScriptNode

Required:
- `script_path`: The path to the Python script to execute.

Optional:
- `args`: List of command-line arguments to pass to the script.

### `dag` DAGNode

Required:
- `path`: The path to the YAML file defining the sub-DAG.

Optional:
- `secrets_path`: The path to the secrets file for the sub-DAG.



## Defining Custom Nodes

To create a custom node type:

1. Create a new class that inherits from the `Node` base class
2. Implement the `execute` and `validate_config` methods
3. Register the custom node type

Example:

```python
from wordcel.dag import Node, NodeRegistry

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

## Backends

WordcelDAG supports the use of backends for caching node results. This can significantly speed up repeated executions of the DAG by avoiding redundant computations.

### Using Backends

To use a backend, specify it in your DAG configuration YAML file:

```yaml
dag:
  name: "Your DAG Name"
  backend:
    type: "local"
    cache_dir: "/path/to/cache/directory"

nodes:
  # ... node definitions ...
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

- Secrets Management: Use a separate YAML file for sensitive information.
- Custom Functions: Pass custom functions to be used in nodes.
- Backends: Use backends to cache node results and speed up repeated executions.
- DAG Visualization: Use `dag.save_image("path/to/image.png")` to visualize your DAG.
