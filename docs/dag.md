<p align="center">
  <img src="../assets/tree_of_life.jpg" height="400" />
</p>

# WordcelDAG Documentation

## Overview

WordcelDAG is a flexible framework for defining and executing Directed Acyclic Graphs (DAGs) of data processing tasks, particularly involving LLMs and dataframes. 

There are plenty of great Pythonic DAG execution frameworks out there. Metaflow is great for data science pipelines; Prefect, Luigi, and Dagster for data pipelines; I really liked ControlFlow for agentic workflows. Rivet by Ironclad was probably the closest thing to what I wanted, but it didn't have great support for Python, and you had to use a visual canvas (as with Flowise, LangFlow, etc).

While great projects in their own right, none of the above quite provided what I was looking for. Accordingly, the underlying motivation for WordcelDAG was to create a DAG framework with a few things in mind: 
1. YAML as a first-class citizen. I didn't want to be writing and maintaining Python, or drawing on a visual canvas.
2. Making it easy to call and chain LLMs.
3. Support for working with dataframes.

### Core Concepts

- **Nodes**: The fundamental building blocks of a WordcelDAG. Each node represents a specific task or operation in the data processing pipeline.
  - **Node types**: Pre-defined and user-defined (custom) node types that encapsulate specific computations (e.g., CSV loading, SQL queries, LLM operations).
- **Edges**: Implicit connections between nodes, defined by the 'input' parameter in DAG configuration file. They determine the flow of data through the DAG.
- **DAG structure and execution flow**: The overall structure of tasks, ensuring a directed and acyclic flow of operations. Nodes are executed in an order that respects their dependencies.
- **YAML configuration**: The primary method for defining DAGs for readability and ease of maintenance.
- **Caching and backends**: Pre-defined and user-defined systems for storing and retrieving intermediate results to optimize repeated executions.
- **Secrets management**: Handling of sensitive information required for DAG execution.


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

You can also give input data to your DAG at runtime.

```python
results = dag.execute(input_data={"node_id": data})
```

where `node_id` is the node that `data` is intended for. 

### Quick Start with the CLI

There is a CLI! `wordcel dag --help`:

```
Usage: wordcel dag [OPTIONS] COMMAND [ARGS]...

  WordcelDAG commands.

Options:
  --help  Show this message and exit.

Commands:
  execute          Execute a pipeline.
  list-node-types  List available node types.
  new              Create a new pipeline.
  visualize        Visualize a pipeline.
```

1. Start by creating a new YAML file with `wordcel dag new your_yaml_file.yaml`.

2. Edit the resulting yaml file according to your needs. You can use `wordcel dag visualize your_yaml_file.yaml visualization.png --custom-nodes /path/to/your/nodes.py` to save an image of your DAG to inspect visually.

3. Once you are satisfied, you can run `wordcel dag execute your_yaml_file.yaml --verbose` to run the DAG and get a nice output.

The `--config` param for `execute` can be used to do a simple string template substitution in the YAML, in case you want to be able to pass in variables in the CLI at runtime. You might use:

`wordcel dag execute pipeline.yaml -c key_to_replace your_value -c another_key another_value` to substitute `"${key_to_replace}"` and `"${another_key}"` wherever it is found in the `pipeline.yaml` file. (Note the `${}` templating with the dollar sign, as it is different than the `{}` templating done at the `LLMNode`, and the same templating as with `StringTemplateNode`. Make sure that your `runtime_config_params` do not overlap with the placeholders in the `StringTemplateNode`s).

Similarly `--input` or `-i` can be used to give values to the DAG as if you were running `.execute(input_data=input_data)` in Python code, though it is not very ergonomic to give more complex values.


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

General parameters for all nodes:
- id (required): A unique identifier for the node (if not provided, it will be auto-generated).
- type (required): Defines what type the node is.
- inputs (optional): List of input node ids (for nodes that accept input from other nodes). Can be none.

### `csv` CSVNode

Returns a pandas DataFrame.

Required:
- `path`: The file path to the CSV file.

Optional:
- None specific to this node.

Input data:
- Does not use any input data.

```yaml
- id: load_csv_data
  type: csv
  path: /path/to/your/data.csv
```

### `yaml` YAMLNode

Required:
- `path`: The file path to the JSON file.

Optional:
- None specific to this node.

Input data:
- Does not use any input data.

```yaml
- id: load_json_data
  type: json
  path: /path/to/your/data.json
```


### `json` JSONNode

Returns a `dict`.

Required:
- `path`: The file path to the JSON file.

Optional:
- None specific to this node.

Input data:
- Does not use any input data.

```yaml
- id: load_json_data
  type: json
  path: /path/to/your/data.json
```

### `json_dataframe` JSONDataFrameNode

This is just a wrapper over `pd.read_json`, so whatever works for `read_json` will work here too.

Required:
- `path`: The file path to the JSON file.

Optional:
- `read_json_kwargs`: A dictionary of keyword arguments to pass to `pd.read_json()`.

Input data:
- Does not use any input data.

```yaml
- id: load_json_as_dataframe
  type: json_dataframe
  path: /path/to/your/data.json
  read_json_kwargs:
    orient: records
```

### `sql` SQLNode

Returns a pandas DataFrame.

Required:
- `query`: The SQL query to execute.

Optional:
- `None` specific to this node, but requires database connection details in secrets.

Input data:
- Does not use any input data.

```yaml
- id: execute_sql_query
  type: sql
  query: "SELECT * FROM users WHERE location = 'SF'"
```


### `string_template` StringTemplateNode

Constructs a string of the form:
```
# Header

{template}
```
where `template` is repeated for each item, if given a list of dictionaries, pandas DataFrame (each row treated as a dict).


Required:
- template: String template using `${keyword}` format.

Optional:
- None specific to this node. 

Input data:
- Expects input_data to be None, a dictionary, DataFrame, or a list of dictionaries. If None, you're just passing a string.

```yaml
- id: format_string
  type: string_template
  header: "# Header that is printed once"
  template: "Hello, ${name}! You are ${age} years old."
  input: previous_node_id
```

### `llm` LLMNode

Returns a string or list, depending on what is given.

Required:
- `template`: The prompt template for the LLM.

Optional:
- `key`: The column name (if given a DataFrame) or field (if given dicts) to use as input when processing a DataFrame.
- `model`: Which model to use. Supported models can be found in `wordcel.llms`, but is generally limited to OpenAI, Anthropic, Gemini.
- `num_threads`: Number of threads for parallel processing (default: 1).

Input data:
- Handles str, list of strings, or pandas DataFrame.

```yaml
- id: generate_summary
  type: llm
  template: "Summarize the following text: {input}"
  input: previous_node_id
  key: text_column
  num_threads: 4
```

### `llm_filter` LLMFilterNode

Returns a pandas DataFrame, slimmed down from what was given. You *must* ask a yes or no question and instruct the LLM to answer with yes or no. 

Required:
- `column`: The column to apply the filter on.
- `prompt`: The prompt to use for filtering.
- `input`: The input node (must be a single input).

Optional:
- `model`: Which model to use. Supported models can be found in `wordcel.llms`, but is generally limited to OpenAI, Anthropic, Gemini.
- `num_threads`: Number of threads for parallel processing (default: 1).

Input data:
- Handles pandas DataFrame only.

```yaml
- id: filter_content
  type: llm_filter
  input: previous_node_id
  column: content
  prompt: "Is this content suitable for all ages? Answer only Yes or No."
  num_threads: 2
```

### `file_writer` FileWriterNode

Returns None, simply writes the file to the given path.

Required:
- `path`: The file path to write the output.

Optional:
- `None` specific to this node.

Input data:
- String only.

```yaml
- id: save_results
  type: file_writer
  input: previous_node_id
  path: /path/to/output/results.txt
```

### `dataframe_operation` DataFrameOperationNode

This is just a wrapper over `pd.DataFrame`, so anything that a pandas DataFrame can accept can be used here too. Also supports `pd.concat` and `pd.merge` if given a list of DataFrames in the input_data. There is also a `set_column` operation as a wrapper around `df["column_name"] = series`, which requires a `column_name`, shown below.


Required:
- `operation`: The DataFrame operation to perform.

Optional:
- `args`: List of positional arguments for the operation.
- `kwargs`: Dictionary of keyword arguments for the operation.

Input data:
- DataFrame or list of DataFrames.


```yaml
- id: process_dataframe
  type: dataframe_operation
  input: previous_node_id
  operation: groupby
  args: ["category"]
  kwargs:
    as_index: false
```

Example with `set_column`:

```yaml
- id: process_dataframe
  type: dataframe_operation
  input: [dataframe_from_previous_node, series_or_list_from_previous_node]
  operation: set_column
  column_name: "new_col_name" 
```

which is a wrapper on 
```python
dataframe_from_previous_node[new_col_name] = series_or_list_from_previous_node
```

### `python_script` PythonScriptNode

Either returns a list consisting of attempts to read JSON from `stdout` from each execution, or from the `return_output_file` if given. Otherwise returns an empty list.

Required:
- `script_path`: The path to the Python script to execute.

Optional:
- `args`: List of command-line arguments to pass to the script.
- `return_stdout` (bool): Attempts to read JSON from whatever is printed to stdout.
- `return_output_file` (str) or `return_stdout` (bool): Either the script must save its output to `return_output_file` or print a JSON-serializable string to stdout.

Input data:
- List of strings only, which are converted to command line arguments.


```yaml
- id: run_custom_script
  type: python_script
  script_path: /path/to/your/script.py
  args: ["arg1", "arg2"]
```

### `dag` DAGNode

Returns a `dict` of DAG results.

Required:
- `path`: The path to the YAML file defining the sub-DAG.

Optional:
- `secrets_path`: The path to the secrets file for the sub-DAG.

Input data:
- None or a dictionary, simialr to how you might use `dag.execute`.

```yaml
- id: sub_dag
  type: dag
  path: /path/to/sub_dag.yaml
  secrets_path: /path/to/sub_dag_secrets.yaml
```


## Defining Custom Functions

To create custom functions, simply pass in a dictionary of functions to the constructor. You can them use them in the YAML by indexing the key.

```python
from wordcel.dag.utils import create_custom_functions_from_files

custom_functions = create_custom_functions_from_files([file_path])
dag = WordcelDAG("path/to/your/dag.yaml", custom_nodes=custom_functions)
```


## Defining Custom Nodes

To create a custom node type:

1. Create a new class that inherits from the `Node` base class
2. Implement the `execute` and `validate_config` methods
3. Pass the custom node type to the constructor. 

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

There is also a utility function to automatically create a dict of custom nodes from a file.

```python
from wordcel.dag.utils import create_custom_nodes_from_files

custom_nodes = create_custom_nodes_from_files([file_path])
dag = WordcelDAG("path/to/your/dag.yaml", custom_nodes=custom_nodes)
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

Like with custom nodes, you can create custom backends by creating a class that inherits from the Backend class and overriding `save`, `load`, and `exists`. 

```python
from wordcel.dag.utils import create_custom_backends_from_files

custom_backends = create_custom_backends_from_files([file_path])
dag = WordcelDAG("path/to/your/dag.yaml", custom_backends=custom_backends)
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
    key: "Country"

  - id: save_results
    type: file_writer
    path: "test_output.txt"
    input: process_filtered
```

## Other Features

- Secrets management: Use a separate YAML file for sensitive information.
- Custom functions: Pass custom functions to be used in nodes.
- Backends: Use backends to cache node results and speed up repeated executions.
- DAG visualization: Use `dag.save_image("path/to/image.png")` to visualize your DAG.
