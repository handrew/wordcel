<p align="center">
  <img src="../assets/tree_of_life.jpg" height="400" />
</p>

# WordcelDAG Documentation

## Overview

WordcelDAG is a flexible framework for defining and executing Directed Acyclic Graphs (DAGs) of data processing tasks, particularly involving LLMs and dataframes.

There are plenty of great Pythonic DAG execution frameworks out there. Metaflow is great for data science pipelines; Prefect, Luigi, and Dagster for data pipelines; I really liked ControlFlow and Burr for agentic workflows. Rivet by Ironclad was probably the closest thing to what I wanted, but it didn't have great support for Python, and you had to use a visual canvas (as with Flowise, LangFlow, etc).

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


The full constructor for WordcelDAG accepts the following parameters:

```
dag_definition (Union[str, Dict[str, Any]]):
    Path to the YAML file containing the DAG definition, or a dict of the DAG itself.

secrets (Union[str, Dict[str, Any]], optional):
    Path to a YAML file containing secrets/credentials or a dict of the secrets. Defaults to None.

runtime_config_params (Dict[str, str], optional):
    Dictionary of configuration parameters that can be passed at runtime.
    These override any configurations defined in the YAML file. Defaults to None.

custom_functions (Dict[str, Callable], optional):
    Dictionary mapping function names to custom Python functions that can be used
    in the DAG. Defaults to None.

custom_nodes (Dict[str, Type[Node]], optional):
    Dictionary mapping node types to custom Node class implementations.
    Use this to extend the DAG with custom node types. Defaults to None.

custom_backends (Dict[str, Type[Backend]], optional):
    Dictionary mapping backend names to custom Backend class implementations.
    Use this to add support for custom execution backends. Defaults to None.
```

You can also use `wordcel.dag.utils.initialize_dag` below which can initialize a `WordcelDAG` with custom nodes, functions, or backends using file paths.

```python
def initialize_dag(
    pipeline_file,       # str.
    config_params=None,  # Dict.
    secrets=None,   # str.
    custom_nodes=None,   # List of file paths, or single file path.
    custom_functions=None,  # List of file paths, or single file path.
    custom_backends=None,   # List of file paths, or single file path.
):
```


### Quick Start with the CLI

There is a powerful CLI available. Get started with `wordcel dag --help`:

```
Usage: wordcel dag [OPTIONS] COMMAND [ARGS]...

  🔗 WordcelDAG commands for pipeline management

  Create, execute, and visualize LLM processing pipelines using
  declarative YAML configuration files.

Options:
  --help  Show this message and exit.

Commands:
  describe         🔎 Show detailed information about a specific node type
  dryrun           🔍 Show what would be executed without running the pipeline
  execute          Execute a pipeline.
  list-node-types  📋 List all available node types with descriptions
  new              📝 Create a new pipeline configuration file
  visualize        Visualize a pipeline.
```

1.  **Create a new pipeline file:** Start by creating a new YAML file with a template.
    `wordcel dag new your_pipeline.yaml --template basic`

2.  **Inspect node types:** See what's available to use in your pipeline.
    `wordcel dag list-node-types`

3.  **Get node details:** Get detailed information about a specific node.
    `wordcel dag describe llm`

4.  **Visualize your DAG:** Before running, you can save an image of your DAG to inspect it visually.
    `wordcel dag visualize your_pipeline.yaml visualization.png`

5.  **Dry run your pipeline:** Validate the configuration and see the execution plan without running any tasks. This is highly recommended.
    `wordcel dag dryrun your_pipeline.yaml`

6.  **Execute the pipeline:** Once you are satisfied, run the DAG.
    `wordcel dag execute your_pipeline.yaml --verbose`

You can pass variables to your pipeline at runtime using `--config-param` and `--input`:

-   `--config-param` (`-c`): Substitutes variables in the YAML file. For example, `wordcel dag execute pipeline.yaml -c key_to_replace your_value` will replace all instances of `${key_to_replace}` in the YAML file with `your_value`.
-   `--input` (`-i`): Provides input data directly to a node. For example, `wordcel dag execute pipeline.yaml -i node_id "some value"` is the CLI equivalent of `dag.execute(input_data={"node_id": "some value"})`.

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
- id (required): A unique identifier for the node.
- type (required): Defines what type the node is.
- input (optional): A single node ID or a list of node IDs that this node depends on.

### `csv` CSVNode

Returns a pandas DataFrame.

Required:
- `path`: The file path to the CSV file.

```yaml
- id: load_csv_data
  type: csv
  path: /path/to/your/data.csv
```

### `yaml` YAMLNode

Returns a dictionary.

Required:
- `path`: The file path to the YAML file.

```yaml
- id: load_yaml_data
  type: yaml
  path: /path/to/your/data.yaml
```


### `json` JSONNode

Returns a `dict`.

Required:
- `path`: The file path to the JSON file.

```yaml
- id: load_json_data
  type: json
  path: /path/to/your/data.json
```

### `json_dataframe` JSONDataFrameNode

A wrapper over `pd.read_json`. Returns a pandas DataFrame.

Required:
- `path`: The file path to the JSON file.

Optional:
- `read_json_kwargs`: A dictionary of keyword arguments to pass to `pd.read_json()`.

```yaml
- id: load_json_as_dataframe
  type: json_dataframe
  path: /path/to/your/data.json
  read_json_kwargs:
    orient: records
```

### `file_directory` FileDirectoryNode

Reads txt, md, and html files from a directory or list of directories, supporting regex patterns. Returns a DataFrame with columns: `file_path`, `content`, `file_type`.

Required:
- `path`: A string or list of strings representing file paths or glob patterns.

Input data:
- Can optionally receive a dictionary with a `path` key to override the path in the config.

```yaml
- id: read_dir
  type: file_directory
  path: "docs/**/*.md"
```


### `sql` SQLNode

Returns a pandas DataFrame.

Required:
- `query`: The SQL query to execute.
- A `database_url` must be provided in the secrets file.

```yaml
- id: execute_sql_query
  type: sql
  query: "SELECT * FROM users WHERE location = 'SF'"
```


### `string_template` StringTemplateNode

Constructs a string by substituting placeholders in a template.

Required:
- `template`: String template using `${placeholder}` format.

Optional:
- `header`: A string to prepend to the output, followed by two newlines.
- `mode`: "single" (default) or "multiple". In "single" mode, it combines all results into one string. In "multiple" mode, it returns a list of strings.

Input data:
- **Dictionary:** Used for a single substitution (e.g., `template: "Hello, ${name}"`).
- **List or DataFrame:** Iterates over each item. In "single" mode, concatenates the results. In "multiple" mode, returns a list of results.
- **String:** Substituted for the `{input}` placeholder.
- **Multiple Nodes:** If `input` is a list of node IDs, the results are combined into a dictionary mapping node IDs to their results, which can then be used in the template (e.g., `template: "Data from ${nodeA} and ${nodeB}"`).

```yaml
# Example with multiple inputs mapped by ID
- id: format_greeting
  type: string_template
  template: "Hello, ${name_node}! Welcome to ${place_node}."
  input:
    - name_node
    - place_node
```

### `string_concat` StringConcatNode

Concatenates strings with an optional separator, prefix, and suffix.

Optional:
- `separator`: String to insert between concatenated strings. Default: " "
- `prefix`: String to add at the beginning. Default: ""
- `suffix`: String to add at the end. Default: ""
- `column`: Required if the input is a DataFrame; specifies which column to use.

Input data:
- A single string, a list of strings, a pandas Series, or a pandas DataFrame.

```yaml
- id: fancy_concat
  type: string_concat
  separator: ", "
  prefix: "Items: ["
  suffix: "]"
  column: "text"  # only needed for DataFrame input
  input: dataframe_node
```

### `llm` LLMNode

Queries an LLM API. The output type generally mirrors the input type (e.g., DataFrame in, modified DataFrame out).

Required:
- `template`: The prompt template for the LLM. Must contain `{input}`.

Optional:
- `input_field`: The column/field to use as input when given a DataFrame or list of dicts.
- `output_field`: The column/field to store the LLM's output. Required if `input_field` is used.
- `model`: The model to use (e.g., "openai/gpt-4o").
- `num_threads`: Number of threads for parallel processing (default: 1).
- `web_search_options`: A dictionary of options for web searching.

Input data:
- Handles `str`, `list` of strings, `dict`, `pd.Series`, or `pd.DataFrame`.

```yaml
- id: generate_summary
  type: llm
  template: "Summarize the following text: {input}"
  input: previous_node_id
  input_field: "text_column"
  output_field: "summary_column"
  num_threads: 4
```

### `llm_filter` LLMFilterNode

Filters a DataFrame based on a "Yes" or "No" response from an LLM.

Required:
- `column`: The column to apply the filter on.
- `prompt`: The prompt for the LLM. It must ask a question that can be answered with "Yes" or "No".
- `input`: A single input node that produces a DataFrame.

Optional:
- `model`: The model to use.
- `num_threads`: Number of threads for parallel processing (default: 1).

```yaml
- id: filter_content
  type: llm_filter
  input: previous_node_id
  column: "content"
  prompt: "Is this content suitable for all ages? Answer only Yes or No."
```

### `file_writer` FileWriterNode

Writes data to a file. Does not return any data.

Required:
- `path`: The file path for the output.

Optional:
- `mode`: "single" (default) or "multiple". If "multiple", the `path` must contain `{i}` as a placeholder for the index.

Input data:
- A string, list, or DataFrame.

```yaml
- id: save_results
  type: file_writer
  input: previous_node_id
  path: "output/results.txt"
```

### `dataframe_operation` DataFrameOperationNode

Performs an operation on a pandas DataFrame. This is a powerful node that can call any DataFrame method. It also supports `pd.concat` and `pd.merge`.

Required:
- `operation`: The DataFrame operation to perform (e.g., "groupby", "head", "concat", "merge", "set_column").

Optional:
- `args`: List of positional arguments for the operation.
- `kwargs`: Dictionary of keyword arguments for the operation.
- `column_name`: Required for the `set_column` operation.

Input data:
- A DataFrame or a list of DataFrames.

```yaml
# Example with set_column
- id: add_new_column
  type: dataframe_operation
  input: [my_dataframe, my_series]
  operation: set_column
  column_name: "new_column"
```

### `python_script` PythonScriptNode

Executes an external Python script.

Required:
- `script_path`: The path to the Python script.

Optional:
- `args`: List of command-line arguments to pass to the script.
- `return_stdout`: (boolean) If true, captures and returns the script's standard output.
- `return_output_file`: (string) If provided, reads and returns the content of this file after the script runs.

Input data:
- Can be a primitive type, list, Series, or DataFrame. The input is converted to a string and passed as the final command-line argument to the script.

```yaml
- id: run_custom_script
  type: python_script
  script_path: /path/to/your/script.py
  args: ["--mode", "fast"]
  return_stdout: true
```


### `python_function` PythonFunctionNode

Executes a specific Python function from a module.

Required:
- `function_path`: The dotted path to the function (e.g., `my_module.my_function`).

Optional:
- `mode`: "single" (default) or "multiple".
- `args`: List of positional arguments for the function.
- `kwargs`: Dictionary of keyword arguments for the function.
- `input_kwarg`: If provided, the input data will be passed as a keyword argument with this name.
- `input_field`: Required for DataFrame input in `multiple` mode.
- `output_field`: The column name for the results when the input is a DataFrame in `multiple` mode.

Input data:
- Any type. Behavior is controlled by `mode`.

```yaml
# Example with DataFrame
- id: analyze_reviews
  type: python_function
  function_path: sentiment.analyze_sentiment
  mode: multiple
  input_field: "review_text"
  output_field: "sentiment"
  input: reviews_dataframe
```


### `dag` DAGNode

Executes a sub-DAG.

Required:
- `path`: The path to the YAML file defining the sub-DAG.

Optional:
- `output_key`: A string or list of strings to select specific results from the sub-DAG's output dictionary.
- `secrets_path`: The path to the secrets file for the sub-DAG.
- `input_nodes`: A list of node IDs in the sub-DAG to which the input data should be passed.
- `runtime_config_params`: A dictionary to pass runtime configuration to the sub-DAG. This allows for dynamic configuration of sub-DAGs. You can even pass results from previous nodes.

```yaml
- id: sub_dag
  type: dag
  path: /path/to/sub_dag.yaml
  runtime_config_params:
    some_value_for_subdag: "hello from parent"
    value_from_previous_node:
      input_param: previous_node_result # Maps previous_node_result to subdag param
```

## Defining Custom Components

You can extend WordcelDAG with your own custom nodes, functions, and backends.

### Custom Functions

Create a Python file with your functions and pass the file path to the `initialize_dag` function or the `WordcelDAG` constructor.

```python
# my_functions.py
def my_custom_logic(text):
    return text.upper()
```
```python
# main.py
from wordcel.dag.utils import initialize_dag
dag = initialize_dag("pipeline.yaml", custom_functions="my_functions.py")
```

### Custom Nodes

1.  Create a class that inherits from `wordcel.dag.Node`.
2.  Implement the `execute` and `validate_config` methods.
3.  Pass a dictionary of your custom nodes to the `WordcelDAG` constructor or use the `create_custom_nodes_from_files` utility.

```python
# my_nodes.py
from wordcel.dag import Node

class MyCustomNode(Node):
    def execute(self, input_data):
        # Custom logic
        return input_data

    def validate_config(self):
        # Validation logic
        return True
```

## Backends

WordcelDAG supports backends for caching results to speed up repeated executions.

```yaml
dag:
  name: "My Caching DAG"
  backend:
    type: "local"
    cache_dir: ".my_cache"
```

You can also create custom backends by inheriting from `wordcel.dag.Backend` and implementing the `save`, `load`, and `exists` methods.

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
    input_field: "Country"
    output_field: "Cuisine"

  - id: save_results
    type: file_writer
    path: "test_output.txt"
    input: process_filtered
```
