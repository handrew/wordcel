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

### `file_directory` FileDirectoryNode

Reads txt, md, and html files from a directory or list of directories, supporting regex patterns. Returns a DataFrame of columns: `file_path`, `content`, `file_type`.

Required:
- `path`: List or string. Can be regex.

Optional:
- None specific to this node.

Input data:
- Can use input data if it is a dict with `path`.

```yaml
- id: read_dir
  type: file_directory
  path: /paths/to/your/file.txt
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
where `template` is repeated for each item, if given a list of dictionaries or pandas DataFrame (each row treated as a dict).


Required:
- template: String template using `${keyword}` format.

Optional:
- None specific to this node. 
- mode: "single" or "multiple". Default "single". If single, then it repeats the template for each item. If multiple, then it returns a list.

Input data:
- Expects input_data to be None, a dictionary, DataFrame, or a list of dictionaries. If None, you're just passing a string.

```yaml
- id: format_string
  type: string_template
  header: "# Header that is printed once"
  template: "Hello, ${name}! You are ${age} years old."
  input: previous_node_id
```


### `string_concat` StringConcatNode

Concatenates strings with optional separator, prefix, and suffix in the form:
```
{prefix}{string1}{separator}{string2}{separator}...{stringN}{suffix}
```
where each string comes from the input data. Handles various input types including single strings, lists, DataFrames, and Series.

Required:
- None. All configuration parameters are optional.

Optional:
- separator: String to insert between concatenated strings. Default: " "
- prefix: String to add at the beginning. Default: ""
- suffix: String to add at the end. Default: ""
- column: Required only when input is DataFrame - specifies which column to use

Input data:
- Single string
- List of strings
- DataFrame (requires 'column' config)
- Pandas Series

```yaml
# Basic concatenation
- id: concat_strings
  type: string_concat
  separator: " "
  input: previous_node_id

# With all options
- id: fancy_concat
  type: string_concat
  separator: ", "
  prefix: "Items: ["
  suffix: "]"
  column: "text"  # only needed for DataFrame input
  input: previous_node_id

# Examples of results:
# Input: ["apple", "banana", "orange"]
# Basic: "apple banana orange"
# Fancy: "Items: [apple, banana, orange]"

# Input: DataFrame with column 'text': ["apple", "banana", "orange"]
# Basic: "apple banana orange"
# Fancy: "Items: [apple, banana, orange]"
```

The node automatically handles:
- Converting non-string elements to strings
- Filtering out None values
- Various input types flexibly
- Proper string formatting with consistent separator/prefix/suffix application


### `llm` LLMNode

Returns a string, list, or DataFrame, depending on what is given. In general, the philosophy is to return a mutated version of what is given to the LLM Node. So, if the node is given a dict or DataFrame, then it simply updates the dict / DataFrame with its answer (according to `output_field`). The exception is for a single dict - it will simply return a string.

Required:
- `template`: The prompt template for the LLM.

Optional:
- `input_field`: The column name (if given a DataFrame or list of DataFrames) or field (if given dicts) to use as input. 
- `output_field`: The column name (if given a DataFrame or list of DataFrames) or field (if given dicts) to use as output. If DataFrame(s), then the new column will be named `output_field`. 
- `model`: Which model to use. Supported models can be found in `wordcel.llms`, but is generally limited to OpenAI, Anthropic, Gemini.
- `num_threads`: Number of threads for parallel processing (default: 1).

Input data:
- Handles str, list of strings, or pandas DataFrame.

```yaml
- id: generate_summary
  type: llm
  template: "Summarize the following text: {input}"
  input: previous_node_id
  input_field: text_column
  output_field: output_field
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
- mode: "single" or "multiple". Default "single". If single, then it attempts to write the input data to a single file. If multiple, it will write multiple files, and requires `path` to have a string format placeholder "{i}" for the index of the item. 

Input data:
- String, list, or DataFrame.

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


### `python_function` PythonFunctionNode

Executes a specific Python function from a module or script. Returns whatever the function returns.

Required:
- `function_path`: Function path should be in the format: package.module.function_name or local.module.function_name.
- `mode`: Either `single` or `multiple`. `single` by default. If given `multiple`, then will process iterables item by item.
- If the input data is a DataFrame, then must give `input_field` and `output_field`. 

Optional:
- `args`: List of positional arguments to pass to the function
- `kwargs`: Dictionary of keyword arguments to pass to the function
- `input_kwarg`: If given, gives the input data as this keyword arg.

Input data:
- Any type. Handling depends on `mode` setting.

Examples:

```yaml
# Example 1: Basic usage with a simple function
# my_functions.py:
# def add_one(x):
#     return x + 1
nodes:
  - id: add_numbers
    type: python_function
    function_path: my_functions.add_one
    mode: single
    input: 5  # Returns 6

# Example 2: Multiple mode with a list input
# text_utils.py:
# def uppercase(text):
#     return text.upper()
nodes:
  - id: uppercase_words
    type: python_function
    function_path: text_utils.uppercase
    mode: multiple
    input: ["hello", "world"]  # Returns ["HELLO", "WORLD"]

# Example 3: Using with DataFrame and specifying input/output fields
# sentiment.py:
# def analyze_sentiment(text):
#     return "positive" if "good" in text.lower() else "negative"
nodes:
  - id: analyze_reviews
    type: python_function
    function_path: sentiment.analyze_sentiment
    mode: multiple
    input_field: review_text
    output_field: sentiment
    input: reviews_dataframe  # DataFrame with 'review_text' column
```


### `dag` DAGNode

Returns a `dict` of DAG results.

```yaml
- id: sub_dag
  type: dag
  path: /path/to/sub_dag.yaml
  secrets_path: /path/to/sub_dag_secrets.yaml
```

Required:
- `path`: The path to the YAML file defining the sub-DAG.

Optional:
- `output_key`: String or list. Since a DAG returns a dict of its results, `output_key` lets us select one or more of intermediate results. If str, then simply selects the results for that key. If `list`, then it gives you a subset of the result dict.
- `secrets_path`: The path to the secrets file for the sub-DAG.
- `input_nodes`: List of `node_id`s to give the `input_data` to. 
- `runtime_config_params`: Any runtime config params you want to provide. These are the string substitutions you do for "${param}" variables at runtime, e.g., from the CLI using `-c param value`. You can either define new ones, or get it from a previous node (see example below).

```yaml
- id: sub_dag
  type: dag
  path: /path/to/sub_dag.yaml
  secrets_path: /path/to/sub_dag_secrets.yaml
  runtime_config_params:
    your_param: your param
    from_input_data: 
      input_param: subdag_param
```
If `input_param` is a key in your `input_data`, then it will grab that and map it to `subdag_param` config param. Otherwise, it will just pass the entire input_data through. The latter is useful if it's just a string, not a dict.

Input data:
- None or a dictionary, similar to how you might use `dag.execute`.


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
    input_field: "Country"
    output_field: "Cuisine"

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
