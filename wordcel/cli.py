"""CLI for Wordcel."""
import os
import importlib.util
import click
from rich import print

PIPELINE_TEMPLATE = """
dag:
  name: "Your DAG Name"
#   backend:
#     type: local
#     cache_dir: path/to/your/cache/folder

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

#   - id: save_results
#     type: file_writer
#     path: "example_output.txt"
#     input: process_filtered
"""


def load_module(file_path, module_name):
    """Load a Python module from a file."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@click.group()
def main():
    """Wordcel CLI."""
    pass


@main.group()
def dag():
    """WordcelDAG commands."""
    pass


@dag.command()
@click.argument("pipeline_file")
def new(pipeline_file):
    """Create a new pipeline."""
    if os.path.exists(pipeline_file):
        click.echo(f"Pipeline file {pipeline_file} already exists.")
        return

    with open(pipeline_file, "w") as f:
        f.write(PIPELINE_TEMPLATE)

    click.echo(f"Pipeline file {pipeline_file} created.")


@dag.command()
def list_node_types():
    """List available node types."""
    from wordcel.dag.nodes import NODE_TYPES

    for node_type, node_class in NODE_TYPES.items():
        click.echo(f"- {node_type}: {node_class.description}")


@dag.command()
@click.argument("pipeline_file")
@click.option("--secrets", default=None, help="Path to secrets file.")
@click.option("--custom-nodes", default=None, help="Path to custom nodes Python file.")
@click.option(
    "--custom-functions", default=None, help="Path to custom functions Python file."
)
@click.option(
    "--custom-backends", default=None, help="Path to custom backends Python file."
)
@click.option("--visualization", default=None, help="Path to save visualization.")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output.")
def execute(
    pipeline_file,
    secrets,
    custom_nodes,
    custom_functions,
    custom_backends,
    visualization,
    verbose,
):
    """Execute a pipeline."""
    from wordcel.dag import WordcelDAG
    from wordcel.dag.nodes import Node
    from wordcel.dag.backends import Backend

    if custom_nodes:
        nodes_module = load_module(custom_nodes, "custom_nodes")
        custom_nodes = {name: cls for name, cls in nodes_module.__dict__.items() if isinstance(cls, type) and issubclass(cls, Node) and cls is not Node}
        print("Created custom nodes:")
        print(custom_nodes)

    if custom_functions:
        functions_module = load_module(custom_functions, "custom_functions")
        custom_functions = {name: func for name, func in functions_module.__dict__.items() if callable(func)}
        print("Created custom functions:")
        print(custom_functions)

    if custom_backends:
        backends_module = load_module(custom_backends, "custom_backends")
        custom_backends = {name: cls for name, cls in backends_module.__dict__.items() if isinstance(cls, type) and issubclass(cls, Backend) and cls is not Backend}
        print("Created custom backends:")
        print(custom_backends)

    dag = WordcelDAG(
        pipeline_file,
        secrets_file=secrets,
        custom_nodes=custom_nodes,
        custom_functions=custom_functions,
        custom_backends=custom_backends,
    )
    results = dag.execute()

    if verbose:
        print("DAG Execution Results:")
        print(results)

    if visualization:
        dag.save_image(visualization)


if __name__ == "__main__":
    main()
