"""CLI for Wordcel."""

import os
import logging
import click
from rich import print
from wordcel.dag.utils import create_custom_functions_from_files
from wordcel.dag.utils import create_custom_nodes_from_files
from wordcel.dag.utils import create_custom_backends_from_files

log: logging.Logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

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


def initialize_dag(
    pipeline_file,
    secrets_file=None,
    custom_nodes=None,
    custom_functions=None,
    custom_backends=None,
):
    """Initialize the DAG."""
    from wordcel.dag import WordcelDAG

    if custom_nodes:
        custom_nodes = create_custom_nodes_from_files(custom_nodes)

    if custom_functions:
        custom_functions = create_custom_functions_from_files(custom_functions)

    if custom_backends:
        custom_backends = create_custom_backends_from_files(custom_backends)

    dag = WordcelDAG(
        pipeline_file,
        secrets_file=secrets_file,
        custom_nodes=custom_nodes,
        custom_functions=custom_functions,
        custom_backends=custom_backends,
    )

    return dag


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
@click.argument("save_path")
@click.option("--custom-nodes", default=None, help="Path to custom nodes Python file.")
def visualize(pipeline_file, save_path, custom_nodes):
    """Visualize a pipeline."""
    dag = initialize_dag(pipeline_file, secrets_file=None, custom_nodes=custom_nodes)
    log.info("Saving visualization to %s.", save_path)
    dag.save_image(save_path)


@dag.command()
@click.argument("pipeline_file")
@click.option("--secrets", default=None, help="Path to secrets file.")
@click.option(
    "--custom-nodes",
    default=None,
    multiple=True,
    help="Path to custom nodes Python file.",
)
@click.option(
    "--custom-functions",
    default=None,
    multiple=True,
    help="Path to custom functions Python file.",
)
@click.option(
    "--custom-backends",
    default=None,
    multiple=True,
    help="Path to custom backends Python file.",
)
@click.option("--visualization", default=None, help="Path to save visualization.")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output.")
@click.option(
    "--input",
    "-i",
    nargs=2,
    type=(str, str),
    multiple=True,
    help="Input data for the pipeline. Given in the format `-i key value`. The key is the node ID that the input data is for.",
)
def execute(
    pipeline_file,
    secrets,
    custom_nodes,
    custom_functions,
    custom_backends,
    visualization,
    verbose,
    input,
):
    """Execute a pipeline."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.tree import Tree

    if input:
        print("Given input data: ", dict(input))
    dag = initialize_dag(
        pipeline_file, secrets, custom_nodes, custom_functions, custom_backends
    )
    results = dag.execute(input_data=dict(input))

    if verbose:
        console = Console()
        tree = Tree("⚡️ [bold blue]Execution Results")

        for node_id, result in results.items():
            node_tree = tree.add(f"[bold green]{node_id}")
            node_tree.add(f"{result}")

        console.print(Panel(tree, expand=False, border_style="bold"))

    if visualization:
        dag.save_image(visualization)


if __name__ == "__main__":
    main()
