"""CLI for Wordcel."""
import os
import pprint
import click
from wordcel.dag import WordcelDAG


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
@click.argument("pipeline_file")
@click.option("--secrets", default=None, help="Path to secrets file.")
@click.option("--visualization", default=None, help="Path to save visualization.")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output.")
def execute(pipeline_file, secrets, visualization, verbose):
    """Execute a pipeline."""
    dag = WordcelDAG(pipeline_file, secrets)
    results = dag.execute()

    if verbose:
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint("DAG Execution Results:")
        pp.pprint(results)

    if visualization:
        dag.save_image(visualization)


if __name__ == "__main__":
    main()
