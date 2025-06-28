"""CLI for Wordcel."""

import os
import click
from rich import print
from rich.console import Console
from wordcel.dag.utils import initialize_dag
from wordcel.logging_config import get_logger
from wordcel.cli_rich import RichGroup, RichCommand, show_version_info
from wordcel.interactive import start_interactive_mode

log = get_logger("cli")
console = Console()

PIPELINE_TEMPLATE = """
# 🧠 Wordcel Pipeline Configuration
# This is a basic example showing common pipeline patterns

dag:
  name: "My First Pipeline"
  # Optional: Configure backend for caching and execution
  # backend:
  #   type: local
  #   cache_dir: .wordcel_cache

nodes:
  # 📥 Data Input - Load data from various sources
  - id: get_data
    type: csv
    path: "https://raw.githubusercontent.com/cs109/2014_data/master/countries.csv"
    # Alternative data sources:
    # type: json
    # type: text
    # type: dataframe

  # 🔧 Data Processing - Transform your data
  - id: sample_data
    type: dataframe_operation
    input: get_data
    operation: "head"  # Get first few rows
    args: [5]

  # 🤖 LLM Processing - Use AI to analyze data
  - id: country_analysis
    type: llm
    template: "What continent is {input} in? Answer with just the continent name."
    input: sample_data
    input_field: "Country"  # Column to process
    output_field: "Continent"  # Where to store results
    # Optional: Specify model
    # model: "openai/gpt-4o"

  # 🔍 Filtering - Keep only relevant data
  - id: filter_africa
    type: llm_filter
    input: country_analysis
    column: "Continent"
    prompt: "Is this in Africa? Answer only Yes or No."

  # 🍽️ More LLM Processing
  - id: cuisine_info
    type: llm
    template: "What cuisine is {input} famous for? Give a brief description."
    input: filter_africa
    input_field: "Country"
    output_field: "Cuisine"

  # 💾 Optional: Save results
  # - id: save_results
  #   type: file_writer
  #   path: "results.json"
  #   input: cuisine_info

# 📚 More node types you can use:

# JSON Data Loading:
# - id: load_json
#   type: json
#   path: "data.json"

# Text Processing:
# - id: process_text
#   type: string_template
#   template: "Hello {name}! You are {age} years old."
#   input: previous_node

# Custom Python Functions:
# - id: custom_processing
#   type: python_function
#   function: my_custom_function
#   input: previous_node

# Sub-pipelines:
# - id: sub_pipeline
#   type: dag
#   path: "sub_pipeline.yaml"
#   input: previous_node

# 💡 Tips:
# - Use ${VARIABLE} for environment variable substitution
# - Set WORDCEL_LOG_LEVEL=DEBUG for detailed execution logs
# - Run 'wordcel dag list-node-types' to see all available nodes
# - Use 'wordcel interactive' for guided pipeline creation

"""


@click.group(cls=RichGroup)
@click.version_option(message=lambda: show_version_info() or "")
def main():
    """🧠 Wordcel - Swiss army-knife for composing LLM outputs

    Wordcel provides a flexible framework for building and executing
    LLM-powered data processing pipelines using simple YAML configuration.
    """
    pass


@main.group(cls=RichGroup)
def dag():
    """🔗 WordcelDAG commands for pipeline management

    Create, execute, and visualize LLM processing pipelines using
    declarative YAML configuration files.
    """
    pass


@dag.command(cls=RichCommand)
@click.argument("pipeline_file")
@click.option(
    "--template", help="Pipeline template to use (basic, advanced, rag, analysis)"
)
@click.option("--force", "-f", is_flag=True, help="Overwrite existing file")
def new(pipeline_file, template, force):
    """📝 Create a new pipeline configuration file

    Creates a new YAML pipeline configuration with helpful examples and comments.
    Choose from different templates based on your use case:

    - basic: Simple CSV processing with LLM
    - advanced: Multi-step complex pipeline
    - rag: Retrieval Augmented Generation
    - analysis: Data analysis pipeline
    """
    if os.path.exists(pipeline_file) and not force:
        console.print(f"[red]✗[/red] Pipeline file {pipeline_file} already exists.")
        console.print(
            "Use [cyan]--force[/cyan] to overwrite or choose a different name."
        )
        return

    # Use template-specific content if specified
    if template:
        content = get_template_content(template)
        if content is None:
            console.print(f"[red]✗[/red] Unknown template: {template}")
            console.print("Available templates: basic, advanced, rag, analysis")
            return
    else:
        content = PIPELINE_TEMPLATE

    with open(pipeline_file, "w") as f:
        f.write(content)

    console.print(f"[green]✅[/green] Pipeline file {pipeline_file} created.")
    if template:
        console.print(f"[dim]Using template:[/dim] {template}")


@dag.command(cls=RichCommand)
def list_node_types():
    """📋 List all available node types with descriptions

    Shows all node types that can be used in pipeline configurations,
    including their descriptions and capabilities.
    """
    from wordcel.dag.nodes import NODE_TYPES
    from rich.table import Table

    table = Table(
        title="Available Node Types", show_header=True, header_style="bold cyan"
    )
    table.add_column("Type", style="cyan", min_width=20)
    table.add_column("Description", style="white")

    for node_type, node_class in sorted(NODE_TYPES.items()):
        description = getattr(node_class, "description", "No description available")
        table.add_row(f"[bold]{node_type}[/bold]", description)

    console.print(table)


@dag.command(cls=RichCommand, name="describe")
@click.argument("node_type")
def describe_node(node_type):
    """🔎 Show detailed information about a specific node type

    Displays the node's description, input specification, and configuration
    parameters to help with pipeline creation.
    """
    from wordcel.dag.nodes import NodeRegistry
    from rich.panel import Panel
    from rich.text import Text

    NodeRegistry.register_default_nodes()
    node_class = NodeRegistry.get(node_type)

    if not node_class:
        console.print(f"[red]✗[/red] Node type '{node_type}' not found.")
        console.print("Use 'wordcel dag list-node-types' to see all available types.")
        return

    # --- Basic Info ---
    console.print(f"\n[bold cyan]Node Type: {node_type}[/bold cyan]")
    description = getattr(node_class, "description", "No description available.")
    console.print(Panel(description, title="Description", border_style="green"))

    # --- Input Spec ---
    spec = getattr(node_class, "input_spec", {})
    spec_type = spec.get("type", "Any")
    spec_desc = spec.get("description", "Not specified.")

    if spec_type is None:
        spec_type_str = "[bold]None[/bold] (Does not accept input)"
    elif spec_type == object:
        spec_type_str = "[bold]Any[/bold]"
    else:
        if isinstance(spec_type, (list, tuple)):
            spec_type_str = ", ".join(
                f"[bold]{t.__name__}[/bold]" for t in spec_type
            )
        else:
            spec_type_str = f"[bold]{spec_type.__name__}[/bold]"

    spec_panel = Text.assemble(
        ("Expected Type: ", "bold"),
        Text.from_markup(f"[yellow]{spec_type_str}[/yellow]"),
        "\n\n",
        (spec_desc, "dim"),
    )
    console.print(
        Panel(spec_panel, title="Input Specification", border_style="yellow")
    )

    # --- Configuration ---
    # This part is more complex as it requires inspecting the __init__ or a config schema
    # For now, we'll add a placeholder.
    # TODO: Implement a more robust way to get config parameters.
    console.print(
        Panel(
            "Configuration parameters are defined in the node's `validate_config` method. Inspect the node's source code for details.",
            title="Configuration",
            border_style="blue",
        )
    )


@dag.command()
@click.argument("pipeline_file")
@click.argument("save_path")
@click.option("--custom-nodes", default=None, help="Path to custom nodes Python file.")
@click.option("--secrets", default=None, help="Path to secrets file.")
def visualize(pipeline_file, save_path, custom_nodes, secrets):
    """Visualize a pipeline."""
    dag = initialize_dag(pipeline_file, secrets=secrets, custom_nodes=custom_nodes)
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
    help="Input data for the pipeline. Given in the format `-i node_id value`.",
)
@click.option(
    "--config-param",
    "-c",
    nargs=2,
    type=(str, str),
    multiple=True,
    help="Substitute values into the config file at runtime. Given in the format `-c param value`.",
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
    config_param,
):
    """Execute a pipeline."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.tree import Tree

    dag = initialize_dag(
        pipeline_file,
        config_params=dict(config_param),
        secrets=secrets,
        custom_nodes=custom_nodes,
        custom_functions=custom_functions,
        custom_backends=custom_backends,
    )

    if input:
        log.info("Given input data: %s", dict(input))

    results = dag.execute(input_data=dict(input), verbose=verbose)

    console = Console()
    tree = Tree("⚡️ [bold blue]Execution Results")

    for node_id, result in results.items():
        node_tree = tree.add(f"[bold green]{node_id}")
        node_tree.add(f"{result}")

    console.print(Panel(tree, expand=False, border_style="bold"))

    if visualization:
        dag.save_image(visualization)


@dag.command(cls=RichCommand)
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
@click.option(
    "--input",
    "-i",
    nargs=2,
    type=(str, str),
    multiple=True,
    help="Input data for the pipeline. Given in the format `-i node_id value`.",
)
@click.option(
    "--config-param",
    "-c",
    nargs=2,
    type=(str, str),
    multiple=True,
    help="Substitute values into the config file at runtime. Given in the format `-c param value`.",
)
def dryrun(
    pipeline_file,
    secrets,
    custom_nodes,
    custom_functions,
    custom_backends,
    input,
    config_param,
):
    """🔍 Show what would be executed without running the pipeline

    Validates the pipeline configuration and displays the execution plan
    including node dependencies, order, and configuration details.
    """
    from rich.console import Console
    from rich.panel import Panel
    from rich.tree import Tree
    from rich.table import Table

    try:
        dag = initialize_dag(
            pipeline_file,
            config_params=dict(config_param),
            secrets=secrets,
            custom_nodes=custom_nodes,
            custom_functions=custom_functions,
            custom_backends=custom_backends,
        )
    except Exception as e:
        console.print(f"[red]✗ Failed to initialize DAG:[/red] {e}")
        return

    console.print(f"[green]✅ Pipeline validation successful:[/green] {pipeline_file}")

    # Show DAG info
    dag_info = Table(title="Pipeline Information", show_header=False)
    dag_info.add_column("Property", style="cyan", min_width=15)
    dag_info.add_column("Value", style="white")

    dag_info.add_row("Name", dag.name or "Unnamed Pipeline")
    dag_info.add_row("Nodes", str(len(dag.nodes)))
    dag_info.add_row("File", pipeline_file)

    if input:
        dag_info.add_row("Input Data", str(dict(input)))

    console.print(dag_info)

    # Show execution order
    execution_order = dag.get_execution_order()
    tree = Tree("🔄 [bold blue]Execution Plan")

    for i, node_id in enumerate(execution_order, 1):
        node = dag.nodes[node_id]
        node_tree = tree.add(
            f"[bold yellow]{i}.[/bold yellow] [bold green]{node_id}[/bold green] ({node.__class__.__name__})"
        )

        # Show dependencies
        if hasattr(node, "input") and node.input:
            if isinstance(node.input, list):
                deps = ", ".join(node.input)
            else:
                deps = str(node.input)
            node_tree.add(f"[dim]depends on:[/dim] {deps}")

        # Show key config
        config_items = []
        for key, value in node.config.items():
            if key not in ["id", "type", "input"] and value is not None:
                if isinstance(value, str) and len(value) > 50:
                    value = value[:47] + "..."
                config_items.append(f"{key}: {value}")

        if config_items:
            node_tree.add(f"[dim]config:[/dim] {', '.join(config_items[:3])}")

    console.print(Panel(tree, expand=False, border_style="bold"))

    console.print("\n[dim]💡 Use 'wordcel dag execute' to run this pipeline[/dim]")


# Add new commands
@main.command(cls=RichCommand)
def interactive():
    """🎮 Launch interactive mode with guided pipeline creation

    Starts an interactive session that guides you through creating,
    editing, and running pipelines with helpful prompts and examples.
    """
    start_interactive_mode()


def get_template_content(template: str) -> str:
    """Get content for a specific template."""
    templates = {
        "basic": PIPELINE_TEMPLATE,
        "advanced": """
# 🧠 Advanced Wordcel Pipeline
# Multi-step pipeline with complex processing

dag:
  name: "Advanced Pipeline"
  backend:
    type: local
    cache_dir: .wordcel_cache

nodes:
  - id: load_data
    type: csv
    path: "data.csv"

  - id: clean_data
    type: dataframe_operation
    input: load_data
    operation: "dropna"

  - id: process_text
    type: llm
    template: "Summarize this text in one sentence: {input}"
    input: clean_data
    input_field: "text"
    output_field: "summary"

  - id: sentiment_analysis
    type: llm
    template: "What is the sentiment of this text? {input}. Answer: Positive, Negative, or Neutral."
    input: process_text
    input_field: "summary"
    output_field: "sentiment"

  - id: filter_positive
    type: llm_filter
    prompt: "Is this sentiment positive?"
    column: "sentiment"
    input: sentiment_analysis
""",
        "rag": """
# 🧠 RAG (Retrieval Augmented Generation) Pipeline
# Document processing and question answering

dag:
  name: "RAG Pipeline"

nodes:
  - id: load_documents
    type: text
    content: "Your document content here"

  - id: chunk_documents
    type: python_function
    function: chunk_text
    input: load_documents

  - id: embed_chunks
    type: python_function  
    function: embed_text
    input: chunk_documents

  - id: query_processing
    type: llm
    template: "Based on this context: {input}, answer: What is the main topic?"
    input: embed_chunks
""",
        "analysis": """
# 🧠 Data Analysis Pipeline
# Statistical analysis with AI insights

dag:
  name: "Data Analysis Pipeline"

nodes:
  - id: load_data
    type: csv
    path: "data.csv"

  - id: basic_stats
    type: dataframe_operation
    input: load_data
    operation: "describe"

  - id: analyze_trends
    type: llm
    template: "Analyze these statistics and identify key trends: {input}"
    input: basic_stats

  - id: generate_insights
    type: llm
    template: "Based on this analysis: {input}, provide 3 key business insights."
    input: analyze_trends
""",
    }
    return templates.get(template)


if __name__ == "__main__":
    main()
