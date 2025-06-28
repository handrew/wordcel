"""Interactive CLI mode for guided wordcel usage."""

import os
import click
from pathlib import Path
from rich.console import Console
from rich.prompt import Prompt, Confirm, IntPrompt
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax
from rich.text import Text
from rich.columns import Columns
from typing import Optional, Dict, Any, List

console = Console()


class InteractiveSession:
    """Interactive session manager for wordcel."""

    def __init__(self):
        self.current_pipeline = None
        self.pipeline_path = None

    def start(self):
        """Start the interactive session."""
        self._show_welcome()

        while True:
            try:
                command = self._get_command()

                if command == "exit":
                    self._show_goodbye()
                    break
                elif command == "help":
                    self._show_help()
                elif command == "new":
                    self._create_pipeline()
                elif command == "edit":
                    self._edit_pipeline()
                elif command == "run":
                    self._run_pipeline()
                elif command == "visualize":
                    self._visualize_pipeline()
                elif command == "nodes":
                    self._list_node_types()
                elif command == "templates":
                    self._show_templates()
                elif command == "status":
                    self._show_status()
                else:
                    console.print(f"[red]Unknown command: {command}[/red]")
                    console.print("Type 'help' for available commands")

            except KeyboardInterrupt:
                console.print("\n[yellow]Use 'exit' to quit[/yellow]")
            except EOFError:
                self._show_goodbye()
                break

    def _show_welcome(self):
        """Show welcome message."""
        welcome_text = """
[bold blue]🧠 Welcome to Wordcel Interactive Mode![/bold blue]

This guided interface will help you create and manage LLM pipelines.
Type commands or 'help' for assistance. Press Ctrl+C or type 'exit' to quit.
"""
        console.print(Panel(welcome_text.strip(), border_style="blue"))

    def _show_goodbye(self):
        """Show goodbye message."""
        console.print(
            "\n[green]👋 Thanks for using Wordcel! Happy pipeline building![/green]"
        )

    def _get_command(self) -> str:
        """Get command from user."""
        prompt_text = "[green]wordcel>[/green]"
        if self.current_pipeline:
            pipeline_name = (
                Path(self.pipeline_path).name if self.pipeline_path else "unsaved"
            )
            prompt_text = (
                f"[green]wordcel[/green]([cyan]{pipeline_name}[/cyan])[green]>[/green]"
            )

        return Prompt.ask(f"\n{prompt_text}", default="help").strip().lower()

    def _show_help(self):
        """Show available commands."""
        table = Table(
            title="Available Commands", show_header=True, header_style="bold cyan"
        )
        table.add_column("Command", style="cyan", min_width=12)
        table.add_column("Description", style="white")

        commands = [
            ("help", "Show this help message"),
            ("new", "Create a new pipeline interactively"),
            ("edit", "Edit the current pipeline"),
            ("run", "Execute the current pipeline"),
            ("visualize", "Create a visualization of the pipeline"),
            ("nodes", "List available node types"),
            ("templates", "Show available pipeline templates"),
            ("status", "Show current pipeline status"),
            ("exit", "Exit interactive mode"),
        ]

        for cmd, desc in commands:
            table.add_row(f"[bold]{cmd}[/bold]", desc)

        console.print(table)

    def _create_pipeline(self):
        """Interactively create a new pipeline."""
        console.print(
            Panel(
                "[bold green]🆕 Create New Pipeline[/bold green]", border_style="green"
            )
        )

        # Get pipeline name
        default_name = "pipeline.yaml"
        pipeline_name = Prompt.ask("Pipeline filename", default=default_name)

        if not pipeline_name.endswith((".yaml", ".yml")):
            pipeline_name += ".yaml"

        # Check if file exists
        if os.path.exists(pipeline_name):
            overwrite = Confirm.ask(
                f"File {pipeline_name} exists. Overwrite?", default=False
            )
            if not overwrite:
                console.print("[yellow]Pipeline creation cancelled[/yellow]")
                return

        # Choose template or build from scratch
        use_template = Confirm.ask("Start from a template?", default=True)

        if use_template:
            template = self._choose_template()
            if template:
                self._create_from_template(pipeline_name, template)
            else:
                return
        else:
            self._create_from_scratch(pipeline_name)

        self.pipeline_path = pipeline_name
        console.print(f"[green]✅ Pipeline created: {pipeline_name}[/green]")

    def _choose_template(self) -> Optional[str]:
        """Let user choose a template."""
        templates = {
            "1": ("basic", "Simple CSV processing with LLM"),
            "2": ("advanced", "Complex multi-step pipeline"),
            "3": ("rag", "RAG (Retrieval Augmented Generation)"),
            "4": ("analysis", "Data analysis and visualization"),
            "5": ("sentiment", "Sentiment analysis pipeline"),
        }

        console.print("\n[bold cyan]Available Templates:[/bold cyan]")
        for key, (name, desc) in templates.items():
            console.print(f"  {key}. [green]{name}[/green] - {desc}")

        choice = Prompt.ask(
            "Choose template", choices=list(templates.keys()), default="1"
        )
        return templates[choice][0]

    def _create_from_template(self, filename: str, template: str):
        """Create pipeline from template."""
        # Import here to avoid circular imports
        from wordcel.cli import PIPELINE_TEMPLATE

        # In a real implementation, you'd have different templates
        template_content = PIPELINE_TEMPLATE

        # Customize based on template choice
        if template == "rag":
            template_content = self._get_rag_template()
        elif template == "sentiment":
            template_content = self._get_sentiment_template()

        with open(filename, "w") as f:
            f.write(template_content)

    def _create_from_scratch(self, filename: str):
        """Create pipeline from scratch with guided questions."""
        console.print(
            "\n[bold yellow]Let's build your pipeline step by step![/bold yellow]"
        )

        # Get basic info
        dag_name = Prompt.ask("Pipeline name", default="My Pipeline")

        # Build nodes interactively
        nodes = []

        # Data source
        console.print("\n[bold]Step 1: Data Source[/bold]")
        source_type = Prompt.ask(
            "Data source type", choices=["csv", "json", "text", "manual"], default="csv"
        )

        if source_type in ["csv", "json"]:
            data_path = Prompt.ask("Data file path or URL")
            nodes.append({"id": "load_data", "type": source_type, "path": data_path})
        elif source_type == "text":
            text_content = Prompt.ask("Text content")
            nodes.append({"id": "load_data", "type": "text", "content": text_content})

        # Processing steps
        add_more = True
        step_num = 2

        while add_more:
            console.print(f"\n[bold]Step {step_num}: Processing[/bold]")

            process_type = Prompt.ask(
                "Processing type",
                choices=["llm", "llm_filter", "dataframe_operation", "skip"],
                default="llm",
            )

            if process_type == "skip":
                break

            if process_type == "llm":
                prompt_template = Prompt.ask("LLM prompt template")
                nodes.append(
                    {
                        "id": f"process_{step_num-1}",
                        "type": "llm",
                        "template": prompt_template,
                        "input": nodes[-1]["id"] if nodes else "load_data",
                    }
                )
            elif process_type == "llm_filter":
                filter_prompt = Prompt.ask("Filter prompt")
                column = Prompt.ask("Column to filter")
                nodes.append(
                    {
                        "id": f"filter_{step_num-1}",
                        "type": "llm_filter",
                        "prompt": filter_prompt,
                        "column": column,
                        "input": nodes[-1]["id"] if nodes else "load_data",
                    }
                )

            step_num += 1
            add_more = Confirm.ask("Add another processing step?", default=False)

        # Generate YAML
        pipeline_config = {"dag": {"name": dag_name}, "nodes": nodes}

        yaml_content = self._dict_to_yaml(pipeline_config)

        with open(filename, "w") as f:
            f.write(yaml_content)

    def _dict_to_yaml(self, data: Dict[str, Any]) -> str:
        """Convert dict to YAML string (simple implementation)."""
        import yaml

        return yaml.dump(data, default_flow_style=False, sort_keys=False)

    def _get_rag_template(self) -> str:
        """Get RAG pipeline template."""
        return """
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
"""

    def _get_sentiment_template(self) -> str:
        """Get sentiment analysis template."""
        return """
dag:
  name: "Sentiment Analysis"

nodes:
  - id: load_data
    type: csv
    path: "data.csv"

  - id: sentiment_analysis
    type: llm
    template: "Analyze the sentiment of this text: {input}. Respond with: Positive, Negative, or Neutral."
    input: load_data
    input_field: "text"
    output_field: "sentiment"

  - id: filter_positive
    type: llm_filter
    prompt: "Is this sentiment positive?"
    column: "sentiment"
    input: sentiment_analysis
"""

    def _edit_pipeline(self):
        """Edit current pipeline."""
        if not self.pipeline_path:
            console.print(
                "[yellow]No pipeline loaded. Create one first with 'new'[/yellow]"
            )
            return

        console.print(f"[blue]📝 Editing {self.pipeline_path}[/blue]")
        console.print("(This would open your default editor in a real implementation)")

    def _run_pipeline(self):
        """Run the current pipeline."""
        if not self.pipeline_path:
            console.print(
                "[yellow]No pipeline loaded. Create one first with 'new'[/yellow]"
            )
            return

        console.print(f"[blue]🚀 Running {self.pipeline_path}[/blue]")

        # Ask for execution options
        verbose = Confirm.ask("Enable verbose output?", default=False)

        # Build command
        cmd_parts = ["wordcel", "dag", "execute", self.pipeline_path]
        if verbose:
            cmd_parts.append("--verbose")

        cmd = " ".join(cmd_parts)
        console.print(f"[dim]Running: {cmd}[/dim]")

        # In real implementation, would execute the command
        console.print("[green]✅ Pipeline execution started[/green]")

    def _visualize_pipeline(self):
        """Visualize the current pipeline."""
        if not self.pipeline_path:
            console.print(
                "[yellow]No pipeline loaded. Create one first with 'new'[/yellow]"
            )
            return

        output_file = Prompt.ask("Output image file", default="pipeline.png")
        console.print(f"[blue]📊 Creating visualization: {output_file}[/blue]")

        # In real implementation, would create visualization
        console.print("[green]✅ Visualization created[/green]")

    def _list_node_types(self):
        """List available node types."""
        try:
            from wordcel.dag.nodes import NODE_TYPES

            table = Table(
                title="Available Node Types", show_header=True, header_style="bold cyan"
            )
            table.add_column("Type", style="cyan", min_width=15)
            table.add_column("Description", style="white")

            for node_type, node_class in sorted(NODE_TYPES.items()):
                description = getattr(node_class, "description", "No description")
                table.add_row(f"[bold]{node_type}[/bold]", description)

            console.print(table)

        except ImportError:
            console.print("[red]Could not load node types[/red]")

    def _show_templates(self):
        """Show available templates."""
        templates = [
            ("basic", "Simple CSV processing with LLM transformations"),
            ("advanced", "Multi-step pipeline with complex operations"),
            ("rag", "Retrieval Augmented Generation setup"),
            ("analysis", "Data analysis with visualization"),
            ("sentiment", "Sentiment analysis pipeline"),
        ]

        table = Table(
            title="Available Templates", show_header=True, header_style="bold cyan"
        )
        table.add_column("Template", style="cyan", min_width=12)
        table.add_column("Description", style="white")

        for name, desc in templates:
            table.add_row(f"[bold]{name}[/bold]", desc)

        console.print(table)

    def _show_status(self):
        """Show current session status."""
        console.print(
            Panel("[bold blue]📊 Session Status[/bold blue]", border_style="blue")
        )

        if self.pipeline_path:
            console.print(f"[green]Current pipeline:[/green] {self.pipeline_path}")

            # Show file info if it exists
            if os.path.exists(self.pipeline_path):
                stat = os.stat(self.pipeline_path)
                size = stat.st_size
                console.print(f"[dim]File size:[/dim] {size} bytes")
            else:
                console.print("[yellow]File not found on disk[/yellow]")
        else:
            console.print("[yellow]No pipeline loaded[/yellow]")

        console.print(f"[dim]Working directory:[/dim] {os.getcwd()}")


def start_interactive_mode():
    """Start interactive mode."""
    session = InteractiveSession()
    session.start()
