"""Rich CLI formatting and enhanced help system."""

import click
import os
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text
from rich.columns import Columns
from rich.markup import escape
from typing import Optional, List, Tuple, Dict, Any

console = Console()


class RichGroup(click.Group):
    """Custom Click Group with Rich formatting."""

    def format_help(self, ctx, formatter):
        """Override help formatting with Rich."""
        # Main header - use better title for root command
        if ctx.parent is None and (self.name == "main" or not self.name):
            title = "Wordcel"
        else:
            title = self.name or "Wordcel"
        description = self.help or "Swiss army-knife for composing LLM outputs"

        # Clean up description to avoid redundancy
        clean_description = description
        if description.startswith("🧠 Wordcel - "):
            clean_description = description[len("🧠 Wordcel - ") :]
        elif description.startswith("🔗 "):
            clean_description = description[2:].strip()

        console.print(
            Panel.fit(
                f"[bold blue]🧠 {title}[/bold blue]\n" f"{clean_description}",
                border_style="blue",
            )
        )

        # Commands table
        if self.commands:
            table = Table(
                title="Available Commands", show_header=True, header_style="bold cyan"
            )
            table.add_column("Command", style="cyan", min_width=15)
            table.add_column("Description", style="white")

            for name, cmd in sorted(self.commands.items()):
                description = cmd.get_short_help_str() or "No description"
                # Escape any Rich markup in the description
                description = escape(description)
                table.add_row(f"[bold]{name}[/bold]", description)

            console.print(table)

        # Show usage
        if ctx.parent is None:  # Root command
            self._show_usage(ctx)

        # Show examples
        self._show_examples(ctx)

        # Show useful tips
        self._show_tips()

    def _show_usage(self, ctx):
        """Show basic usage information."""
        console.print(f"\n[bold yellow]📋 Usage:[/bold yellow]")
        console.print(
            f"  [dim]$[/dim] [green]wordcel[/green] [cyan]COMMAND[/cyan] [yellow][OPTIONS][/yellow]"
        )
        console.print(f"  [dim]$[/dim] [green]wordcel[/green] [cyan]--help[/cyan]")
        console.print(
            f"  [dim]$[/dim] [green]wordcel[/green] [cyan]COMMAND[/cyan] [cyan]--help[/cyan]"
        )

    def _show_examples(self, ctx):
        """Show practical examples."""
        if ctx.parent is None:  # Root command examples
            examples = [
                ("Create new pipeline", "wordcel dag new my-pipeline.yaml"),
                ("Execute pipeline", "wordcel dag execute my-pipeline.yaml"),
                (
                    "Visualize pipeline",
                    "wordcel dag visualize my-pipeline.yaml output.png",
                ),
                ("List node types", "wordcel dag list-node-types"),
                ("Interactive mode", "wordcel interactive"),
            ]
        else:  # Subcommand examples
            examples = self._get_subcommand_examples(ctx.info_name)

        if examples:
            console.print(f"\n[bold yellow]📚 Examples:[/bold yellow]")
            for desc, cmd in examples:
                console.print(f"  • {desc}:")
                console.print(f"    [dim]$[/dim] [green]{cmd}[/green]")

    def _get_subcommand_examples(self, command_name: str) -> List[Tuple[str, str]]:
        """Get examples for specific subcommands."""
        examples_map = {
            "dag": [
                (
                    "Create and run pipeline",
                    "wordcel dag new test.yaml && wordcel dag execute test.yaml",
                ),
                (
                    "Run with custom input",
                    "wordcel dag execute pipeline.yaml -i input_node 'Hello World'",
                ),
                ("Debug execution", "wordcel dag execute pipeline.yaml --verbose"),
            ]
        }
        return examples_map.get(command_name, [])

    def _show_tips(self):
        """Show helpful tips."""
        tips = [
            "Use [cyan]--help[/cyan] with any command for detailed options",
            "Set [cyan]WORDCEL_LOG_LEVEL=DEBUG[/cyan] for verbose output",
            "Pipeline files support environment variable substitution with [cyan]${VAR}[/cyan]",
            "Use [cyan]wordcel interactive[/cyan] for guided pipeline creation",
        ]

        console.print(f"\n[bold blue]💡 Tips:[/bold blue]")
        for tip in tips:
            console.print(f"  • {tip}")


class RichCommand(click.Command):
    """Custom Click Command with Rich formatting."""

    def format_help(self, ctx, formatter):
        """Override help formatting with Rich."""
        # Command header
        console.print(
            Panel.fit(
                f"[bold green]{self.name}[/bold green]\n"
                f"{escape(self.help or 'No description available')}",
                border_style="green",
            )
        )

        # Usage section
        self._show_usage(ctx)

        # Parameters table
        if self.params:
            self._show_parameters()

        # Examples specific to this command
        self._show_command_examples(ctx.command_path)

        # Additional notes
        self._show_notes(ctx.command_path)

    def _show_usage(self, ctx):
        """Show command usage."""
        usage_parts = [f"[green]{ctx.command_path}[/green]"]

        # Add arguments and options
        for param in self.params:
            if isinstance(param, click.Argument):
                name = param.name.upper()
                if param.required:
                    usage_parts.append(f"[yellow]{name}[/yellow]")
                else:
                    usage_parts.append(f"[yellow][{name}][/yellow]")
            elif isinstance(param, click.Option):
                opt_str = f"[cyan]{param.opts[0]}[/cyan]"
                if not param.is_flag and param.type.name != "flag":
                    opt_str += f" [yellow]{param.name.upper()}[/yellow]"
                if not param.required:
                    opt_str = f"[{opt_str}]"
                usage_parts.append(opt_str)

        console.print(f"\n[bold]Usage:[/bold] {' '.join(usage_parts)}")

    def _show_parameters(self):
        """Show parameters in a nice table."""
        # Separate arguments and options
        arguments = [p for p in self.params if isinstance(p, click.Argument)]
        options = [p for p in self.params if isinstance(p, click.Option)]

        if arguments:
            table = Table(
                title="Arguments", show_header=True, header_style="bold yellow"
            )
            table.add_column("Name", style="yellow")
            table.add_column("Description", style="white")
            table.add_column("Required", style="cyan")

            for param in arguments:
                name = param.name.upper()
                description = escape(getattr(param, "help", None) or "No description")
                required = "Yes" if param.required else "No"
                table.add_row(name, description, required)

            console.print(table)

        if options:
            table = Table(title="Options", show_header=True, header_style="bold cyan")
            table.add_column("Option", style="cyan", min_width=12)
            table.add_column("Type", style="yellow", min_width=8)
            table.add_column("Description", style="white")
            table.add_column("Default", style="dim", min_width=8)

            for param in options:
                # Handle multiple option names
                option_names = "/".join(param.opts)
                param_type = self._get_param_type_display(param)
                description = escape(getattr(param, "help", None) or "No description")
                default = self._get_param_default_display(param)

                table.add_row(option_names, param_type, description, default)

            console.print(table)

    def _get_param_type_display(self, param) -> str:
        """Get display string for parameter type."""
        if param.is_flag:
            return "flag"
        elif hasattr(param.type, "choices"):
            return f"choice"
        elif hasattr(param.type, "name"):
            return param.type.name
        else:
            return str(param.type).lower()

    def _get_param_default_display(self, param) -> str:
        """Get display string for parameter default."""
        if param.is_flag:
            return "False"
        elif param.default is None:
            return "-"
        elif isinstance(param.default, (list, tuple)) and len(param.default) == 0:
            return "[]"
        else:
            return str(param.default)

    def _show_command_examples(self, command_path: str):
        """Show examples for specific commands."""
        examples_map = {
            "dag new": [
                ("Basic pipeline", "wordcel dag new pipeline.yaml"),
                ("From template", "wordcel dag new --template advanced pipeline.yaml"),
            ],
            "dag execute": [
                ("Simple execution", "wordcel dag execute pipeline.yaml"),
                (
                    "With input data",
                    "wordcel dag execute pipeline.yaml -i node1 'hello world'",
                ),
                ("Debug mode", "wordcel dag execute pipeline.yaml --verbose"),
                (
                    "With secrets",
                    "wordcel dag execute pipeline.yaml --secrets secrets.yaml",
                ),
                (
                    "Custom config",
                    "wordcel dag execute pipeline.yaml -c model_name gpt-4o",
                ),
            ],
            "dag visualize": [
                ("PNG output", "wordcel dag visualize pipeline.yaml graph.png"),
                (
                    "With custom nodes",
                    "wordcel dag visualize pipeline.yaml graph.png --custom-nodes nodes.py",
                ),
            ],
        }

        examples = examples_map.get(command_path)
        if examples:
            console.print(f"\n[bold yellow]📚 Examples:[/bold yellow]")
            for desc, cmd in examples:
                console.print(f"  • {desc}:")
                console.print(f"    [dim]$[/dim] [green]{cmd}[/green]")

    def _show_notes(self, command_path: str):
        """Show additional notes for specific commands."""
        notes_map = {
            "dag execute": [
                "Pipeline files can use environment variables: [cyan]${API_KEY}[/cyan]",
                "Use [cyan]--verbose[/cyan] to see detailed execution logs",
                'Input data supports JSON: [cyan]-i node1 \'{"key": "value"}\'[/cyan]',
            ],
            "dag new": [
                "Generated pipelines include helpful comments and examples",
                "Templates provide pre-configured setups for common use cases",
            ],
        }

        notes = notes_map.get(command_path)
        if notes:
            console.print(f"\n[bold blue]📝 Notes:[/bold blue]")
            for note in notes:
                console.print(f"  • {note}")


def show_version_info():
    """Show detailed version information."""
    try:
        import wordcel

        version = getattr(wordcel, "__version__", "unknown")
    except:
        version = "unknown"

    console.print(
        Panel.fit(
            f"[bold blue]🧠 Wordcel v{version}[/bold blue]\n"
            f"Swiss army-knife for composing LLM outputs\n\n"
            f"[dim]Python: {'.'.join(map(str, __import__('sys').version_info[:3]))}[/dim]",
            border_style="blue",
        )
    )
