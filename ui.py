"""
All terminal styling lives here so agent.py stays focused on logic.
Swap this file out entirely if you ever want a different look -- nothing
else needs to change.
"""

from rich.console import Console
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.panel import Panel

console = Console()


def banner():
    console.print(Panel.fit(
        "[bold cyan]Local Agent[/bold cyan]  --  type [bold]'exit'[/bold] to quit, "
        "[bold]'clear'[/bold] to reset conversation",
        border_style="cyan",
    ))


def user_prompt() -> str:
    return console.input("\n[bold green]›[/bold green] ")


def assistant_answer(text: str):
    console.print(Panel(
        Markdown(text),
        title="[bold cyan]agent[/bold cyan]",
        title_align="left",
        border_style="cyan",
        padding=(0, 1),
    ))


def tool_call(name: str, args: str):
    console.print(f"  [dim]→ calling[/dim] [bold yellow]{name}[/bold yellow]([dim]{args}[/dim])")


def tool_error(msg: str):
    console.print(f"  [bold red]✗ {msg}[/bold red]")


def diff(diff_text: str, path: str = ""):
    if not diff_text.strip() or diff_text.startswith("No changes"):
        console.print(f"  [dim](no changes -- {diff_text})[/dim]")
        return
    syntax = Syntax(diff_text, "diff", theme="ansi_dark", background_color="default")
    console.print(Panel(
        syntax,
        title=f"[bold magenta]proposed diff[/bold magenta]  [dim]{path}[/dim]",
        title_align="left",
        border_style="magenta",
    ))


def system_msg(msg: str):
    console.print(f"[dim italic]({msg})[/dim italic]")


def debug(msg: str):
    console.print(f"[dim]{msg}[/dim]")


def thinking():
    """Context manager: `with ui.thinking(): ...`"""
    return console.status("[bold cyan]thinking…[/bold cyan]", spinner="dots")