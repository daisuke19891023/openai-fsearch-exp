"""Typer CLI entry points for the Code Agent Experiments project."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from openai import OpenAI
from rich.console import Console

from .agent import ResponsesAgent
from .agent.builtin_tools import create_ripgrep_tool
from .agent.tooling import ToolRegistry
from .domain.models import ToolLimits

app = typer.Typer(help="Run single-shot code agent experiments against a repository.")
console = Console()

PROMPT_ARGUMENT = typer.Argument(
    ...,
    help="Natural language description of the issue to investigate.",
)
REPO_OPTION = typer.Option(
    None,
    "--repo",
    "-r",
    exists=True,
    file_okay=False,
    readable=True,
    help="Path to the repository that should be searched.",
)
MODEL_OPTION = typer.Option(
    "gpt-4.1-mini",
    help="OpenAI model identifier to use for the Responses API.",
)
MAX_CALLS_OPTION = typer.Option(
    6,
    min=1,
    help="Maximum number of tool calls permitted in a single run.",
)


@app.command()
def query(
    prompt: str = PROMPT_ARGUMENT,
    repo: Path | None = REPO_OPTION,
    model: str = MODEL_OPTION,
    max_calls: int = MAX_CALLS_OPTION,
) -> None:
    """Run a Responses API-backed agent for a single query."""
    client = OpenAI()
    registry = ToolRegistry([create_ripgrep_tool()])
    limits = ToolLimits(max_calls=max_calls)
    if repo is None:
        repo = Path.cwd()
    repository_root = repo.resolve()
    system_prompt = (
        "You are a code search assistant helping triage issues in a repository.\n"
        f"The repository root on disk is: {repository_root}.\n"
        "Always call the `ripgrep` tool with the 'root' argument set to this absolute path.\n"
        "Return concise summaries of the files and lines that look relevant."
    )
    agent = ResponsesAgent(
        client,
        model=model,
        tool_registry=registry,
        tool_limits=limits,
        system_prompt=system_prompt,
    )
    result = agent.run(prompt)

    console.print("[bold]Model response:[/bold]")
    console.print(result.output_text or "<no assistant message>")

    console.print("\n[bold]Tool calls:[/bold]")
    if not result.tool_calls:
        console.print("  (no tool calls recorded)")
    for call in result.tool_calls:
        args_preview = json.dumps(call.args, ensure_ascii=False)
        console.print(f"  - {call.name} {args_preview} ({call.latency_ms} ms)")
        if call.stdout_preview:
            console.print(f"    Output preview: {call.stdout_preview[:200]}")
    if result.limit_reached:
        console.print("[yellow]Tool call limit reached before the response completed.[/yellow]")


def main() -> None:  # pragma: no cover - Typer entry point
    """Invoke the Typer application."""
    app()


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()
