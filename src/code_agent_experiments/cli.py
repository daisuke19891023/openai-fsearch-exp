"""Typer CLI entry points for the Code Agent Experiments project."""

from __future__ import annotations

import json
import os
from enum import Enum
from pathlib import Path
from typing import Any, cast

import typer
from openai import OpenAI
from rich.console import Console

from .agent import AgentsSDKAgent, ResponsesAgent
from .agent.builtin_tools import create_ripgrep_tool
from .agent.tooling import ToolRegistry
from .domain.models import Metrics, ReportSummary, ToolLimits
from .evaluation.metrics import aggregate_metrics
from .evaluation.reporting import render_html_report, render_markdown_report

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
    help="OpenAI model identifier to use for the agent backend.",
)
MAX_CALLS_OPTION = typer.Option(
    6,
    min=1,
    help="Maximum number of tool calls permitted in a single run.",
)
METRICS_FILE_ARGUMENT = typer.Argument(
    ...,
    exists=True,
    file_okay=True,
    dir_okay=False,
    readable=True,
    resolve_path=True,
    help="Path to a metrics JSON or JSONL file.",
)

REPORT_OUTPUT_OPTION = typer.Option(
    Path("cae-report.md"),
    "--output",
    "-o",
    dir_okay=False,
    help="Path where the generated report should be written.",
)

FORMAT_OPTION = typer.Option(
    "markdown",
    "--format",
    "-f",
    case_sensitive=False,
    help="Report output format: 'markdown' or 'html'.",
)


def _load_metrics(metrics_file: Path) -> list[Metrics]:
    text = metrics_file.read_text(encoding="utf-8")
    try:
        payload: Any = json.loads(text)
    except json.JSONDecodeError:
        payload = [json.loads(line) for line in text.splitlines() if line.strip()]

    metrics: list[Metrics]
    if isinstance(payload, dict) and "per_scenario" in payload:
        summary = ReportSummary.model_validate(payload)
        metrics = summary.per_scenario
    elif isinstance(payload, list):
        items = cast(list[Any], payload)
        metrics = [Metrics.model_validate(item) for item in items]
    else:
        message = "Metrics input must be JSONL, JSON array, or a serialized ReportSummary."
        raise typer.BadParameter(message)
    if not metrics:
        message = "No metrics found in the provided file."
        raise typer.BadParameter(message)
    return metrics


class AgentDriver(str, Enum):
    """Supported agent execution backends."""

    RESPONSES = "responses"
    AGENTS = "agents"


DRIVER_OPTION = typer.Option(
    AgentDriver.RESPONSES,
    "--driver",
    case_sensitive=False,
    help="Agent backend to use: 'responses' or 'agents'.",
)
WORKFLOW_OPTION = typer.Option(
    "code-agent-experiments.sample",
    "--workflow-name",
    help="Tracing workflow name when using the Agents SDK backend.",
)
DISABLE_TRACE_OPTION = typer.Option(
    default=False,
    help="Disable Agents SDK tracing export for this run.",
)


@app.command()
def query(
    prompt: str = PROMPT_ARGUMENT,
    repo: Path | None = REPO_OPTION,
    model: str = MODEL_OPTION,
    max_calls: int = MAX_CALLS_OPTION,
    driver: AgentDriver = DRIVER_OPTION,
    workflow_name: str = WORKFLOW_OPTION,
    disable_trace: bool = DISABLE_TRACE_OPTION,
) -> None:
    """Run a single query using the selected agent backend."""
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
    if driver is AgentDriver.RESPONSES:
        client = OpenAI()
        agent = ResponsesAgent(
            client,
            model=model,
            tool_registry=registry,
            tool_limits=limits,
            system_prompt=system_prompt,
        )
        result = agent.run(prompt)
    else:
        metadata = {"repository": str(repository_root)}
        agent = AgentsSDKAgent(
            model=model,
            tool_registry=registry,
            tool_limits=limits,
            system_prompt=system_prompt,
            workflow_name=workflow_name,
            trace_export_api_key=os.environ.get("OPENAI_API_KEY"),
            tracing_enabled=not disable_trace,
        )
        result = agent.run(
            prompt,
            workflow_name=workflow_name,
            trace_metadata=metadata,
        )

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
    if driver is AgentDriver.AGENTS and not disable_trace:
        console.print(
            "[dim]Trace exported via Agents SDK. Review it in the OpenAI dashboard under "
            f"workflow '{workflow_name}'.[/dim]",
        )


@app.command()
def report(
    metrics_file: Path = METRICS_FILE_ARGUMENT,
    output: Path = REPORT_OUTPUT_OPTION,
    report_format: str = FORMAT_OPTION,
) -> None:
    """Generate an evaluation report from stored metrics."""
    metrics = _load_metrics(metrics_file)
    run_ids = {metric.run_id for metric in metrics}
    if len(run_ids) != 1:
        message = "Metrics must originate from a single run."
        raise typer.BadParameter(message)

    summary = ReportSummary(
        run_id=run_ids.pop(),
        per_scenario=list(metrics),
        aggregates=aggregate_metrics(metrics),
    )

    fmt = report_format.lower()
    if fmt == "markdown":
        output = render_markdown_report(summary, output)
    elif fmt == "html":
        output = render_html_report(summary, output)
    else:
        message = "Unsupported format. Choose 'markdown' or 'html'."
        raise typer.BadParameter(message)

    console.print(f"[green]Report written to {output}[/green]")


def main() -> None:  # pragma: no cover - Typer entry point
    """Invoke the Typer application."""
    app()


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()
