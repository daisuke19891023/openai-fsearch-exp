"""Typer CLI entry points for the Code Agent Experiments project."""

from __future__ import annotations

import json
import os
from datetime import datetime, UTC
from enum import Enum
from pathlib import Path
from typing import Any, cast

import typer
from openai import OpenAI
from rich.console import Console
from rich.table import Table

from .agent import AgentsSDKAgent, ResponsesAgent
from .agent.builtin_tools import create_ripgrep_tool
from .agent.tooling import ToolRegistry
from .config import LoadError, load_run_configs, load_scenarios
from .domain.models import Metrics, ReportSummary, ToolLimits
from .evaluation.metrics import aggregate_metrics
from .evaluation.reporting import render_html_report, render_markdown_report
from .orchestration import ExperimentOrchestrator, ExperimentStorage, default_executor

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

RUNS_FILE_OPTION = typer.Option(
    ...,
    "--runs-file",
    file_okay=True,
    dir_okay=False,
    exists=True,
    readable=True,
    resolve_path=True,
    help="Path to the YAML file containing run configurations.",
)
SCENARIOS_FILE_OPTION = typer.Option(
    ...,
    "--scenarios-file",
    file_okay=True,
    dir_okay=False,
    exists=True,
    readable=True,
    resolve_path=True,
    help="Path to the YAML file containing scenario definitions.",
)
OUTPUT_DIR_OPTION = typer.Option(
    Path("artifacts"),
    "--output-dir",
    "-o",
    file_okay=False,
    help="Directory where orchestrator artefacts should be stored.",
)
ENV_FILE_OPTION = typer.Option(
    None,
    "--env-file",
    "-e",
    resolve_path=True,
    file_okay=True,
    dir_okay=False,
    help="Optional .env file(s) applied to both run and scenario templates.",
)

DEFAULT_EXECUTOR = default_executor


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
        items = cast("list[Any]", payload)
        metrics = [Metrics.model_validate(item) for item in items]
    else:
        message = "Metrics input must be JSONL, JSON array, or a serialized ReportSummary."
        raise typer.BadParameter(message)
    if not metrics:
        message = "No metrics found in the provided file."
        raise typer.BadParameter(message)
    return metrics


def _lookup_metric_value(values: dict[Any, Any] | None, key: int) -> float:
    """Fetch a metric value for ``key`` from ``values`` handling str/int keys."""
    if not values:
        return 0.0
    if key in values:
        return float(values[key])
    str_key = str(key)
    if str_key in values:
        return float(values[str_key])
    return 0.0


def _print_strategy_table(comparison: dict[str, dict[str, Any]]) -> None:
    """Render a Rich table summarising aggregated strategy metrics."""
    if not comparison:
        console.print("[yellow]No metrics were produced during the run.[/yellow]")
        return

    table = Table(title="Strategy Comparison")
    table.add_column("Strategy", style="bold")
    table.add_column("Scenarios", justify="right")
    table.add_column("Mean MRR", justify="right")
    table.add_column("Recall@5", justify="right")
    table.add_column("Recall@10", justify="right")
    table.add_column("Total Tool Calls", justify="right")

    for strategy in sorted(comparison):
        aggregates = comparison[strategy]
        scenario_count = int(aggregates.get("scenario_count", 0))
        mean_mrr = float(aggregates.get("mean_mrr", 0.0) or 0.0)
        recall_map = aggregates.get("avg_recall_at_k")
        recall_5 = _lookup_metric_value(recall_map, 5)
        recall_10 = _lookup_metric_value(recall_map, 10)
        total_calls = int(aggregates.get("total_tool_calls", 0))

        table.add_row(
            strategy,
            str(scenario_count),
            f"{mean_mrr:.3f}",
            f"{recall_5:.3f}",
            f"{recall_10:.3f}",
            str(total_calls),
        )

    console.print(table)


@app.command("run-scenarios")
def run_scenarios(
    runs_file: Path = RUNS_FILE_OPTION,
    scenarios_file: Path = SCENARIOS_FILE_OPTION,
    output_dir: Path = OUTPUT_DIR_OPTION,
    env_file: list[Path] | None = ENV_FILE_OPTION,
) -> None:
    """Execute a batch of scenarios across the configured run profiles."""
    env_files = list(env_file) if env_file else None

    try:
        run_result = load_run_configs(runs_file, env_files=env_files)
    except LoadError as exc:
        raise typer.BadParameter(str(exc)) from exc

    try:
        scenario_result = load_scenarios(scenarios_file, env_files=env_files)
    except LoadError as exc:
        raise typer.BadParameter(str(exc)) from exc

    run_configs = list(run_result.items)
    scenarios = list(scenario_result.items)

    if not run_configs:
        message = f"No run configurations found in {runs_file}"
        raise typer.BadParameter(message)
    if not scenarios:
        message = f"No scenarios found in {scenarios_file}"
        raise typer.BadParameter(message)

    timestamp = datetime.now(tz=UTC).strftime("%Y%m%d-%H%M%S")
    batch_dir = (output_dir / f"run-{timestamp}").resolve()

    storage = ExperimentStorage(batch_dir)
    orchestrator = ExperimentOrchestrator(DEFAULT_EXECUTOR, storage)
    result = orchestrator.run(run_configs, scenarios)

    console.print(f"[dim]Runs loaded from {run_result.source}[/dim]")
    console.print(f"[dim]Scenarios loaded from {scenario_result.source}[/dim]")

    _print_strategy_table(result.comparison)
    console.print(f"[green]Retrieval records written to {result.records_path}[/green]")
    console.print(f"[green]Metrics written to {result.metrics_path}[/green]")
    console.print(f"[green]Comparison summary written to {result.summary_path}[/green]")

    if result.failures:
        plural = "s" if len(result.failures) != 1 else ""
        console.print(
            "[yellow]"
            f"{len(result.failures)} scenario execution{plural} failed. "
            f"See {result.failures_path} for details."
            "[/yellow]",
        )

    console.print(f"[green]Artefacts stored in {result.output_dir}[/green]")


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
