"""Typer CLI entry points for the Code Agent Experiments project."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence, cast

import typer
from openai import OpenAI
from rich.console import Console
from rich.table import Table

from .agent import ResponsesAgent
from .agent.builtin_tools import create_ripgrep_tool
from .agent.tooling import ToolRegistry
from .config import load_run_configs, load_scenarios
from .domain.models import Metrics, ReportSummary, ToolLimits
from .evaluation.metrics import aggregate_metrics
from .evaluation.reporting import render_html_report, render_markdown_report
from .orchestration import RunSummary, ScenarioRunner

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

RUNS_OPTION = typer.Option(
    ...,
    "--runs",
    "-c",
    exists=True,
    file_okay=True,
    dir_okay=False,
    resolve_path=True,
    help="Path to the run configuration YAML file.",
)
SCENARIOS_OPTION = typer.Option(
    ...,
    "--scenarios",
    "-s",
    exists=True,
    file_okay=True,
    dir_okay=False,
    resolve_path=True,
    help="Path to the scenarios YAML file.",
)
OUTPUT_DIR_OPTION = typer.Option(
    Path("artifacts"),
    "--output-dir",
    "-o",
    dir_okay=True,
    file_okay=False,
    resolve_path=True,
    help="Directory where retrieval records and metrics will be stored.",
)

if TYPE_CHECKING:
    _ENV_OPTION: Any = None
else:
    _ENV_OPTION = typer.Option(
        None,
        "--env",
        "-e",
        help="Optional .env file(s) applied to both run and scenario configurations.",
    )


def _render_summary_table(summaries: Sequence[RunSummary]) -> None:
    """Print a Rich table comparing aggregate metrics across runs."""
    table = Table(title="Run Comparison")
    table.add_column("Run ID", style="bold")
    table.add_column("Strategy")
    table.add_column("Scenarios", justify="right")
    table.add_column("Recall@5", justify="right")
    table.add_column("Recall@10", justify="right")
    table.add_column("MRR", justify="right")

    for summary in summaries:
        aggregates = summary.aggregates
        recall = aggregates.get("avg_recall_at_k", {})
        recall5 = 0.0
        recall10 = 0.0
        if isinstance(recall, dict):
            recall5 = float(recall.get(5, 0.0))
            recall10 = float(recall.get(10, 0.0))
        scenario_value = aggregates.get("scenario_count", 0)
        scenario_count = int(scenario_value) if isinstance(scenario_value, (int, float)) else 0
        mrr_value = aggregates.get("mean_mrr", 0.0)
        mrr = float(mrr_value) if isinstance(mrr_value, (int, float)) else 0.0
        table.add_row(
            summary.run_id,
            summary.strategy,
            f"{scenario_count}",
            f"{recall5:.2f}",
            f"{recall10:.2f}",
            f"{mrr:.3f}",
        )

    console.print(table)


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


@app.command("run-scenarios")
def run_scenarios(
    runs: Path = RUNS_OPTION,
    scenarios: Path = SCENARIOS_OPTION,
    output_dir: Path = OUTPUT_DIR_OPTION,
    env: str | None = _ENV_OPTION,
) -> None:
    """Execute experiment runs across all provided scenarios."""
    env_files = [Path(env)] if env else None
    run_configs = load_run_configs(runs, env_files=env_files)
    scenario_configs = load_scenarios(scenarios, env_files=env_files)

    if not run_configs.items:
        message = "Run configuration file did not contain any runs."
        raise typer.BadParameter(message)
    if not scenario_configs.items:
        message = "Scenario file did not contain any scenarios."
        raise typer.BadParameter(message)

    runner = ScenarioRunner(
        run_configs.items,
        scenario_configs.items,
        output_dir,
        log=console.print,
    )
    summaries = runner.execute()
    if not summaries:
        console.print("[yellow]No runs were executed.[/yellow]")
        return

    _render_summary_table(summaries)
    artefact_root = output_dir.resolve()
    console.print(f"[green]Experiment artefacts written to {artefact_root}[/green]")


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
