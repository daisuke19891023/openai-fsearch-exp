"""Typer CLI entry points for the Code Agent Experiments project."""

from __future__ import annotations

import json
import os
import statistics
import time
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping, Protocol, Sequence, cast

import typer
from openai import OpenAI
from rich.console import Console
from rich.table import Table

from .agent import AgentsSDKAgent, ResponsesAgent
from .agent.builtin_tools import create_ripgrep_tool
from .agent.tooling import ToolRegistry
from .chunking.fixed import FixedTokenChunker
from .config import LoadError, load_environment, load_run_configs, load_scenarios
from .domain.models import Metrics, ReportSummary, RetrievalRecord, RunConfig, Scenario, ToolLimits
from .embeddings import OpenAIEmbeddingsClient
from .evaluation.metrics import aggregate_metrics, compute_retrieval_metrics
from .evaluation.reporting import render_html_report, render_markdown_report
from .orchestration import ExperimentOrchestrator, ExperimentStorage, default_executor
from .tools import run_fd, run_find, run_grep, run_ripgrep, warn_if_missing_binaries
from .vector import FaissVectorStore, VectorRecord

if TYPE_CHECKING:  # pragma: no cover - typing helper
    from .embeddings.openai_client import EmbeddingResult
else:  # pragma: no cover - runtime placeholder
    EmbeddingResult = Any

app = typer.Typer(help="Run single-shot code agent experiments against a repository.")
bench_app = typer.Typer(help="Benchmark local shell tooling used by the agent.")
mcp_app = typer.Typer(help="Validate MCP/LSP bridge configuration.")
app.add_typer(bench_app, name="bench")
app.add_typer(mcp_app, name="mcp")
console = Console()
warn_if_missing_binaries(console=console)

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

INDEX_OUTPUT_OPTION = typer.Option(
    Path(".index/faiss/index.faiss"),
    "--output",
    "-o",
    file_okay=True,
    dir_okay=False,
    help="Path where the persisted vector index should be written.",
)

INDEX_BACKEND_OPTION = typer.Option(
    "faiss",
    "--backend",
    case_sensitive=False,
    help="Vector backend to use when indexing the repository.",
)

INDEX_METRIC_OPTION = typer.Option(
    "cosine",
    "--metric",
    case_sensitive=False,
    help="Vector similarity metric for the configured backend.",
)

INDEX_EMBEDDING_MODEL_OPTION = typer.Option(
    "text-embedding-3-small",
    "--embedding-model",
    help="Embedding model identifier used for chunk encoding.",
)

INDEX_GLOB_OPTION = typer.Option(
    ["**/*.py"],
    "--glob",
    help="Glob patterns (relative to the repository) selecting files to index.",
)

BENCH_ITERATIONS_OPTION = typer.Option(
    3,
    "--iterations",
    min=1,
    help="Number of iterations to time for each tool during benchmarking.",
)

BENCH_MAX_RESULTS_OPTION = typer.Option(
    100,
    "--max-results",
    min=1,
    help="Maximum number of results requested from each tool during benchmarking.",
)

INIT_FORCE_OPTION = typer.Option(
    default=False,
    help="Overwrite existing scaffold files if they are already present.",
)

INIT_TARGET_ARGUMENT = typer.Argument(
    Path(),
    file_okay=False,
    resolve_path=True,
    help="Directory where configuration templates should be generated.",
)

METRICS_OUTPUT_OPTION = typer.Option(
    Path("metrics.jsonl"),
    "--output",
    "-o",
    file_okay=True,
    dir_okay=False,
    help="Destination path for computed metrics (JSONL).",
)

EVAL_K_OPTION = typer.Option(
    [5, 10, 20],
    "--k",
    min=1,
    help="Ranking cutoffs used when computing recall/nDCG metrics.",
)

MCP_REQUIRED_OPTION = typer.Option(
    ["MCP_SERVER_COMMAND", "MCP_SERVER_ARGS"],
    "--require",
    help="Environment variables that must be present for MCP connectivity.",
)

DEFAULT_EXECUTOR = default_executor


class EmbeddingClientProtocol(Protocol):
    """Protocol describing the embedding client used by the indexer."""

    def embed(self, texts: Sequence[str]) -> Sequence[EmbeddingResult]:  # pragma: no cover - protocol definition
        """Return embeddings for ``texts``."""
        ...


class VectorStoreProtocol(Protocol):
    """Protocol describing vector store operations used by the CLI indexer."""

    def add(self, records: Sequence[VectorRecord]) -> None:  # pragma: no cover - protocol definition
        """Insert vector ``records`` into the store."""
        ...

    def save(self, path: Path | None = None) -> Path:  # pragma: no cover - protocol definition
        """Persist the store to ``path`` and return the resolved location."""
        ...


def _default_embedding_client_factory(model: str) -> EmbeddingClientProtocol:
    return OpenAIEmbeddingsClient(model=model)


def _default_faiss_factory(dimension: int, metric: str, index_path: Path) -> VectorStoreProtocol:
    return FaissVectorStore(dimension=dimension, metric=metric, index_path=index_path)


EMBEDDING_CLIENT_FACTORY: Callable[[str], EmbeddingClientProtocol] = _default_embedding_client_factory
VECTOR_STORE_FACTORIES: dict[str, Callable[[int, str, Path], VectorStoreProtocol]] = {
    "faiss": _default_faiss_factory,
}

INIT_RUNS_TEMPLATE = """
runs:
  - id: rg-only
    description: Ripgrep-only keyword baseline
    model:
      name: gpt-4.1-mini
      temperature: 0.0
    tools_enabled:
      - ripgrep
    chunking:
      type: fixed
      size_tokens: 256
      overlap_tokens: 64
      languages:
        - python
    tool_limits:
      max_calls: 8
      per_call_timeout_sec: 20
    seed: 7
    replicate: 3

  - id: hybrid
    description: Hybrid keyword + FAISS baseline
    model:
      name: gpt-4.1-mini
      temperature: 0.0
    tools_enabled:
      - ripgrep
      - fd
      - vector.faiss
    chunking:
      type: fixed
      size_tokens: 400
      overlap_tokens: 80
      languages:
        - python
    vector:
      backend: faiss
      index_path: .index/faiss
      metric: cosine
    tool_limits:
      max_calls: 12
      per_call_timeout_sec: 20
    seed: 11
    replicate: 3
""".strip()

INIT_SCENARIOS_TEMPLATE = """
scenarios:
  - id: sample-scenario
    repo_path: ./example-repo
    query: Identify files mentioning authentication middleware
    ground_truth_files:
      - src/auth.py
    metadata:
      issue: SAMPLE-123
      language: python
""".strip()

INIT_ENV_TEMPLATE = """
# Environment variables consumed by Code Agent Experiments CLI
OPENAI_API_KEY=
CAE_EMBEDDING_MODEL=text-embedding-3-small
""".strip()


def _resolve_repo(path: Path | None) -> Path:
    if path is None:
        return Path.cwd().resolve()
    return path.resolve()


def _is_within_repo(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root)
    except ValueError:
        return False
    return True


def _relative_to_root(path: Path, root: Path) -> Path:
    try:
        return path.resolve().relative_to(root)
    except ValueError:
        return path.resolve()


def _resolve_index_files(repository_root: Path, patterns: Sequence[str]) -> list[Path]:
    matched: list[Path] = []
    for pattern in patterns:
        matched.extend(repository_root.glob(pattern))
    unique = {
        candidate.resolve()
        for candidate in matched
        if candidate.is_file() and _is_within_repo(candidate, repository_root)
    }
    return sorted(unique)


def _collect_chunk_entries(
    files: Sequence[Path],
    repository_root: Path,
    chunker: FixedTokenChunker,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for file_path in files:
        try:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
        except OSError as exc:
            console.print(f"[yellow]Skipping unreadable file {file_path}: {exc}[/yellow]")
            continue
        if not text.strip():
            continue
        relative = _relative_to_root(file_path, repository_root)
        chunks = chunker.chunk(text, metadata={"path": str(relative)})
        for index, chunk in enumerate(chunks, start=1):
            entries.append(
                {
                    "id": f"{relative}#{index}",
                    "path": str(relative),
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "metadata": dict(chunk.metadata),
                    "text": chunk.text,
                },
            )
    return entries


def _load_run_config_result(
    runs_file: Path,
    *,
    env_files: Sequence[Path] | None,
) -> tuple[list[RunConfig], Path]:
    try:
        result = load_run_configs(runs_file, env_files=env_files)
    except LoadError as exc:  # pragma: no cover - propagated as CLI error
        raise typer.BadParameter(str(exc)) from exc
    items = list(result.items)
    if not items:
        message = f"No run configurations found in {runs_file}"
        raise typer.BadParameter(message)
    return items, result.source


def _load_scenario_result(
    scenarios_file: Path,
    *,
    env_files: Sequence[Path] | None,
) -> tuple[list[Scenario], Path]:
    try:
        result = load_scenarios(scenarios_file, env_files=env_files)
    except LoadError as exc:  # pragma: no cover - propagated as CLI error
        raise typer.BadParameter(str(exc)) from exc
    items = list(result.items)
    if not items:
        message = f"No scenarios found in {scenarios_file}"
        raise typer.BadParameter(message)
    return items, result.source


def _create_embedding_client(model: str) -> EmbeddingClientProtocol:
    return EMBEDDING_CLIENT_FACTORY(model)


def _create_vector_store(backend: str, *, dimension: int, metric: str, index_path: Path) -> VectorStoreProtocol:
    factory = VECTOR_STORE_FACTORIES.get(backend.lower())
    if factory is None:
        available = ", ".join(sorted(VECTOR_STORE_FACTORIES)) or "<none>"
        message = f"Unknown vector backend '{backend}'. Available: {available}"
        raise typer.BadParameter(message)
    return factory(dimension, metric, index_path)


def _write_template(path: Path, content: str, *, force: bool) -> bool:
    if path.exists() and not force:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")
    return True


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


@app.command()
def init(target: Path = INIT_TARGET_ARGUMENT, force: bool = INIT_FORCE_OPTION) -> None:
    """Bootstrap run/scenario templates and environment scaffolding."""
    target = target.resolve()
    target.mkdir(parents=True, exist_ok=True)
    templates = {
        "runs.yaml": INIT_RUNS_TEMPLATE,
        "scenarios.yaml": INIT_SCENARIOS_TEMPLATE,
        ".env.example": INIT_ENV_TEMPLATE,
    }
    skipped: list[str] = []
    for name, content in templates.items():
        path = target / name
        created = _write_template(path, content, force=force)
        status = "created" if created else "skipped"
        color = "green" if created else "yellow"
        console.print(f"[{color}]{status.capitalize()} {path}[/]")
        if not created:
            skipped.append(name)
    if skipped and not force:
        console.print(
            "[yellow]Some files already existed. Use --force to overwrite them.[/yellow]",
        )
        raise typer.Exit(code=1)
    console.print(f"[green]Project scaffolding written to {target}[/green]")


@app.command()
def index(
    repo: Path | None = REPO_OPTION,
    output: Path = INDEX_OUTPUT_OPTION,
    backend: str = INDEX_BACKEND_OPTION,
    metric: str = INDEX_METRIC_OPTION,
    embedding_model: str = INDEX_EMBEDDING_MODEL_OPTION,
    glob: list[str] = INDEX_GLOB_OPTION,
) -> None:
    """Generate a vector index for the repository using the configured backend."""
    repository_root = _resolve_repo(repo)
    patterns = tuple(glob) if glob else ("**/*.py",)
    unique_files = _resolve_index_files(repository_root, patterns)
    if not unique_files:
        message = "No files matched the provided glob patterns."
        raise typer.BadParameter(message)

    chunker = FixedTokenChunker()
    chunk_entries = _collect_chunk_entries(unique_files, repository_root, chunker)

    if not chunk_entries:
        message = "No chunks were generated from the selected files."
        raise typer.BadParameter(message)

    client = _create_embedding_client(embedding_model)
    embeddings = list(client.embed([entry["text"] for entry in chunk_entries]))
    if not embeddings:
        message = "Embedding client returned no vectors for the generated chunks."
        raise typer.BadParameter(message)

    dimension = len(embeddings[0].embedding)
    index_path = output.resolve()
    vector_store = _create_vector_store(backend, dimension=dimension, metric=metric, index_path=index_path)

    records: list[VectorRecord] = []
    for entry, embedding in zip(chunk_entries, embeddings, strict=False):
        metadata = {
            "path": entry["path"],
            "start_line": entry["start_line"],
            "end_line": entry["end_line"],
        }
        metadata.update(entry["metadata"])
        records.append(
            VectorRecord(
                id=entry["id"],
                vector=list(embedding.embedding),
                metadata=metadata,
            ),
        )

    vector_store.add(records)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    saved_path = vector_store.save(index_path)

    manifest_path = Path(f"{saved_path}.manifest.json")
    manifest = {
        "repository": str(repository_root),
        "backend": backend,
        "metric": metric,
        "embedding_model": embedding_model,
        "files_indexed": len(unique_files),
        "chunk_count": len(records),
        "chunks": [
            {
                "id": record.id,
                "path": record.metadata.get("path"),
                "start_line": record.metadata.get("start_line"),
                "end_line": record.metadata.get("end_line"),
                "metadata": {k: v for k, v in record.metadata.items() if k not in {"path", "start_line", "end_line"}},
            }
            for record in records
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    console.print(f"[green]Indexed {len(unique_files)} files into {saved_path}[/green]")
    console.print(f"[green]Manifest written to {manifest_path}[/green]")


RECORDS_FILE_ARGUMENT = typer.Argument(
    ..., exists=True, file_okay=True, dir_okay=False, resolve_path=True, help="Path to a retrieval records JSONL file.",
)


@app.command(name="eval")
def evaluate(
    records_file: Path = RECORDS_FILE_ARGUMENT,
    scenarios_file: Path = SCENARIOS_FILE_OPTION,
    output: Path = METRICS_OUTPUT_OPTION,
    env_file: list[Path] | None = ENV_FILE_OPTION,
    k: list[int] = EVAL_K_OPTION,
) -> None:
    """Compute retrieval metrics for stored records using scenario definitions."""
    env_files = list(env_file) if env_file else None
    scenarios, scenarios_source = _load_scenario_result(scenarios_file, env_files=env_files)
    scenario_map = {scenario.id: scenario for scenario in scenarios}

    raw_lines = [line for line in records_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not raw_lines:
        message = f"No retrieval records found in {records_file}"
        raise typer.BadParameter(message)

    metrics: list[Metrics] = []
    for index, line in enumerate(raw_lines, start=1):
        try:
            record = RetrievalRecord.model_validate_json(line)
        except Exception as exc:  # pragma: no cover - defensive guard
            message = f"Invalid retrieval record on line {index}: {exc}"
            raise typer.BadParameter(message) from exc
        scenario = scenario_map.get(record.scenario_id)
        if scenario is None:
            message = f"Scenario '{record.scenario_id}' referenced in record {index} not found."
            raise typer.BadParameter(message)
        metrics.append(compute_retrieval_metrics(scenario, record, k_values=k))

    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for entry in metrics:
            handle.write(entry.model_dump_json() + "\n")

    grouped: dict[str, list[Metrics]] = {}
    for entry in metrics:
        grouped.setdefault(entry.strategy, []).append(entry)
    comparison = {name: aggregate_metrics(items) for name, items in grouped.items()}

    _print_strategy_table(comparison)

    summary_path = Path(f"{output}.summary.json")
    summary_payload = {
        "records": len(metrics),
        "strategies": comparison,
        "records_source": str(records_file),
        "scenarios_source": str(scenarios_source),
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    console.print(f"[green]Metrics written to {output}[/green]")
    console.print(f"[green]Summary written to {summary_path}[/green]")


@bench_app.command("tools")
def bench_tools(
    pattern: str = typer.Argument(..., help="Pattern evaluated by shell tools during the benchmark."),
    repo: Path | None = REPO_OPTION,
    iterations: int = BENCH_ITERATIONS_OPTION,
    max_results: int = BENCH_MAX_RESULTS_OPTION,
) -> None:
    """Benchmark local search tools against the provided pattern."""
    repository_root = _resolve_repo(repo)

    bench_cases: list[tuple[str, Callable[[], Sequence[Any]]]] = [
        ("ripgrep", lambda: run_ripgrep(pattern, repository_root, max_count=max_results)),
        ("grep", lambda: run_grep(pattern, repository_root, max_count=max_results)),
        ("fd", lambda: run_fd(pattern, repository_root, limit=max_results)),
        ("find", lambda: run_find(repository_root, name=pattern, include_directories=False)),
    ]

    table = Table(title="Tool Benchmark")
    table.add_column("Tool", style="bold")
    table.add_column("Avg Duration (ms)", justify="right")
    table.add_column("Last Result Count", justify="right")

    for name, runner in bench_cases:
        durations: list[float] = []
        result_count = 0
        for _ in range(iterations):
            start = time.perf_counter()
            results = runner()
            elapsed = time.perf_counter() - start
            durations.append(elapsed)
            result_count = len(results)
        avg_ms = statistics.fmean(durations) * 1000 if durations else 0.0
        table.add_row(name, f"{avg_ms:.2f}", str(result_count))

    console.print(table)


@mcp_app.command("check")
def mcp_check(
    env_file: list[Path] | None = ENV_FILE_OPTION,
    require: list[str] = MCP_REQUIRED_OPTION,
) -> None:
    """Validate MCP bridge environment variables."""
    base_env: Mapping[str, str] = os.environ
    env_files = list(env_file) if env_file else None
    values = load_environment(env_files, base=base_env)

    required = tuple(require) if require else ()
    table = Table(title="MCP Environment Check")
    table.add_column("Variable", style="bold")
    table.add_column("Status")
    table.add_column("Value", overflow="fold")

    missing: list[str] = []
    for key in required:
        value = values.get(key)
        if value:
            table.add_row(key, "[green]set[/green]", value)
        else:
            table.add_row(key, "[red]missing[/red]", "")
            missing.append(key)

    console.print(table)

    if missing:
        console.print(
            "[yellow]Missing required MCP variables: " + ", ".join(missing) + "[/yellow]",
        )
        raise typer.Exit(code=1)

    console.print("[green]All required MCP variables are configured.[/green]")


@app.command("run-scenarios")
def run_scenarios(
    runs_file: Path = RUNS_FILE_OPTION,
    scenarios_file: Path = SCENARIOS_FILE_OPTION,
    output_dir: Path = OUTPUT_DIR_OPTION,
    env_file: list[Path] | None = ENV_FILE_OPTION,
) -> None:
    """Execute a batch of scenarios across the configured run profiles."""
    env_files = list(env_file) if env_file else None
    run_configs, runs_source = _load_run_config_result(runs_file, env_files=env_files)
    scenarios, scenarios_source = _load_scenario_result(scenarios_file, env_files=env_files)

    timestamp = datetime.now(tz=UTC).strftime("%Y%m%d-%H%M%S")
    batch_dir = (output_dir / f"run-{timestamp}").resolve()

    storage = ExperimentStorage(batch_dir)
    orchestrator = ExperimentOrchestrator(DEFAULT_EXECUTOR, storage)
    result = orchestrator.run(run_configs, scenarios)

    console.print(f"[dim]Runs loaded from {runs_source}[/dim]")
    console.print(f"[dim]Scenarios loaded from {scenarios_source}[/dim]")

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
