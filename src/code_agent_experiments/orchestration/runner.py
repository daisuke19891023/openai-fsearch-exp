"""Scenario execution orchestration for experiment runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
import re
import time
from pathlib import Path
from typing import Callable, Sequence, cast

from code_agent_experiments.domain.models import (
    Candidate,
    Metrics,
    RetrievalRecord,
    RunConfig,
    Scenario,
    ToolCall,
    ToolName,
)
from code_agent_experiments.evaluation.metrics import compute_retrieval_metrics, aggregate_metrics
from code_agent_experiments.tools.shell import RipgrepMatch, run_fd, run_ripgrep

__all__ = ["RunSummary", "ScenarioRunner"]

_HIGHLIGHT_LIMIT = 5
_MIN_TOKEN_LENGTH = 3


@dataclass(slots=True)
class RunSummary:
    """Aggregate results for an executed run configuration."""

    run_id: str
    strategy: str
    metrics: list[Metrics]
    metrics_path: Path
    record_paths: list[Path]

    @property
    def aggregates(self) -> dict[str, float | int | dict[int, float] | None]:
        """Return aggregate statistics for the run."""
        return aggregate_metrics(self.metrics)


@dataclass(slots=True)
class _CandidateAccum:
    """Intermediate container to build ``Candidate`` objects."""

    score_keyword: float = 0.0
    score_dense: float = 0.0
    highlights: list[str] = field(default_factory=lambda: cast(list[str], []))
    sources: set[str] = field(default_factory=lambda: cast(set[str], set()))

    def to_candidate(self, path: str, rank: int) -> Candidate:
        features = {
            "highlights": list(self.highlights),
            "sources": sorted(self.sources),
        }
        return Candidate(
            path=path,
            score_keyword=self.score_keyword,
            score_dense=self.score_dense,
            score_rerank=self.score_keyword + self.score_dense,
            rank=rank,
            features=features,
        )


@dataclass(slots=True)
class ScenarioRunner:
    """Execute experiment runs across a collection of scenarios."""

    runs: Sequence[RunConfig]
    scenarios: Sequence[Scenario]
    output_dir: Path
    log: Callable[[str], None] | None = None

    def execute(self) -> list[RunSummary]:
        """Run all configured scenarios and persist artefacts."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        summaries: list[RunSummary] = []
        for run_config in self.runs:
            for replicate_index in range(1, run_config.replicate + 1):
                run_id = _run_instance_id(run_config, replicate_index)
                run_dir = self.output_dir / run_id
                record_dir = run_dir / "records"
                record_dir.mkdir(parents=True, exist_ok=True)
                metrics_path = run_dir / "metrics.jsonl"
                if metrics_path.exists():
                    metrics_path.unlink()
                metrics_for_run: list[Metrics] = []
                record_paths: list[Path] = []
                for scenario in self.scenarios:
                    scenario_instance = _scenario_instance(scenario, replicate_index, run_config.replicate)
                    record, metrics = self._execute_single(run_config, scenario_instance, run_id)
                    record_paths.append(_persist_record(record_dir, record))
                    _append_metric(metrics_path, metrics)
                    metrics_for_run.append(metrics)
                summaries.append(
                    RunSummary(
                        run_id=run_id,
                        strategy=run_config.id,
                        metrics=metrics_for_run,
                        metrics_path=metrics_path,
                        record_paths=record_paths,
                    ),
                )
        return summaries

    def _execute_single(
        self,
        run_config: RunConfig,
        scenario: Scenario,
        run_id: str,
    ) -> tuple[RetrievalRecord, Metrics]:
        """Run a single scenario under ``run_config`` and return artefacts."""
        start_time = time.perf_counter()
        repo_path = Path(scenario.repo_path).expanduser().resolve()
        if not repo_path.exists():
            message = f"Repository path not found: {repo_path}"
            self._emit(f"[red]Scenario {scenario.id} failed:[/red] {message}")
            record = _failure_record(run_config, scenario, run_id, RuntimeError(message))
            metrics = compute_retrieval_metrics(scenario, record)
            return record, metrics

        try:
            candidates, tool_calls = _collect_candidates(run_config, scenario, repo_path)
        except Exception as exc:  # pragma: no cover - defensive guard
            self._emit(f"[red]Scenario {scenario.id} failed:[/red] {exc}")
            record = _failure_record(run_config, scenario, run_id, exc)
        else:
            elapsed_ms = int((time.perf_counter() - start_time) * 1000)
            record = RetrievalRecord(
                scenario_id=scenario.id,
                run_id=run_id,
                strategy=run_config.id,
                candidates=candidates,
                tool_calls=tool_calls,
                elapsed_ms=elapsed_ms,
            )
        metrics = compute_retrieval_metrics(scenario, record)
        return record, metrics

    def _emit(self, message: str) -> None:
        if self.log is not None:
            self.log(message)


def _collect_candidates(
    run_config: RunConfig,
    scenario: Scenario,
    repo_path: Path,
) -> tuple[list[Candidate], list[ToolCall]]:
    tokens = _tokenize_query(scenario.query)
    accumulator: dict[str, _CandidateAccum] = {}
    tool_calls: list[ToolCall] = []

    if "ripgrep" in run_config.tools_enabled:
        matches, call = _keyword_search(repo_path, tokens, scenario.query)
        tool_calls.append(call)
        for match in matches:
            path_key = str(match.path)
            entry = accumulator.setdefault(path_key, _CandidateAccum())
            entry.score_keyword += 1.0
            entry.sources.add("ripgrep")
            if len(entry.highlights) < _HIGHLIGHT_LIMIT:
                entry.highlights.append(f"{match.line_number}:{match.line.strip()}")

    if "fd" in run_config.tools_enabled:
        scores, call = _filename_search(repo_path, tokens, tool_name="fd")
        tool_calls.append(call)
        for path_key, score in scores.items():
            entry = accumulator.setdefault(path_key, _CandidateAccum())
            entry.score_keyword += score
            entry.sources.add("fd")

    if "vector.faiss" in run_config.tools_enabled:
        scores, call = _filename_search(repo_path, tokens, tool_name="vector.faiss", weight_exact=1.0)
        tool_calls.append(call)
        for path_key, score in scores.items():
            entry = accumulator.setdefault(path_key, _CandidateAccum())
            entry.score_dense += score
            entry.sources.add("vector.faiss")

    if not accumulator:
        return [], tool_calls

    ranked_items = sorted(
        accumulator.items(),
        key=lambda item: (item[1].score_keyword + item[1].score_dense, item[1].score_keyword, item[1].score_dense),
        reverse=True,
    )
    candidates: list[Candidate] = []
    for index, (path, entry) in enumerate(ranked_items, start=1):
        candidates.append(entry.to_candidate(path, index))
    return candidates, tool_calls


def _keyword_search(
    repo_path: Path,
    tokens: Sequence[str],
    query: str,
) -> tuple[list[RipgrepMatch], ToolCall]:
    pattern = _compile_pattern(tokens, query)
    started_at = datetime.now(UTC)
    matches = run_ripgrep(pattern, repo_path, ignore_case=True, max_count=200)
    ended_at = datetime.now(UTC)
    call = ToolCall(
        name="ripgrep",
        args={"pattern": pattern, "root": str(repo_path)},
        started_at=started_at,
        ended_at=ended_at,
        success=True,
        stdout_preview=f"{len(matches)} matches",
        stderr_preview=None,
        token_in=0,
        token_out=0,
        latency_ms=int((ended_at - started_at).total_seconds() * 1000),
    )
    return matches, call


def _filename_search(
    repo_path: Path,
    tokens: Sequence[str],
    *,
    tool_name: ToolName,
    weight_exact: float = 0.5,
) -> tuple[dict[str, float], ToolCall]:
    started_at = datetime.now(UTC)
    scores: dict[str, float] = {}
    patterns = list(tokens) if tokens else [_default_pattern(repo_path)]
    for token in patterns:
        matches = run_fd(
            token,
            repo_path,
            include_directories=False,
            hidden=False,
            follow_symlinks=False,
            limit=200,
        )
        token_lower = token.lower()
        for path in matches:
            resolved = str(path.resolve())
            score = scores.get(resolved, 0.0)
            score += 1.0
            if token_lower and token_lower in path.name.lower():
                score += weight_exact
            scores[resolved] = score
    ended_at = datetime.now(UTC)
    call = ToolCall(
        name=tool_name,
        args={"patterns": patterns, "root": str(repo_path)},
        started_at=started_at,
        ended_at=ended_at,
        success=True,
        stdout_preview=f"{len(scores)} matches",
        stderr_preview=None,
        token_in=0,
        token_out=0,
        latency_ms=int((ended_at - started_at).total_seconds() * 1000),
    )
    return scores, call


def _compile_pattern(tokens: Sequence[str], query: str) -> str:
    unique_tokens = list(dict.fromkeys(tokens))
    if unique_tokens:
        escaped = [re.escape(token) for token in unique_tokens]
        return "|".join(escaped)
    query = query.strip()
    if not query:
        return "."
    return re.escape(query)


def _tokenize_query(query: str) -> list[str]:
    tokens = [match.group(0).lower() for match in re.finditer(r"[A-Za-z0-9_]+", query)]
    filtered = [token for token in tokens if len(token) >= _MIN_TOKEN_LENGTH]
    if filtered:
        return filtered[:10]
    if tokens:
        return tokens[:5]
    return []


def _default_pattern(repo_path: Path) -> str:
    return repo_path.name if repo_path.name else "."


def _scenario_instance(scenario: Scenario, replicate: int, total: int) -> Scenario:
    if total <= 1:
        return scenario
    scenario_id = f"{scenario.id}#r{replicate:02d}"
    return scenario.model_copy(update={"id": scenario_id})


def _run_instance_id(run_config: RunConfig, replicate: int) -> str:
    if run_config.replicate <= 1:
        return run_config.id
    return f"{run_config.id}-rep{replicate:02d}"


def _failure_record(
    run_config: RunConfig,
    scenario: Scenario,
    run_id: str,
    error: Exception,
) -> RetrievalRecord:
    now = datetime.now(UTC)
    tool_name = run_config.tools_enabled[0] if run_config.tools_enabled else "ripgrep"
    tool_call = ToolCall(
        name=tool_name,  # type: ignore[arg-type]
        args={},
        started_at=now,
        ended_at=now,
        success=False,
        stdout_preview=None,
        stderr_preview=str(error),
        token_in=0,
        token_out=0,
        latency_ms=0,
    )
    return RetrievalRecord(
        scenario_id=scenario.id,
        run_id=run_id,
        strategy=run_config.id,
        candidates=[],
        tool_calls=[tool_call],
        elapsed_ms=0,
    )


def _persist_record(directory: Path, record: RetrievalRecord) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{record.scenario_id}.json"
    path.write_text(record.model_dump_json(indent=2), encoding="utf-8")
    return path


def _append_metric(path: Path, metrics: Metrics) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = metrics.model_dump_json()
    with path.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")
