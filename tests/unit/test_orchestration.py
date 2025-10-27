from __future__ import annotations

from typing import TYPE_CHECKING

import sqlite3

from code_agent_experiments.domain.models import (
    Candidate,
    Metrics,
    ModelConfig,
    RetrievalRecord,
    RunConfig,
    Scenario,
    ToolLimits,
)
from code_agent_experiments.orchestration import (
    ExperimentOrchestrator,
    ExperimentStorage,
)

if TYPE_CHECKING:
    from pathlib import Path


def _run_config(run_id: str, replicate: int = 1) -> RunConfig:
    return RunConfig(
        id=run_id,
        model=ModelConfig(name="test-model"),
        tools_enabled=["ripgrep"],
        tool_limits=ToolLimits(max_calls=4, per_call_timeout_sec=5),
        replicate=replicate,
    )


def _scenario(scenario_id: str) -> Scenario:
    return Scenario(
        id=scenario_id,
        repo_path="./repo",
        query="Find failure",
        ground_truth_files=["src/app.py"],
    )


def _success_record(run_config: RunConfig, scenario: Scenario, replicate_index: int) -> RetrievalRecord:
    run_id = f"{run_config.id}-rep{replicate_index + 1:02d}"
    return RetrievalRecord(
        scenario_id=scenario.id,
        run_id=run_id,
        strategy=run_config.id,
        candidates=[
            Candidate(path=scenario.ground_truth_files[0], rank=1),
            Candidate(path="src/extra.py", rank=2),
        ],
        tool_calls=[],
        elapsed_ms=750,
    )


def test_orchestrator_persists_records_and_metrics(tmp_path: Path) -> None:
    """Metrics and retrieval records are persisted for each replicate."""
    run_config = _run_config("rg-only", replicate=2)
    scenario = _scenario("scenario-1")

    def executor(run_config: RunConfig, scenario: Scenario, replicate_index: int) -> RetrievalRecord:
        return _success_record(run_config, scenario, replicate_index)

    storage = ExperimentStorage(tmp_path)
    orchestrator = ExperimentOrchestrator(executor, storage)
    result = orchestrator.run([run_config], [scenario])

    metrics_lines = result.metrics_path.read_text(encoding="utf-8").strip().splitlines()
    records_lines = result.records_path.read_text(encoding="utf-8").strip().splitlines()

    assert len(metrics_lines) == 2
    assert len(records_lines) == 2
    assert set(result.per_run) == {"rg-only-rep01", "rg-only-rep02"}
    assert result.comparison["rg-only"]["scenario_count"] == 2

    with sqlite3.connect(storage.db_path) as connection:
        metrics_count = connection.execute("SELECT COUNT(*) FROM metrics").fetchone()[0]
        records_count = connection.execute("SELECT COUNT(*) FROM retrieval_records").fetchone()[0]

    assert metrics_count == 2
    assert records_count == 2


def test_orchestrator_records_failures(tmp_path: Path) -> None:
    """Failures are captured and produce empty metrics allowing aggregation."""
    run_config = _run_config("rg-only", replicate=2)
    scenario = _scenario("scenario-2")

    def executor(run_config: RunConfig, scenario: Scenario, replicate_index: int) -> RetrievalRecord:
        if replicate_index == 1:
            raise RuntimeError("boom")
        return _success_record(run_config, scenario, replicate_index)

    storage = ExperimentStorage(tmp_path)
    orchestrator = ExperimentOrchestrator(executor, storage)
    result = orchestrator.run([run_config], [scenario])

    assert len(result.failures) == 1
    assert result.failures_path is not None
    failure_log = result.failures_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(failure_log) == 1

    metrics_objects = [
        Metrics.model_validate_json(line)
        for line in result.metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    failed_metric = next(item for item in metrics_objects if item.run_id.endswith("rep02"))
    assert all(value == 0.0 for value in failed_metric.recall_at_k.values())
    assert result.comparison["rg-only"]["scenario_count"] == 2

    with sqlite3.connect(storage.db_path) as connection:
        failure_count = connection.execute("SELECT COUNT(*) FROM failures").fetchone()[0]

    assert failure_count == 1
