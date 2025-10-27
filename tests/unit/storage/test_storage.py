from __future__ import annotations

import json
import sqlite3
from typing import TYPE_CHECKING

from code_agent_experiments.domain.models import (
    Candidate,
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


def _run_config(run_id: str) -> RunConfig:
    return RunConfig(
        id=run_id,
        model=ModelConfig(name="test-model"),
        tools_enabled=["ripgrep"],
        tool_limits=ToolLimits(max_calls=2, per_call_timeout_sec=3),
    )


def _scenario(scenario_id: str) -> Scenario:
    return Scenario(
        id=scenario_id,
        repo_path="./repo",
        query="Locate bug",
        ground_truth_files=["src/app.py"],
    )


def _record(run_config: RunConfig, scenario: Scenario) -> RetrievalRecord:
    return RetrievalRecord(
        scenario_id=scenario.id,
        run_id=f"{run_config.id}-rep01",
        strategy=run_config.id,
        candidates=[Candidate(path=scenario.ground_truth_files[0], rank=1)],
        tool_calls=[],
        elapsed_ms=1000,
    )


def test_sqlite_roundtrip_matches_json_artifacts(tmp_path: Path) -> None:
    """SQLite payloads align with the emitted JSON artefacts."""
    run_config = _run_config("baseline")
    scenario = _scenario("scenario-a")

    def executor(
        run_config: RunConfig,
        scenario: Scenario,
        replicate_index: int,
    ) -> RetrievalRecord:
        del replicate_index
        return _record(run_config, scenario)

    storage = ExperimentStorage(tmp_path)
    orchestrator = ExperimentOrchestrator(executor, storage)
    result = orchestrator.run([run_config], [scenario])

    metrics_lines = [
        line
        for line in result.metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    records_lines = [
        line
        for line in result.records_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    with sqlite3.connect(storage.db_path) as connection:
        connection.row_factory = sqlite3.Row
        metrics_rows = connection.execute(
            "SELECT payload FROM metrics ORDER BY id",
        ).fetchall()
        record_rows = connection.execute(
            "SELECT payload FROM retrieval_records ORDER BY id",
        ).fetchall()
        run_config_rows = connection.execute(
            "SELECT payload FROM run_configs",
        ).fetchall()
        scenario_rows = connection.execute(
            "SELECT payload FROM scenarios",
        ).fetchall()

    metrics_payloads = [row["payload"] for row in metrics_rows]
    record_payloads = [row["payload"] for row in record_rows]

    assert metrics_payloads == metrics_lines
    assert record_payloads == records_lines

    # Round-trip validation ensures the stored JSON can be reconstructed.
    for payload in metrics_payloads:
        assert payload
        _ = json.loads(payload)

    for payload in record_payloads:
        assert payload
        _ = json.loads(payload)

    assert {row["payload"] for row in run_config_rows} == {run_config.model_dump_json()}
    assert {row["payload"] for row in scenario_rows} == {scenario.model_dump_json()}
