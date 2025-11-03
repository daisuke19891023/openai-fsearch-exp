from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from code_agent_experiments.agent.responses import AgentRunResult
from code_agent_experiments.domain.models import (
    ModelConfig,
    RetrievalRecord,
    RunConfig,
    Scenario,
    ToolCall,
    ToolLimits,
)
from code_agent_experiments.domain.tool_schemas import RipgrepToolMatch, RipgrepToolOutput
from code_agent_experiments.orchestration import (
    AgentScenarioExecutor,
    ExperimentOrchestrator,
    ExperimentStorage,
)

if TYPE_CHECKING:
    from pathlib import Path

    from code_agent_experiments.agent.tooling import ToolRegistry


class _StubAgent:
    """Test double returning predetermined tool telemetry."""

    def __init__(self, tool_calls: list[ToolCall]) -> None:
        self._tool_calls = tool_calls

    def run(self, query: str) -> AgentRunResult:
        _ = query
        return AgentRunResult(
            response_id="resp-1",
            output_text="completed",
            tool_calls=list(self._tool_calls),
            limit_reached=False,
        )


def _run_config(run_id: str) -> RunConfig:
    return RunConfig(
        id=run_id,
        model=ModelConfig(name="test-model"),
        tools_enabled=["ripgrep"],
        tool_limits=ToolLimits(max_calls=4, per_call_timeout_sec=5),
    )


def _scenario(repo: str, scenario_id: str, ground_truth: str) -> Scenario:
    return Scenario(
        id=scenario_id,
        repo_path=repo,
        query="Find matching files",
        ground_truth_files=[ground_truth],
    )


def _ripgrep_call(paths: list[str]) -> ToolCall:
    matches = [
        RipgrepToolMatch(
            path=path,
            line_number=index + 1,
            line=f"line {index}",
            submatches=("needle",),
        )
        for index, path in enumerate(paths)
    ]
    payload = RipgrepToolOutput(matches=matches, truncated=False).model_dump_json()
    started = datetime.now(tz=UTC)
    return ToolCall(
        name="ripgrep",
        args={"pattern": "needle"},
        started_at=started,
        ended_at=started,
        success=True,
        stdout_preview=payload,
        stderr_preview=None,
        token_in=0,
        token_out=0,
        latency_ms=12,
    )


def test_agent_executor_persists_records_for_multiple_scenarios(tmp_path: Path) -> None:
    """Integration run persists aggregated artefacts for each scenario."""
    repo_root = tmp_path / "repo"
    (repo_root / "src").mkdir(parents=True)
    file_a = repo_root / "src" / "app.py"
    file_b = repo_root / "src" / "extra.py"
    file_a.write_text("alpha", encoding="utf-8")
    file_b.write_text("beta", encoding="utf-8")

    tool_calls = {
        "scenario-one": [_ripgrep_call([str(file_a)])],
        "scenario-two": [_ripgrep_call([str(file_b), str(file_a)])],
    }

    def agent_factory(
        _run_config: RunConfig,
        scenario: Scenario,
        _repository_root: Path,
        _tool_registry: ToolRegistry,
        _system_prompt: str,
    ) -> _StubAgent:
        return _StubAgent(tool_calls[scenario.id])

    executor = AgentScenarioExecutor(agent_factory=agent_factory)
    storage = ExperimentStorage(tmp_path / "artifacts")
    orchestrator = ExperimentOrchestrator(executor, storage)

    run_config = _run_config("ripgrep-only")
    scenarios = [
        _scenario(str(repo_root), "scenario-one", "src/app.py"),
        _scenario(str(repo_root), "scenario-two", "src/extra.py"),
    ]

    result = orchestrator.run([run_config], scenarios)

    assert not result.failures
    assert result.records_path.exists()
    assert result.metrics_path.exists()

    records = [
        RetrievalRecord.model_validate_json(line)
        for line in result.records_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert len(records) == 2
    first_record = next(item for item in records if item.scenario_id == "scenario-one")
    second_record = next(item for item in records if item.scenario_id == "scenario-two")

    assert first_record.candidates[0].path == "src/app.py"
    assert second_record.candidates[0].path == "src/extra.py"
    assert second_record.candidates[1].path == "src/app.py"

    metrics = [
        json.loads(line)
        for line in result.metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(metrics) == 2
    assert all(metric["scenario_id"] in {"scenario-one", "scenario-two"} for metric in metrics)

    sqlite_path = storage.db_path
    assert sqlite_path.exists()


def test_agent_executor_records_failures(tmp_path: Path) -> None:
    """Failures from the agent are persisted with telemetry."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    good_call = _ripgrep_call([str(repo_root / "hit.py")])
    tool_calls = {"scenario-success": [good_call]}

    def agent_factory(
        _run_config: RunConfig,
        scenario: Scenario,
        _repository_root: Path,
        _tool_registry: ToolRegistry,
        _system_prompt: str,
    ) -> _StubAgent:
        if scenario.id == "scenario-failure":
            raise RuntimeError("boom")
        return _StubAgent(tool_calls[scenario.id])

    executor = AgentScenarioExecutor(agent_factory=agent_factory)
    storage = ExperimentStorage(tmp_path / "artifacts")
    orchestrator = ExperimentOrchestrator(executor, storage)

    run_config = _run_config("ripgrep-only")
    scenarios = [
        _scenario(str(repo_root), "scenario-success", "hit.py"),
        _scenario(str(repo_root), "scenario-failure", "missing.py"),
    ]

    result = orchestrator.run([run_config], scenarios)

    assert len(result.failures) == 1
    assert result.failures_path is not None
    failure_payloads = result.failures_path.read_text(encoding="utf-8").splitlines()
    assert len(failure_payloads) == 1

    records = [
        RetrievalRecord.model_validate_json(line)
        for line in result.records_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(records) == 2
    failed_record = next(item for item in records if item.scenario_id == "scenario-failure")
    assert not failed_record.candidates

    metrics = [
        json.loads(line)
        for line in result.metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(metrics) == 2
    failed_metrics = next(item for item in metrics if item["scenario_id"] == "scenario-failure")
    assert all(value == 0.0 for value in failed_metrics["recall_at_k"].values())
