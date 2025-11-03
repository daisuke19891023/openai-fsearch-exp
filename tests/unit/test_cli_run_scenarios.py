from __future__ import annotations

import json
from typing import TYPE_CHECKING

from typer.testing import CliRunner

from code_agent_experiments.cli import app
from code_agent_experiments.domain.models import Candidate, RetrievalRecord, RunConfig, Scenario

if TYPE_CHECKING:
    import pytest
    from pathlib import Path

runner = CliRunner()


def _write_run_yaml(path: Path) -> None:
    path.write_text(
        """
        runs:
          - id: rg-only
            model:
              name: test-model
            tools_enabled: [ripgrep]
            replicate: 1
          - id: hybrid
            model:
              name: test-model
            tools_enabled: [ripgrep, fd]
            replicate: 1
        """.strip(),
        encoding="utf-8",
    )


def _write_scenarios_yaml(path: Path) -> None:
    path.write_text(
        """
        scenarios:
          - id: scenario-1
            repo_path: ./repo
            query: Locate failure
            ground_truth_files:
              - src/app.py
        """.strip(),
        encoding="utf-8",
    )


def test_run_scenarios_generates_comparison_table(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """The CLI runs scenarios, persists artefacts, and prints a comparison table."""
    runs_yaml = tmp_path / "runs.yaml"
    scenarios_yaml = tmp_path / "scenarios.yaml"
    _write_run_yaml(runs_yaml)
    _write_scenarios_yaml(scenarios_yaml)

    def executor(run_cfg: RunConfig, scenario: Scenario, replicate_index: int) -> RetrievalRecord:
        run_id = f"{run_cfg.id}-rep{replicate_index + 1:02d}"
        candidates = [Candidate(path=scenario.ground_truth_files[0], rank=1)]
        if run_cfg.id == "hybrid":
            candidates.append(Candidate(path="extra/file.py", rank=2))
        return RetrievalRecord(
            scenario_id=scenario.id,
            run_id=run_id,
            strategy=run_cfg.id,
            candidates=candidates,
            tool_calls=[],
            elapsed_ms=500,
        )

    monkeypatch.setattr(
        "code_agent_experiments.cli._build_cli_executor",
        lambda: executor,
    )

    output_dir = tmp_path / "artifacts"
    result = runner.invoke(
        app,
        [
            "run-scenarios",
            "--runs-file",
            str(runs_yaml),
            "--scenarios-file",
            str(scenarios_yaml),
            "--output-dir",
            str(output_dir),
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    assert "Strategy Comparison" in result.stdout
    assert "rg-only" in result.stdout
    assert "hybrid" in result.stdout

    batch_dirs = list(output_dir.iterdir())
    assert len(batch_dirs) == 1
    batch_dir = batch_dirs[0]

    metrics_path = batch_dir / "metrics.jsonl"
    summary_path = batch_dir / "comparison.json"
    assert metrics_path.exists()
    assert summary_path.exists()

    metrics_lines = metrics_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(metrics_lines) == 2

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert set(summary["strategies"].keys()) == {"rg-only", "hybrid"}
