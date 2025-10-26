"""Tests for the `cae eval` command."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from code_agent_experiments.cli import app
from code_agent_experiments.domain.models import Candidate, RetrievalRecord


runner = CliRunner()


def _write_scenarios(path: Path) -> None:
    """Create a minimal scenarios YAML document for tests."""
    path.write_text(
        """
        scenarios:
          - id: scenario-1
            repo_path: ./repo
            query: Find the answer helper
            ground_truth_files:
              - sample.py
        """.strip(),
        encoding="utf-8",
    )


def _write_records(path: Path) -> None:
    """Write a single retrieval record covering the sample scenario."""
    record = RetrievalRecord(
        scenario_id="scenario-1",
        run_id="rg-only-rep01",
        strategy="rg-only",
        candidates=[Candidate(path="sample.py", rank=1)],
        tool_calls=[],
        elapsed_ms=2500,
    )
    path.write_text(record.model_dump_json() + "\n", encoding="utf-8")


def test_eval_generates_metrics_and_summary(tmp_path: Path) -> None:
    """The eval command computes metrics and writes a JSON summary."""
    records_file = tmp_path / "records.jsonl"
    scenarios_file = tmp_path / "scenarios.yaml"
    output_file = tmp_path / "metrics.jsonl"
    _write_records(records_file)
    _write_scenarios(scenarios_file)

    result = runner.invoke(
        app,
        [
            "eval",
            str(records_file),
            "--scenarios-file",
            str(scenarios_file),
            "--output",
            str(output_file),
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    metrics_lines = [line for line in output_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(metrics_lines) == 1
    payload = json.loads(metrics_lines[0])
    assert payload["scenario_id"] == "scenario-1"
    assert payload["recall_at_k"]["5"] == 1

    summary_path = Path(f"{output_file}.summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["records"] == 1
    assert "rg-only" in summary["strategies"]
