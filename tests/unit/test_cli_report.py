"""Tests for the `cae report` CLI command."""

from __future__ import annotations

from typing import TYPE_CHECKING

from typer.testing import CliRunner

from code_agent_experiments.cli import app
from code_agent_experiments.domain.models import Metrics

if TYPE_CHECKING:
    import pathlib


runner = CliRunner()


def _write_metrics(file_path: pathlib.Path) -> None:
    records = [
        Metrics(
            scenario_id="scenario-1",
            run_id="run-123",
            strategy="keyword",
            k_values=[1, 5],
            recall_at_k={1: 1.0, 5: 1.0},
            ndcg_at_k={1: 1.0, 5: 1.0},
            mrr=1.0,
            tool_calls=2,
            wall_ms=1500,
            cost_usd=None,
        ),
        Metrics(
            scenario_id="scenario-2",
            run_id="run-123",
            strategy="keyword",
            k_values=[1, 5],
            recall_at_k={1: 0.0, 5: 0.4},
            ndcg_at_k={1: 0.0, 5: 0.3},
            mrr=0.0,
            tool_calls=1,
            wall_ms=800,
            cost_usd=0.12,
        ),
    ]
    file_path.write_text(
        "\n".join(metric.model_dump_json() for metric in records),
        encoding="utf-8",
    )


def test_report_generates_markdown_file(tmp_path: pathlib.Path) -> None:
    """Generate a Markdown report with aggregate statistics."""
    metrics_file = tmp_path / "metrics.jsonl"
    _write_metrics(metrics_file)

    output_file = tmp_path / "report.md"
    result = runner.invoke(
        app,
        [
            "report",
            str(metrics_file),
            "--output",
            str(output_file),
            "--format",
            "markdown",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    contents = output_file.read_text(encoding="utf-8")
    assert "Mean MRR" in contents
    assert "Scenario scenario-1" in contents
    assert "Total Tool Calls" in contents
