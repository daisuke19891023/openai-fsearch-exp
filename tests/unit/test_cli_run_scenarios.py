from __future__ import annotations

import json
from typing import TYPE_CHECKING

from typer.testing import CliRunner

from code_agent_experiments.cli import app

if TYPE_CHECKING:
    import pathlib


runner = CliRunner()


def _write_run_config(path: pathlib.Path) -> None:
    path.write_text(
        """
        runs:
          - id: rg-only
            model:
              name: offline
            tools_enabled: [ripgrep]
          - id: hybrid
            model:
              name: offline
            tools_enabled: [ripgrep, vector.faiss]
        """,
        encoding="utf-8",
    )


def _write_scenarios(path: pathlib.Path, repo: pathlib.Path) -> None:
    path.write_text(
        f"""
        scenarios:
          - id: scenario-1
            repo_path: {repo}
            query: add helper
            ground_truth_files:
              - src/app.py
        """,
        encoding="utf-8",
    )


def _create_repo(root: pathlib.Path) -> pathlib.Path:
    repo = root / "repo"
    src = repo / "src"
    src.mkdir(parents=True)
    (src / "app.py").write_text("""def add(a, b):\n    return a + b\n""", encoding="utf-8")
    (src / "add_helper.py").write_text("""def add_helper(x, y):\n    return add(x, y)\n""", encoding="utf-8")
    return repo



def test_cli_run_scenarios_executes_and_reports(tmp_path: pathlib.Path) -> None:
    """Invoke the CLI command and render a comparison table."""
    repo = _create_repo(tmp_path)
    runs_yaml = tmp_path / "runs.yaml"
    scenarios_yaml = tmp_path / "scenarios.yaml"
    _write_run_config(runs_yaml)
    _write_scenarios(scenarios_yaml, repo)

    output_dir = tmp_path / "artifacts"
    result = runner.invoke(
        app,
        [
            "run-scenarios",
            "--runs",
            str(runs_yaml),
            "--scenarios",
            str(scenarios_yaml),
            "--output-dir",
            str(output_dir),
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    assert "Run Comparison" in result.stdout
    assert "rg-only" in result.stdout
    assert "hybrid" in result.stdout

    rg_metrics = output_dir / "rg-only" / "metrics.jsonl"
    hybrid_metrics = output_dir / "hybrid" / "metrics.jsonl"
    assert rg_metrics.exists()
    assert hybrid_metrics.exists()

    with rg_metrics.open(encoding="utf-8") as handle:
        entries = [json.loads(line) for line in handle if line.strip()]
    assert entries
    assert entries[0]["run_id"] == "rg-only"

    with hybrid_metrics.open(encoding="utf-8") as handle:
        hybrid_entries = [json.loads(line) for line in handle if line.strip()]
    assert hybrid_entries
    assert hybrid_entries[0]["run_id"] == "hybrid"
