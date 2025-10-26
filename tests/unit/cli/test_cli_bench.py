"""Tests for the `cae bench tools` command."""

from pathlib import Path

from typer.testing import CliRunner

from code_agent_experiments.cli import app


runner = CliRunner()


def test_bench_tools_renders_table(tmp_path: Path) -> None:
    """The bench command renders a table summarising tool timings."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "file.txt").write_text("target marker\n", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "bench",
            "tools",
            "target",
            "--repo",
            str(repo),
            "--iterations",
            "1",
            "--max-results",
            "10",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    assert "Tool Benchmark" in result.stdout
    assert "ripgrep" in result.stdout
