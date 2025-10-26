"""Tests for the `cae init` command."""

from pathlib import Path

from typer.testing import CliRunner

from code_agent_experiments.cli import app


runner = CliRunner()


def test_init_generates_templates_and_requires_force(tmp_path: Path) -> None:
    """The init command writes templates and requires --force to overwrite."""
    result = runner.invoke(app, ["init", str(tmp_path)], catch_exceptions=False)

    assert result.exit_code == 0
    runs_file = tmp_path / "runs.yaml"
    scenarios_file = tmp_path / "scenarios.yaml"
    env_file = tmp_path / ".env.example"
    assert runs_file.exists()
    assert scenarios_file.exists()
    assert env_file.exists()

    repeat = runner.invoke(app, ["init", str(tmp_path)], catch_exceptions=False)
    assert repeat.exit_code == 1
    assert "Use --force" in repeat.stdout

    forced = runner.invoke(app, ["init", str(tmp_path), "--force"], catch_exceptions=False)
    assert forced.exit_code == 0
    assert "Project scaffolding" in forced.stdout
