"""Tests for the `cae mcp check` command."""

from pathlib import Path
from typing import TYPE_CHECKING

from typer.testing import CliRunner

from code_agent_experiments.cli import app


if TYPE_CHECKING:
    import pytest


runner = CliRunner()


def test_mcp_check_reads_env_file(tmp_path: Path) -> None:
    """The MCP check command reads variables from .env files."""
    env_file = tmp_path / "mcp.env"
    env_file.write_text("MCP_SERVER_COMMAND=uvx\nMCP_SERVER_ARGS=serve\n", encoding="utf-8")

    result = runner.invoke(
        app,
        ["mcp", "check", "--env-file", str(env_file)],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    assert "All required MCP variables" in result.stdout


def test_mcp_check_reports_missing(monkeypatch: "pytest.MonkeyPatch") -> None:
    """When required variables are missing the command exits with failure."""
    monkeypatch.delenv("MCP_SERVER_COMMAND", raising=False)
    monkeypatch.delenv("MCP_SERVER_ARGS", raising=False)

    result = runner.invoke(app, ["mcp", "check"], catch_exceptions=False)

    assert result.exit_code == 1
    assert "Missing required MCP variables" in result.stdout
