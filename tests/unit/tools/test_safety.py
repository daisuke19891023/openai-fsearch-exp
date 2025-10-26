"""Tests for runtime binary safety checks."""

from __future__ import annotations

import pytest
from rich.console import Console

from code_agent_experiments.tools import safety
from code_agent_experiments.tools.safety import BinaryCheckResult, ToolingWarning, warn_if_missing_binaries


def test_check_binary_reports_version(monkeypatch: pytest.MonkeyPatch) -> None:
    """check_binary returns resolved version metadata when available."""

    def fake_which(name: str) -> str:
        return f"/usr/bin/{name}"

    def fake_version_command(_args: tuple[str, ...]) -> tuple[int, str, str]:
        return (0, "ripgrep 14.0 (rev hash)", "")

    monkeypatch.setattr(safety.shutil, "which", fake_which)
    monkeypatch.setattr(safety, "_run_version_command", fake_version_command)

    result = safety.check_binary("rg")
    assert result.available is True
    assert result.version == "ripgrep 14.0 (rev hash)"


def test_warn_if_missing_binaries_emits_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing binaries surface ToolingWarning with actionable guidance."""

    def missing_binary(_name: str) -> None:
        return None

    monkeypatch.setattr(safety.shutil, "which", missing_binary)

    with pytest.warns(ToolingWarning, match="rg"):
        warn_if_missing_binaries(names=["rg"], console=None)


def test_warn_if_missing_binaries_logs_to_console(monkeypatch: pytest.MonkeyPatch) -> None:
    """Console output mirrors warning messaging for interactive CLI usage."""
    status = BinaryCheckResult(name="fd", available=False, error="not found on PATH")

    def fake_check_required(names: list[str] | None = None) -> list[BinaryCheckResult]:
        assert names == ["fd"]
        return [status]

    monkeypatch.setattr(safety, "check_required_binaries", fake_check_required)

    console = Console(record=True)

    with pytest.warns(ToolingWarning):
        warn_if_missing_binaries(console=console, names=["fd"])

    output = console.export_text()
    assert "fd" in output
