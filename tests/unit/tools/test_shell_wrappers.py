"""Tests for shell tool wrappers."""

from __future__ import annotations

from pathlib import Path

import pytest

from code_agent_experiments.tools import (
    BinaryNotFoundError,
    RipgrepMatch,
    ensure_binary,
    run_fd,
    run_find,
    run_grep,
    run_ripgrep,
)
from code_agent_experiments.tools import shell


@pytest.fixture
def sample_repo(tmp_path: Path) -> Path:
    """Create a small repository layout for shell tool tests."""
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "tests").mkdir()
    (repo / ".gitignore").write_text("ignored.txt\n", encoding="utf-8")
    (repo / "src" / "main.py").write_text("""def important():\n    pass  # TODO implement\n""", encoding="utf-8")
    (repo / "tests" / "test_app.py").write_text("""def test_dummy():\n    assert True\n""", encoding="utf-8")
    (repo / "ignored.txt").write_text("should not appear\n", encoding="utf-8")
    return repo


def test_run_ripgrep_returns_structured_matches(sample_repo: Path) -> None:
    """Ripgrep wrapper should surface structured match objects."""
    matches = run_ripgrep("TODO", sample_repo)
    assert matches, "expected at least one match"
    first = matches[0]
    assert isinstance(first, RipgrepMatch)
    assert first.path.exists()
    assert first.path.relative_to(sample_repo) == Path("src/main.py")
    assert first.line_number > 0
    assert "TODO" in first.line
    assert "TODO" in first.submatches[0]


def test_run_fd_python_fallback_respects_gitignore(sample_repo: Path) -> None:
    """Fd fallback honours gitignore entries and hidden toggles."""
    results = run_fd("main", sample_repo)
    relative = sorted(path.relative_to(sample_repo) for path in results)
    assert relative == [Path("src/main.py")]

    hidden_file = sample_repo / ".hidden.py"
    hidden_file.write_text("print('hidden')\n", encoding="utf-8")
    assert all(hidden_file != path for path in run_fd("hidden", sample_repo))
    included_hidden = run_fd("hidden", sample_repo, hidden=True)
    assert hidden_file in included_hidden


def test_run_grep_matches_format(sample_repo: Path) -> None:
    """Grep wrapper returns matches with file metadata."""
    matches = run_grep("TODO", sample_repo)
    assert matches
    assert matches[0].path.relative_to(sample_repo) == Path("src/main.py")


def test_run_find_filters_by_extension(sample_repo: Path) -> None:
    """Find wrapper should locate files matching the provided pattern."""
    results = run_find(sample_repo, name="*.py")
    relative = sorted(path.relative_to(sample_repo) for path in results)
    assert Path("ignored.txt") not in relative
    assert Path("src/main.py") in relative
    assert Path("tests/test_app.py") in relative


def test_ensure_binary_raises_for_unknown() -> None:
    """ensure_binary raises when a binary cannot be located."""
    with pytest.raises(BinaryNotFoundError):
        ensure_binary("definitely-not-a-binary")


def test_run_ripgrep_globs_and_invalid_pattern(sample_repo: Path) -> None:
    """Ripgrep respects glob filters and validates patterns eagerly."""
    extra = sample_repo / "tests" / "test_extra.py"
    extra.write_text("""# TODO: should be excluded by glob\n""", encoding="utf-8")

    scoped = run_ripgrep("TODO", sample_repo, globs=["src/*.py"])
    assert scoped
    assert all(match.path.relative_to(sample_repo).parts[0] == "src" for match in scoped)

    with pytest.raises(ValueError, match="Invalid ripgrep pattern"):
        run_ripgrep("[", sample_repo)


def test_run_fd_includes_directories_and_respects_limit(sample_repo: Path) -> None:
    """Fd wrapper can include directories and enforce a limit."""
    include_dirs = run_fd("src", sample_repo, include_directories=True)
    relative_dirs = {path.relative_to(sample_repo) for path in include_dirs}
    assert Path("src") in relative_dirs

    unbounded = run_fd("test", sample_repo, include_directories=True)
    assert len(unbounded) > 1
    limited = run_fd("test", sample_repo, include_directories=True, limit=1)
    assert len(limited) == 1


def test_run_ripgrep_enforces_output_cap(monkeypatch: pytest.MonkeyPatch, sample_repo: Path) -> None:
    """Ripgrep wrapper stops at the configured output clamp."""
    monkeypatch.setattr(shell, "MAX_TEXT_MATCHES", 3)
    target = sample_repo / "src" / "overflow.py"
    target.write_text("\n".join("TODO" for _ in range(10)), encoding="utf-8")

    matches = run_ripgrep("TODO", sample_repo)
    assert len(matches) == 3


def test_run_fd_enforces_output_cap(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Fd wrapper clamps runaway directory enumerations."""
    monkeypatch.setattr(shell, "MAX_PATH_RESULTS", 5)
    repo = tmp_path / "repo"
    repo.mkdir()
    for index in range(20):
        (repo / f"file_{index}.txt").write_text("hello\n", encoding="utf-8")

    results = run_fd("file", repo)
    assert len(results) == 5


def test_run_find_enforces_output_cap(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Find wrapper clamps runaway directory enumerations."""
    monkeypatch.setattr(shell, "MAX_PATH_RESULTS", 4)
    repo = tmp_path / "repo"
    repo.mkdir()
    for index in range(10):
        (repo / f"nested_{index}.py").write_text("print('hi')\n", encoding="utf-8")

    results = run_find(repo, name="*.py")
    assert len(results) == 4
