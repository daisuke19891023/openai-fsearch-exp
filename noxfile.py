from __future__ import annotations

from pathlib import Path
import platform
import sys
from typing import TYPE_CHECKING

import nox

if TYPE_CHECKING:
    from nox.sessions import Session

nox.options.default_venv_backend = "uv"
nox.options.reuse_existing_virtualenvs = True

COVER_MIN = 60


def has_test_targets() -> bool:
    """Check if there are any Python files in the src directory to test."""
    src_path = Path("src")
    if not src_path.exists():
        return False

    return any(src_path.glob("**/*.py"))


def constraints(session: Session) -> Path:
    """Generate constraints file path for the session."""
    filename = f"python{session.python}-{sys.platform}-{platform.machine()}.txt"
    return Path("constraints", filename)


@nox.session(python=["3.13"], venv_backend="uv")
def lock(session: Session) -> None:
    """Lock dependencies."""
    filename = constraints(session)
    filename.parent.mkdir(exist_ok=True)
    session.run(
        "uv",
        "pip",
        "compile",
        "pyproject.toml",
        "--upgrade",
        "--quiet",
        "--all-extras",
        f"--output-file={filename}",
    )


@nox.session(python=["3.13"], tags=["lint"])
def lint(session: Session) -> None:
    """Run linting with Ruff."""
    session.install("-c", constraints(session).as_posix(), "ruff")
    session.run("ruff", "check", "--fix")


@nox.session(python=["3.13"], tags=["format"])
def format_code(session: Session) -> None:
    """Format code with Ruff."""
    session.install("-c", constraints(session).as_posix(), "ruff")
    session.run("ruff", "format")


@nox.session(python=["3.13"], tags=["sort"])
def sort(session: Session) -> None:
    """Sort imports with Ruff."""
    session.install("-c", constraints(session).as_posix(), "ruff")
    session.run("ruff", "check", "--select", "I", "--fix")


@nox.session(python=["3.13"], tags=["typing"])
def typing(session: Session) -> None:
    """Run type checking with Pyright."""
    session.install("-c", constraints(session).as_posix(), ".[dev]")
    session.run("pyright")


@nox.session(python=["3.13"], tags=["test"])
def test(session: Session) -> None:
    """Run pytest if test target files exist in src directory."""
    if not has_test_targets():
        session.skip("No test targets found in src directory")

    session.install("-c", constraints(session).as_posix(), ".[dev]")
    session.run("pytest", "--cov=code_agent_experiments", f"--cov-fail-under={COVER_MIN}")


@nox.session(python=["3.13"], tags=["ci"])
def ci(session: Session) -> None:
    """Run all CI checks."""
    session.notify("lint")
    session.notify("sort")
    session.notify("format_code")
    session.notify("typing")
    session.notify("test")


@nox.session(python=["3.13"], tags=["all"])
def all_checks(session: Session) -> None:
    """Run all quality checks."""
    session.notify("ci")
