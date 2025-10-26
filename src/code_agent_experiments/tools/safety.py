"""Runtime safety checks for external command dependencies."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import shutil
import warnings
from typing import TYPE_CHECKING, Iterable, Sequence

if TYPE_CHECKING:
    from rich.console import Console

__all__ = [
    "BinaryCheckResult",
    "ToolingWarning",
    "check_binary",
    "check_required_binaries",
    "warn_if_missing_binaries",
]

CHECK_TIMEOUT_SECONDS = 5.0
"""Maximum time to wait for ``--version`` calls."""


class ToolingWarning(UserWarning):
    """Warning emitted when recommended tooling is unavailable."""


@dataclass(frozen=True, slots=True)
class BinaryCheckResult:
    """Outcome of probing an external binary."""

    name: str
    available: bool
    version: str | None = None
    error: str | None = None

    @property
    def message(self) -> str:
        """Human readable status string."""
        if self.available:
            version = self.version or "unknown version"
            return f"{self.name} detected ({version})"
        reason = self.error or "not found on PATH"
        return f"{self.name} unavailable: {reason}"


def check_binary(name: str, *, version_args: Sequence[str] | None = None) -> BinaryCheckResult:
    """Inspect ``name`` returning availability and version metadata."""
    resolved = shutil.which(name)
    if not resolved:
        return BinaryCheckResult(name=name, available=False, error="not found on PATH")

    args = [resolved]
    if version_args:
        args.extend(version_args)
    else:
        args.append("--version")

    try:
        returncode, stdout, stderr = _run_version_command(tuple(args))
    except TimeoutError as exc:
        return BinaryCheckResult(name=name, available=False, error=str(exc))
    except OSError as exc:  # pragma: no cover - defensive
        return BinaryCheckResult(name=name, available=False, error=str(exc))

    if returncode != 0:
        return BinaryCheckResult(name=name, available=False, error=stderr.strip() or stdout.strip())

    output = stdout.strip() or stderr.strip()
    version_line = output.splitlines()[0] if output else ""
    return BinaryCheckResult(name=name, available=True, version=version_line or None)


async def _async_version_probe(args: Sequence[str]) -> tuple[int, str, str]:
    process = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=CHECK_TIMEOUT_SECONDS)
    except TimeoutError as exc:
        process.kill()
        await process.communicate()
        message = f"version probe timed out after {CHECK_TIMEOUT_SECONDS:.1f}s"
        raise TimeoutError(message) from exc
    returncode = process.returncode if process.returncode is not None else -1
    return (
        returncode,
        stdout.decode("utf-8", errors="replace"),
        stderr.decode("utf-8", errors="replace"),
    )


def _run_version_command(args: Sequence[str]) -> tuple[int, str, str]:
    """Execute the ``--version`` command capturing decoded output."""
    try:
        return asyncio.run(_async_version_probe(args))
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(_async_version_probe(args))
        finally:
            loop.close()


def check_required_binaries(names: Iterable[str] | None = None) -> list[BinaryCheckResult]:
    """Check the default set of binaries required for shell tooling."""
    targets = list(names) if names is not None else ["rg", "fd"]
    return [check_binary(name) for name in targets]


def warn_if_missing_binaries(
    *,
    console: Console | None = None,
    names: Iterable[str] | None = None,
) -> list[BinaryCheckResult]:
    """Emit warnings for missing binaries and optionally log to ``console``."""
    results = check_required_binaries(names)
    for result in results:
        if result.available:
            continue
        message = (
            f"Optional binary '{result.name}' is unavailable; install it for improved search performance."
        )
        warnings.warn(message, ToolingWarning, stacklevel=2)
        if console is not None:
            console.print(f"[yellow]{result.message}[/yellow]")
    return results

