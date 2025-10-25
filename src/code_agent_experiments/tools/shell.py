"""Lightweight wrappers around command-line search tools."""
from __future__ import annotations
from dataclasses import dataclass
from fnmatch import fnmatch
import os
import re
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence
    from code_agent_experiments.domain.models import ToolName

__all__ = [
    "BinaryNotFoundError",
    "RipgrepMatch",
    "ToolExecutionError",
    "ensure_binary",
    "run_fd",
    "run_find",
    "run_grep",
    "run_ripgrep",
]
class BinaryNotFoundError(FileNotFoundError):
    """Raised when a requested binary is not available on the system."""

    def __init__(self, binary: str) -> None:
        """Store the missing binary name for downstream diagnostics."""
        message = f"Required binary '{binary}' was not found on PATH"
        super().__init__(message)
        self.binary = binary
class ToolExecutionError(RuntimeError):
    """Raised when a tool exits with a non-zero status code."""

    def __init__(self, tool: ToolName, returncode: int, stderr: str) -> None:
        """Capture tool metadata and error payload."""
        message = f"{tool} exited with status {returncode}: {stderr.strip()}"
        super().__init__(message)
        self.tool = tool
        self.returncode = returncode
        self.stderr = stderr
@dataclass(slots=True)
class RipgrepMatch:
    """Single line match reported by ripgrep-compatible tools."""

    path: Path
    line_number: int
    line: str
    submatches: tuple[str, ...]
    def __iter__(self) -> Iterator[str]:  # pragma: no cover - convenience only
        """Yield a tuple of human-readable fields for logging."""
        yield from (str(self.path), str(self.line_number), self.line, "|".join(self.submatches))
def ensure_binary(*names: str) -> str:
    """Return the first binary available on ``PATH`` among ``names``."""
    for name in names:
        located = shutil.which(name)
        if located:
            return located
    raise BinaryNotFoundError(names[0])
def run_ripgrep(
    pattern: str,
    root: Path,
    *,
    globs: Sequence[str] | None = None,
    ignore_case: bool = False,
    max_count: int | None = None,
    _context: int | None = None,
) -> list[RipgrepMatch]:
    """Search files using a ripgrep-compatible regex matcher."""
    try:
        flags = re.MULTILINE | (re.IGNORECASE if ignore_case else 0)
        compiled = re.compile(pattern, flags)
    except re.error as exc:  # pragma: no cover - defensive guard
        message = f"Invalid ripgrep pattern '{pattern}': {exc}"
        raise ValueError(message) from exc
    search_root, _ = _resolve_root(root)
    candidates = (
        path
        for path in _iter_target_paths(root, hidden=True, include_dirs=False, follow_symlinks=False)
        if path.is_file()
    )
    matches: list[RipgrepMatch] = []
    for file_path in candidates:
        if globs and not _matches_glob(file_path, globs, search_root):
            continue
        try:
            content = file_path.read_text(encoding="utf-8")
        except OSError:
            continue
        for line_number, line in enumerate(content.splitlines(), start=1):
            found = list(compiled.finditer(line))
            if not found:
                continue
            matches.append(
                RipgrepMatch(
                    path=file_path.resolve(),
                    line_number=line_number,
                    line=line,
                    submatches=tuple(match.group(0) for match in found),
                ),
            )
            if max_count is not None and len(matches) >= max_count:
                return matches
    return matches
def run_fd(
    pattern: str,
    root: Path,
    *,
    include_directories: bool = False,
    hidden: bool = False,
    follow_symlinks: bool = False,
    exclude: Sequence[str] | None = None,
    limit: int | None = None,
) -> list[Path]:
    """Return filesystem paths matching ``fd``-style heuristics."""
    search_root, _ = _resolve_root(root)
    gitignore_patterns = _load_gitignore(search_root)
    pattern_lower = pattern.lower()
    results: list[Path] = []
    for candidate in _iter_target_paths(
        root,
        hidden=hidden,
        include_dirs=True,
        follow_symlinks=follow_symlinks,
    ):
        if candidate == search_root:
            continue
        is_dir = candidate.is_dir()
        if is_dir and not include_directories:
            continue
        relative = candidate.relative_to(search_root)
        if _is_ignored(relative, gitignore_patterns):
            continue
        if exclude and any(fnmatch(str(relative), rule) for rule in exclude):
            continue
        if pattern_lower in relative.name.lower():
            results.append(candidate)
            if limit is not None and len(results) >= limit:
                break
    if not include_directories:
        results = [path for path in results if path.is_file()]
    return results
def run_grep(
    pattern: str,
    root: Path,
    *,
    ignore_case: bool = False,
) -> list[RipgrepMatch]:
    """Perform a simple substring search similar to ``grep -nR``."""
    needle = pattern.lower() if ignore_case else pattern
    matches: list[RipgrepMatch] = []
    for file_path in _iter_target_paths(root, hidden=False, include_dirs=False, follow_symlinks=False):
        if not file_path.is_file():
            continue
        try:
            content = file_path.read_text(encoding="utf-8")
        except OSError:
            continue
        for line_number, line in enumerate(content.splitlines(), start=1):
            haystack = line.lower() if ignore_case else line
            if needle in haystack:
                matches.append(
                    RipgrepMatch(
                        path=file_path.resolve(),
                        line_number=line_number,
                        line=line,
                        submatches=(pattern,),
                    ),
                )
    return matches
def run_find(
    root: Path,
    *,
    name: str | None = None,
    include_directories: bool = False,
    max_depth: int | None = None,
) -> list[Path]:
    """Enumerate filesystem paths matching ``find`` semantics."""
    search_root, _ = _resolve_root(root)
    results: list[Path] = []
    for candidate in _iter_target_paths(root, hidden=False, include_dirs=True, follow_symlinks=False):
        if candidate == search_root:
            continue
        relative = candidate.relative_to(search_root)
        depth = len(relative.parts)
        if max_depth is not None and depth > max_depth:
            continue
        if candidate.is_dir() and not include_directories:
            continue
        if name and not fnmatch(relative.name, name):
            continue
        if not include_directories and candidate.is_dir():
            continue
        results.append(candidate)
    if not include_directories:
        results = [path for path in results if path.is_file()]
    return results
def _resolve_root(root: Path) -> tuple[Path, str]:
    root = root.resolve()
    if root.is_dir():
        return root, "."
    return root.parent, root.name
def _iter_target_paths(
    root: Path,
    *,
    hidden: bool,
    include_dirs: bool,
    follow_symlinks: bool,
) -> Iterable[Path]:
    base, target = _resolve_root(root)
    if target != ".":
        yield base / target
        return
    yield from _iter_paths(base, hidden=hidden, include_dirs=include_dirs, follow_symlinks=follow_symlinks)
def _load_gitignore(root: Path) -> list[str]:
    path = root / ".gitignore"
    if not path.exists():
        return []
    patterns: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        patterns.append(stripped)
    return patterns
def _is_ignored(relative: Path, patterns: Sequence[str]) -> bool:
    text = str(relative)
    for rule in patterns:
        candidate = rule.removeprefix("/")
        if fnmatch(text, candidate) or fnmatch(relative.name, candidate):
            return True
    return False
def _matches_glob(path: Path, globs: Sequence[str], root: Path) -> bool:
    if not globs:
        return True
    relative_text = str(path.relative_to(root))
    return any(fnmatch(relative_text, glob) or fnmatch(path.name, glob) for glob in globs)
def _iter_paths(
    root: Path,
    *,
    hidden: bool,
    include_dirs: bool,
    follow_symlinks: bool,
) -> Iterable[Path]:
    for current_root, dirs, files in os.walk(root, followlinks=follow_symlinks):
        current_path = Path(current_root)
        rel = current_path.relative_to(root)
        if not hidden and rel != Path() and any(part.startswith(".") for part in rel.parts):
            dirs[:] = []
            continue
        if include_dirs and current_path != root:
            yield current_path
        for name in list(dirs):
            if not hidden and name.startswith("."):
                dirs.remove(name)
        for file_name in files:
            if not hidden and file_name.startswith("."):
                continue
            yield current_path / file_name
