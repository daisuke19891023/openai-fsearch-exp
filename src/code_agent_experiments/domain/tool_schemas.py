"""JSON schema models for LLM tool definitions."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TYPE_CHECKING

from pydantic import BaseModel, Field, field_validator

if TYPE_CHECKING:
    from collections.abc import Sequence
    from code_agent_experiments.tools.shell import RipgrepMatch

__all__ = [
    "RipgrepToolInput",
    "RipgrepToolMatch",
    "RipgrepToolOutput",
]


class RipgrepToolInput(BaseModel):
    """Input payload accepted by the ripgrep search tool."""

    pattern: str = Field(..., description="Regular expression used to search for matches.")
    root: str = Field(
        ...,
        description=(
            "Absolute or relative path to the repository root that should be searched. "
            "The path is resolved on the server running the tool."
        ),
    )
    globs: list[str] | None = Field(
        default=None,
        description="Optional list of glob patterns that candidate files must satisfy.",
    )
    ignore_case: bool = Field(
        default=False,
        description="When true, perform a case-insensitive search.",
    )
    max_count: int | None = Field(
        default=50,
        ge=1,
        le=200,
        description=(
            "Maximum number of matches to return."
            " Results are truncated when the limit is reached."
        ),
    )

    @field_validator("root")
    @classmethod
    def _validate_root(cls, value: str) -> str:
        path = Path(value)
        if not value.strip():
            msg = "Search root must not be empty"
            raise ValueError(msg)
        if value.strip().startswith("~"):
            msg = "Search root must be an absolute or relative path, not containing a home shortcut"
            raise ValueError(msg)
        if path.is_absolute():
            return str(path)
        return value


class RipgrepToolMatch(BaseModel):
    """Single search hit returned by the ripgrep tool."""

    path: str = Field(..., description="Absolute path to the file containing the match.")
    line_number: int = Field(..., ge=1, description="1-indexed line number where the match occurred.")
    line: str = Field(..., description="Source line text containing the match.")
    submatches: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Regex capture groups returned by ripgrep for the match.",
    )

    @classmethod
    def from_runtime(cls, match: Any) -> RipgrepToolMatch:
        """Construct a schema instance from a ``RipgrepMatch`` runtime object."""
        return cls(
            path=str(Path(match.path).resolve()),
            line_number=int(match.line_number),
            line=str(match.line),
            submatches=tuple(str(value) for value in getattr(match, "submatches", ()) or ()),
        )


def _empty_matches() -> list[RipgrepToolMatch]:
    """Return an empty collection of ripgrep matches."""
    return []


class RipgrepToolOutput(BaseModel):
    """Structured payload returned by the ripgrep tool."""

    matches: list[RipgrepToolMatch] = Field(
        default_factory=_empty_matches,
        description="Collection of matches found by ripgrep.",
    )
    truncated: bool = Field(
        default=False,
        description="Indicates whether more matches were available but omitted due to limits.",
    )

    @classmethod
    def from_runtime(
        cls,
        matches: Sequence[RipgrepMatch | RipgrepToolMatch],
        *,
        limit: int | None,
    ) -> RipgrepToolOutput:
        """Create the output payload from Python ``RipgrepMatch`` objects."""
        converted: list[RipgrepToolMatch] = [
            RipgrepToolMatch.from_runtime(match) for match in matches
        ]
        truncated = bool(limit is not None and len(matches) >= limit)
        return cls(matches=converted, truncated=truncated)

