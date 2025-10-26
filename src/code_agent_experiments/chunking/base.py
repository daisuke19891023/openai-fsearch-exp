"""Shared chunking primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["Chunk", "merge_metadata"]


@dataclass(slots=True)
class Chunk:
    """Represents a contiguous region of text produced by a chunker."""

    text: str
    start_line: int
    end_line: int
    metadata: dict[str, Any]


def merge_metadata(base: dict[str, Any], extra: dict[str, Any] | None) -> dict[str, Any]:
    """Return a defensive merge of ``base`` metadata with ``extra``."""
    if not extra:
        return dict(base)
    merged = dict(base)
    merged.update(extra)
    return merged
