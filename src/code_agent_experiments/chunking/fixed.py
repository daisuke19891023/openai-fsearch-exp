"""Fixed-size token chunker implementation."""

from __future__ import annotations

import re
from typing import Any

from .base import Chunk, merge_metadata

__all__ = ["FixedTokenChunker"]

DEFAULT_TOKEN_PATTERN = re.compile(r"\S+")


class SimpleTokenizer:
    """A minimal tokenizer based on contiguous non-whitespace runs."""

    def __call__(self, text: str) -> list[tuple[int, int]]:
        return [(match.start(), match.end()) for match in DEFAULT_TOKEN_PATTERN.finditer(text)]


class FixedTokenChunker:
    """Split documents into overlapping token windows."""

    def __init__(
        self,
        size_tokens: int = 400,
        *,
        overlap_tokens: int = 80,
        tokenizer: SimpleTokenizer | None = None,
    ) -> None:
        """Initialise the chunker with window size and overlap parameters."""
        if size_tokens <= 0:
            message = "size_tokens must be positive"
            raise ValueError(message)
        if overlap_tokens < 0:
            message = "overlap_tokens must be non-negative"
            raise ValueError(message)
        if overlap_tokens >= size_tokens:
            message = "overlap_tokens must be smaller than size_tokens"
            raise ValueError(message)
        self.size_tokens = size_tokens
        self.overlap_tokens = overlap_tokens
        self._tokenizer = tokenizer or SimpleTokenizer()

    def chunk(self, text: str, *, base_line: int = 1, metadata: dict[str, Any] | None = None) -> list[Chunk]:
        """Chunk ``text`` into overlapping windows."""
        spans = self._tokenizer(text)
        if not spans:
            return []
        step = self.size_tokens - self.overlap_tokens
        chunks: list[Chunk] = []
        index = 0
        total = len(spans)
        while index < total:
            end_index = min(index + self.size_tokens, total)
            span_start = spans[index][0]
            span_end = spans[end_index - 1][1]
            snippet = text[span_start:span_end]
            rel_start_line = _line_number(text, span_start)
            rel_end_line = _line_number(text, span_end - 1)
            chunk_metadata = merge_metadata(
                {
                    "strategy": "fixed",
                    "token_start": index,
                    "token_end": end_index,
                    "token_total": total,
                },
                metadata,
            )
            chunks.append(
                Chunk(
                    text=snippet,
                    start_line=base_line + rel_start_line - 1,
                    end_line=base_line + rel_end_line - 1,
                    metadata=chunk_metadata,
                ),
            )
            if end_index == total:
                break
            index += step
        return chunks

    def tokenize(self, text: str) -> list[tuple[int, int]]:
        """Expose token spans for downstream consumers (primarily tests)."""
        return self._tokenizer(text)


def _line_number(text: str, char_index: int) -> int:
    """Return the 1-indexed line number at ``char_index``."""
    if not text:
        return 1
    char_index = min(max(char_index, 0), len(text) - 1)
    return text.count("\n", 0, char_index) + 1
