"""Tests for the fixed token chunker."""

from __future__ import annotations

import pytest

from code_agent_experiments.chunking import FixedTokenChunker


@pytest.fixture
def sample_text() -> str:
    """Return a short Python module used across tests."""
    return """def alpha():\n    return 1\n\n# comment line\ndef beta():\n    return 2\n"""


def test_fixed_chunker_produces_overlapping_windows(sample_text: str) -> None:
    """Chunker should emit overlapping spans when configured with overlap."""
    chunker = FixedTokenChunker(size_tokens=5, overlap_tokens=2)
    chunks = chunker.chunk(sample_text)
    assert len(chunks) >= 2
    first, second = chunks[0], chunks[1]
    assert first.metadata["token_end"] - first.metadata["token_start"] <= 5
    assert first.metadata["token_end"] > second.metadata["token_start"]
    assert first.end_line >= first.start_line
    assert first.metadata["strategy"] == "fixed"


def test_fixed_chunker_respects_base_line(sample_text: str) -> None:
    """Chunks should offset line numbers by the provided base line."""
    chunker = FixedTokenChunker(size_tokens=4, overlap_tokens=1)
    base_line = 10
    chunks = chunker.chunk(sample_text, base_line=base_line, metadata={"source": "test"})
    assert all(chunk.start_line >= base_line for chunk in chunks)
    assert all(chunk.metadata["source"] == "test" for chunk in chunks)


def test_fixed_chunker_validates_arguments() -> None:
    """Invalid configuration should raise informative errors."""
    with pytest.raises(ValueError, match="positive"):
        FixedTokenChunker(size_tokens=0)
    with pytest.raises(ValueError, match="smaller"):
        FixedTokenChunker(size_tokens=4, overlap_tokens=5)
