"""Tests for the tree-sitter chunker."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from code_agent_experiments.chunking import TreeSitterChunker, TreeSitterProvider


@dataclass
class FakeNode:
    """Minimal tree node stub used for chunker tests."""

    type: str
    start_byte: int
    end_byte: int
    start_point: tuple[int, int]
    end_point: tuple[int, int]
    children: tuple[Any, ...] = ()


class FakeTree:
    """Container exposing a root node attribute."""

    def __init__(self, root: FakeNode) -> None:
        """Store the provided root node."""
        self.root_node = root


class FakeParser:
    """Parser stub returning a preconstructed tree."""

    def __init__(self, tree: FakeTree) -> None:
        """Record the tree that should be returned on parse."""
        self._tree = tree
        self.calls: list[bytes] = []

    def parse(self, source: bytes) -> FakeTree:
        """Capture the incoming source bytes and return the stub tree."""
        self.calls.append(source)
        return self._tree


def build_sample_tree(text: str) -> FakeTree:
    """Construct a tree with two function nodes for the provided ``text``."""
    beta_start = text.index("def beta")
    func_a = FakeNode(
        type="function_definition",
        start_byte=text.index("def alpha"),
        end_byte=beta_start,
        start_point=(0, 0),
        end_point=(3, 0),
    )
    func_b = FakeNode(
        type="function_definition",
        start_byte=beta_start,
        end_byte=len(text),
        start_point=(4, 0),
        end_point=(7, 0),
    )
    root = FakeNode(
        type="module",
        start_byte=0,
        end_byte=len(text),
        start_point=(0, 0),
        end_point=(7, 0),
        children=(func_a, func_b),
    )
    return FakeTree(root)


def test_tree_sitter_chunker_aligns_to_nodes() -> None:
    """Chunker should align output to AST nodes when available."""
    source = """def alpha():\n    return 1\n\n\ndef beta():\n    return 2\n\n"""
    tree = build_sample_tree(source)
    parser = FakeParser(tree)
    provider = TreeSitterProvider(parser=parser, language="python")
    chunker = TreeSitterChunker(provider, size_tokens=10, overlap_tokens=2)
    chunks = chunker.chunk(source)
    assert chunks
    assert all(chunk.metadata["strategy"] == "ast" for chunk in chunks)
    assert {chunk.metadata["node_type"] for chunk in chunks} == {"function_definition"}
    assert parser.calls
    assert parser.calls[0] == source.encode("utf-8")


def test_tree_sitter_chunker_fallback_to_document_chunking() -> None:
    """If no nodes match, the chunker should fall back to document windows."""
    source = """print('no functions')\nprint('just text')\n"""
    empty_root = FakeNode(
        type="module",
        start_byte=0,
        end_byte=len(source),
        start_point=(0, 0),
        end_point=(1, 0),
        children=(),
    )
    parser = FakeParser(FakeTree(empty_root))
    provider = TreeSitterProvider(parser=parser, language="python")
    chunker = TreeSitterChunker(provider, size_tokens=5, overlap_tokens=1)
    chunks = chunker.chunk(source)
    assert chunks
    assert chunks[0].metadata["node_type"] == "document"
