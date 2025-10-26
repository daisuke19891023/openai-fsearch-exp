"""Tree-sitter assisted chunking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
else:
    from typing import Iterable as _Iterable, Sequence as _Sequence
    Iterable = _Iterable
    Sequence = _Sequence

from .base import Chunk, merge_metadata
from .fixed import FixedTokenChunker

__all__ = ["TreeSitterChunker", "TreeSitterProvider"]

DEFAULT_NODE_TYPES: dict[str, tuple[str, ...]] = {
    "python": ("function_definition", "class_definition"),
    "javascript": ("function_declaration", "class_declaration", "method_definition"),
}


@dataclass(slots=True)
class TreeSitterProvider:
    """Lightweight wrapper describing the API expected from a tree-sitter parser."""

    parser: Any
    language: str

    def parse(self, source: str) -> Any:
        """Parse ``source`` and return a tree-sitter tree."""
        return self.parser.parse(source.encode("utf-8"))


class TreeSitterChunker:
    """Chunker that aligns windows to syntax nodes discovered by tree-sitter."""

    def __init__(
        self,
        provider: TreeSitterProvider,
        *,
        size_tokens: int = 400,
        overlap_tokens: int = 80,
        include_node_types: Sequence[str] | None = None,
    ) -> None:
        """Configure the chunker with a parser provider and window settings."""
        self.provider = provider
        self._fixed = FixedTokenChunker(size_tokens, overlap_tokens=overlap_tokens)
        self._language = provider.language.lower()
        defaults = DEFAULT_NODE_TYPES.get(self._language, ("function_definition", "class_definition"))
        self._target_types: tuple[str, ...] = tuple(include_node_types) if include_node_types else defaults

    def chunk(self, text: str) -> list[Chunk]:
        """Return syntax-aware chunks for ``text``."""
        tree = self.provider.parse(text)
        root = tree.root_node
        nodes = sorted(self._select_nodes(root), key=lambda node: node.start_byte)
        if not nodes:
            return self._fixed.chunk(text, metadata={"strategy": "ast", "node_type": "document"})
        results: list[Chunk] = []
        for index, node in enumerate(nodes):
            start_byte = int(node.start_byte)
            end_byte = int(node.end_byte)
            snippet = text[start_byte:end_byte]
            if not snippet.strip():
                continue
            start_row = int(node.start_point[0]) + 1
            metadata = {
                "strategy": "ast",
                "node_type": node.type,
                "node_index": index,
            }
            segments = self._fixed.chunk(snippet, base_line=start_row, metadata=metadata)
            for chunk_index, chunk in enumerate(segments):
                extra = {"chunk_index": chunk_index, "strategy": "ast"}
                merged = merge_metadata(chunk.metadata, extra)
                results.append(
                    Chunk(
                        text=chunk.text,
                        start_line=chunk.start_line,
                        end_line=chunk.end_line,
                        metadata=merged,
                    ),
                )
        return results

    def _select_nodes(self, root: Any) -> Iterable[Any]:
        """Yield nodes of interest from ``root`` in depth-first order."""
        stack = [root]
        while stack:
            node = stack.pop()
            node_type = getattr(node, "type", "")
            if node_type in self._target_types:
                yield node
            raw_children = getattr(node, "children", None)
            children: Sequence[Any] = (
                cast("Sequence[Any]", ())
                if raw_children is None
                else cast("Sequence[Any]", raw_children)
            )
            stack.extend(reversed(children))
