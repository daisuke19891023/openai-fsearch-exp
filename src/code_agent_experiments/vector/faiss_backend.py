"""FAISS-based vector store with simple persistence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence as SequenceType
else:
    from typing import Sequence as _Sequence
    SequenceType = _Sequence


import numpy as np

Sequence = SequenceType

__all__ = ["FaissVectorStore", "VectorRecord", "VectorSearchResult"]


@dataclass(slots=True)
class VectorRecord:
    """Payload describing a vector to be stored."""

    id: str
    vector: Sequence[float]
    metadata: dict[str, Any]


@dataclass(slots=True)
class VectorSearchResult:
    """Result returned from a similarity search."""

    id: str
    score: float
    metadata: dict[str, Any]


class FaissVectorStore:
    """Minimal vector store supporting add/search/save/load operations."""

    def __init__(
        self,
        dimension: int,
        *,
        metric: str = "cosine",
        index_path: str | Path | None = None,
    ) -> None:
        """Initialise a FAISS-backed store with the desired metric."""
        if dimension <= 0:
            message = "dimension must be positive"
            raise ValueError(message)
        if metric not in {"cosine", "l2", "ip"}:
            message = "metric must be one of 'cosine', 'l2', or 'ip'"
            raise ValueError(message)
        self.dimension = dimension
        self.metric = metric
        self.index_path = Path(index_path) if index_path else None
        self._faiss = _load_faiss()
        self._index = self._create_index()
        self._ids: list[str] = []
        self._metadata: list[dict[str, Any]] = []

    def add(self, records: Sequence[VectorRecord]) -> None:
        """Insert ``records`` into the FAISS index."""
        if not records:
            return
        vectors = np.asarray([record.vector for record in records], dtype="float32")
        if vectors.shape[1] != self.dimension:
            message = f"Expected vectors of dimension {self.dimension}, received {vectors.shape[1]}"
            raise ValueError(message)
        if self.metric == "cosine":
            self._faiss.normalize_L2(vectors)
        self._index.add(vectors)
        for record in records:
            self._ids.append(record.id)
            self._metadata.append(dict(record.metadata))

    def search(self, query: Sequence[float], *, top_k: int = 5) -> list[VectorSearchResult]:
        """Return the ``top_k`` nearest neighbours for ``query``."""
        if top_k <= 0:
            message = "top_k must be positive"
            raise ValueError(message)
        if not self._ids:
            return []
        query_vector = np.asarray(query, dtype="float32").reshape(1, -1)
        if query_vector.shape[1] != self.dimension:
            message = f"Query vector must have dimension {self.dimension}"
            raise ValueError(message)
        if self.metric == "cosine":
            self._faiss.normalize_L2(query_vector)
        distances, indices = self._index.search(query_vector, top_k)
        results: list[VectorSearchResult] = []
        for distance, index in zip(distances[0], indices[0], strict=False):
            idx = int(index)
            if idx < 0 or idx >= len(self._ids):
                continue
            record_metadata = self._metadata[idx]
            metadata = dict(record_metadata)
            results.append(
                VectorSearchResult(
                    id=self._ids[idx],
                    score=float(distance),
                    metadata=metadata,
                ),
            )
        return results

    def save(self, path: str | Path | None = None) -> Path:
        """Persist the FAISS index and metadata to disk."""
        target = Path(path) if path is not None else self.index_path
        if target is None:
            message = "No index path configured for persistence"
            raise ValueError(message)
        target.parent.mkdir(parents=True, exist_ok=True)
        self._faiss.write_index(self._index, str(target))
        metadata_path = target.with_suffix(".meta.json")
        payload = {
            "ids": self._ids,
            "metadata": self._metadata,
            "metric": self.metric,
            "dimension": self.dimension,
        }
        metadata_path.write_text(json.dumps(payload), encoding="utf-8")
        self.index_path = target
        return target

    @classmethod
    def load(cls, path: str | Path) -> FaissVectorStore:
        """Load a persisted FAISS index from ``path``."""
        faiss = _load_faiss()
        path = Path(path)
        metadata_path = path.with_suffix(".meta.json")
        if not metadata_path.exists():
            message = f"Missing metadata file alongside {path}"
            raise FileNotFoundError(message)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        store = cls(dimension=int(metadata["dimension"]), metric=str(metadata["metric"]), index_path=path)
        store._index = faiss.read_index(str(path))
        store._ids = list(metadata["ids"])
        store._metadata = [dict(entry) for entry in metadata["metadata"]]
        return store

    def _create_index(self) -> Any:
        if self.metric == "l2":
            return self._faiss.IndexFlatL2(self.dimension)
        # cosine and ip both use inner product, but cosine vectors are normalised.
        return self._faiss.IndexFlatIP(self.dimension)


def _load_faiss() -> Any:
    try:
        import faiss  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - depends on environment
        message = "faiss-cpu package is required for FaissVectorStore"
        raise RuntimeError(message) from exc
    return faiss
