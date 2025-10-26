"""Tests for the FAISS vector store abstraction."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from code_agent_experiments.vector import FaissVectorStore, VectorRecord


class StubIndex:
    """In-memory approximation of a FAISS flat index."""

    def __init__(self, dimension: int, metric: str) -> None:
        """Initialise the stub with an empty vector store."""
        self.d = dimension
        self.metric = metric
        self.vectors = np.empty((0, dimension), dtype="float32")

    def add(self, array: np.ndarray) -> None:
        """Append vectors to the in-memory index."""
        self.vectors = np.vstack([self.vectors, array])

    def search(self, query: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return indices and distances for the provided query."""
        if self.vectors.size == 0:
            distances = np.zeros((1, top_k), dtype="float32")
            indices = np.full((1, top_k), -1, dtype="int64")
            return distances, indices
        query_vec = query[0]
        if self.metric == "l2":
            diff = self.vectors - query_vec
            scores = np.sum(diff * diff, axis=1)
            order = np.argsort(scores)
            distances = scores[order][:top_k]
        else:
            scores = self.vectors @ query_vec
            order = np.argsort(scores)[::-1]
            distances = scores[order][:top_k]
        indices = order[:top_k]
        padded_distances = np.zeros(top_k, dtype="float32")
        padded_distances[: len(distances)] = distances.astype("float32")
        padded_indices = np.full(top_k, -1, dtype="int64")
        padded_indices[: len(indices)] = indices.astype("int64")
        return padded_distances.reshape(1, -1), padded_indices.reshape(1, -1)


def _index_l2(dimension: int) -> StubIndex:
    return StubIndex(dimension, "l2")


def _index_ip(dimension: int) -> StubIndex:
    return StubIndex(dimension, "ip")


def _normalize(array: np.ndarray) -> None:
    for i, row in enumerate(array):
        norm = np.linalg.norm(row)
        if norm:
            array[i] = row / norm


def _write_index(index: StubIndex, path: str) -> None:
    payload = {
        "dimension": index.d,
        "metric": index.metric,
        "vectors": index.vectors.tolist(),
    }
    Path(path).write_text(json.dumps(payload), encoding="utf-8")


def _read_index(path: str) -> StubIndex:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    stub = StubIndex(payload["dimension"], payload["metric"])
    stub.vectors = np.asarray(payload["vectors"], dtype="float32")
    return stub


@pytest.fixture(autouse=True)
def stub_faiss(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install a stub ``faiss`` module for the duration of each test."""
    module = SimpleNamespace(
        IndexFlatL2=_index_l2,
        IndexFlatIP=_index_ip,
        normalize_L2=_normalize,
        write_index=_write_index,
        read_index=_read_index,
    )
    monkeypatch.setitem(sys.modules, "faiss", module)


def test_faiss_store_add_and_search(tmp_path: Path) -> None:
    """Vectors should be retrievable and persisted across saves."""
    store = FaissVectorStore(dimension=3, metric="cosine")
    vectors = [
        VectorRecord(id="a", vector=[1.0, 0.0, 0.0], metadata={"file": "a.py"}),
        VectorRecord(id="b", vector=[0.0, 1.0, 0.0], metadata={"file": "b.py"}),
    ]
    store.add(vectors)
    query = [0.9, 0.1, 0.0]
    results = store.search(query, top_k=2)
    assert results
    assert results[0].id == "a"
    assert results[0].metadata["file"] == "a.py"

    index_path = tmp_path / "index.faiss"
    store.save(index_path)
    reloaded = FaissVectorStore.load(index_path)
    again = reloaded.search(query, top_k=2)
    assert again
    assert again[0].id == "a"
