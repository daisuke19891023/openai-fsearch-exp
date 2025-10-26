"""Tests for the `cae index` command."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TYPE_CHECKING

from typer.testing import CliRunner

from code_agent_experiments import cli
from code_agent_experiments.cli import app
from code_agent_experiments.embeddings.openai_client import EmbeddingResult


if TYPE_CHECKING:
    import pytest


runner = CliRunner()


class StubEmbeddingClient:
    """Stub embedding client capturing embed requests."""

    def __init__(self, model: str) -> None:
        """Store the provided model identifier."""
        self.model = model
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[EmbeddingResult]:
        """Return deterministic embeddings for ``texts``."""
        self.calls.append(list(texts))
        return [
            EmbeddingResult(embedding=[float(index), float(index) + 1.0], index=index, text=text)
            for index, text in enumerate(texts)
        ]


class StubVectorStore:
    """Stub vector store capturing added records."""

    def __init__(self, dimension: int, metric: str, index_path: Path) -> None:
        """Initialise the store with metadata about the target index."""
        self.dimension = dimension
        self.metric = metric
        self.index_path = Path(index_path)
        self.records: list[Any] = []
        self.saved_path: Path | None = None

    def add(self, records: list[Any]) -> None:
        """Record added vector payloads."""
        self.records.extend(records)

    def save(self, path: Path | None = None) -> Path:
        """Persist the stub index and capture the destination path."""
        target = Path(path) if path is not None else self.index_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("stub-index", encoding="utf-8")
        self.saved_path = target
        return target


def test_index_uses_stubbed_clients(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The index command delegates to the configured embedding client and vector store."""
    repo = tmp_path / "repo"
    repo.mkdir()
    file_path = repo / "sample.py"
    file_path.write_text("def answer():\n    return 42\n", encoding="utf-8")

    created: dict[str, Any] = {}

    def embedding_factory(model: str) -> StubEmbeddingClient:
        client = StubEmbeddingClient(model)
        created["client"] = client
        return client

    def vector_factory(dimension: int, metric: str, index_path: Path) -> StubVectorStore:
        store = StubVectorStore(dimension, metric, index_path)
        created["store"] = store
        return store

    monkeypatch.setattr(cli, "EMBEDDING_CLIENT_FACTORY", embedding_factory, raising=False)
    monkeypatch.setitem(cli.VECTOR_STORE_FACTORIES, "faiss", vector_factory)

    output_path = tmp_path / "index.faiss"
    result = runner.invoke(
        app,
        ["index", "--repo", str(repo), "--output", str(output_path)],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    store = created["store"]
    assert len(store.records) >= 1
    manifest_path = Path(f"{output_path}.manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["files_indexed"] == 1
    assert manifest["chunk_count"] == len(store.records)
    assert any(chunk["path"].endswith("sample.py") for chunk in manifest["chunks"])
