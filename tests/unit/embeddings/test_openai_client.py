"""Tests for the OpenAI embeddings client wrapper."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from code_agent_experiments.embeddings import EmbeddingError, OpenAIEmbeddingsClient


class FakeEmbeddingsEndpoint:
    """Lightweight stub mimicking the OpenAI embeddings endpoint."""

    def __init__(self) -> None:
        """Initialise call tracking state."""
        self.calls: list[dict[str, Any]] = []
        self.failures: int = 0

    def create(self, *, model: str, **kwargs: Any) -> Any:
        """Return deterministic embedding vectors, optionally raising first."""
        if self.failures:
            self.failures -= 1
            raise RuntimeError("retry me")
        inputs = list(kwargs["input"])
        vectors = [[float(index + i) for i in range(3)] for index, _ in enumerate(inputs)]
        data = [SimpleNamespace(embedding=vector) for vector in vectors]
        self.calls.append({"model": model, "input": inputs})
        return SimpleNamespace(data=data)


class FakeClient:
    """Container exposing the mocked embeddings endpoint."""

    def __init__(self) -> None:
        """Expose an embeddings attribute for the wrapper under test."""
        self.embeddings = FakeEmbeddingsEndpoint()


def test_openai_embeddings_client_batches_requests() -> None:
    """Multiple inputs should be dispatched across batches."""
    client = FakeClient()
    wrapper = OpenAIEmbeddingsClient("text-embedding-3", batch_size=2, client=client)
    inputs = [f"text-{i}" for i in range(5)]
    results = wrapper.embed(inputs)
    assert len(results) == len(inputs)
    assert len(client.embeddings.calls) == 3
    assert all(result.embedding for result in results)
    assert results[0].index == 0
    assert results[-1].index == len(inputs) - 1


def test_openai_embeddings_client_retries_and_raises() -> None:
    """Retry logic should back off and surface repeated failures."""
    client = FakeClient()
    client.embeddings.failures = 1
    slept: list[float] = []

    def fake_sleep(duration: float) -> None:
        slept.append(duration)

    wrapper = OpenAIEmbeddingsClient(
        "text-embedding-3",
        batch_size=2,
        max_retries=2,
        client=client,
        sleep=fake_sleep,
    )
    results = wrapper.embed(["a", "b"])
    assert results
    assert slept == [0.5]

    client.embeddings.failures = 3
    wrapper = OpenAIEmbeddingsClient("text-embedding-3", batch_size=2, max_retries=1, client=client, sleep=fake_sleep)
    with pytest.raises(EmbeddingError):
        wrapper.embed(["x"])
