"""OpenAI embeddings client with batching and retry support."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence
else:
    from typing import Callable as _Callable, Iterable as _Iterable, Sequence as _Sequence
    Callable = _Callable
    Iterable = _Iterable
    Sequence = _Sequence

__all__ = ["EmbeddingError", "EmbeddingResult", "OpenAIEmbeddingsClient"]


@dataclass(slots=True)
class EmbeddingResult:
    """Response container for individual embedding vectors."""

    embedding: list[float]
    index: int
    text: str


class EmbeddingError(RuntimeError):
    """Raised when an embedding request cannot be fulfilled."""


class OpenAIEmbeddingsClient:
    """Thin wrapper around ``openai`` that handles batching and retries."""

    def __init__(
        self,
        model: str,
        *,
        batch_size: int = 128,
        max_retries: int = 3,
        backoff_seconds: float = 0.5,
        client: Any | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        """Initialise the client with retry configuration."""
        if batch_size <= 0:
            message = "batch_size must be positive"
            raise ValueError(message)
        if max_retries < 0:
            message = "max_retries must be non-negative"
            raise ValueError(message)
        self.model = model
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self._sleep = sleep or time.sleep
        self._client = client or self._default_client()

    def embed(self, texts: Sequence[str]) -> list[EmbeddingResult]:
        """Return embeddings for ``texts`` using batched API calls."""
        results: list[EmbeddingResult] = []
        for offset, batch in enumerate(_batched(texts, self.batch_size)):
            batch_results = self._embed_batch(batch, offset * self.batch_size)
            results.extend(batch_results)
        return results

    def _embed_batch(self, batch: Sequence[str], offset: int) -> list[EmbeddingResult]:
        """Embed a single batch with retry handling."""
        attempt = 0
        while True:
            try:
                response = self._client.embeddings.create(model=self.model, input=list(batch))
            except Exception as exc:  # pragma: no cover - exercised via retry tests
                if attempt >= self.max_retries:
                    message = "Embedding request failed"
                    raise EmbeddingError(message) from exc
                delay = self.backoff_seconds * (2**attempt)
                self._sleep(delay)
                attempt += 1
                continue
            data = getattr(response, "data", None)
            if data is None:
                message = "Embedding response missing 'data' field"
                raise EmbeddingError(message)
            parsed: list[EmbeddingResult] = []
            for index, item in enumerate(data):
                embedding = getattr(item, "embedding", None)
                if embedding is None:
                    message = "Embedding item missing vector"
                    raise EmbeddingError(message)
                parsed.append(
                    EmbeddingResult(embedding=list(map(float, embedding)), index=offset + index, text=batch[index]),
                )
            return parsed

    @staticmethod
    def _default_client() -> Any:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - exercised in environments without openai installed
            message = "openai package is required to create a default client"
            raise RuntimeError(message) from exc
        return OpenAI()


def _batched(items: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    """Yield ``items`` in batches of ``size``."""
    total = len(items)
    for start in range(0, total, size):
        yield items[start : start + size]
