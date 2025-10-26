"""Cross-encoder based reranking utilities using the OpenAI SDK."""

from __future__ import annotations

import json
import typing as t
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from openai.types.responses.response_input_param import ResponseInputParam
    from openai.types.responses.response_text_config_param import ResponseTextConfigParam
    from code_agent_experiments.domain.models import Candidate
else:  # pragma: no cover - runtime fallbacks for typing imports
    ResponseInputParam = Any  # type: ignore[assignment]
    ResponseTextConfigParam = dict[str, Any]  # type: ignore[assignment]

ResponseInput = str | ResponseInputParam


class ResponseLike(Protocol):
    """Subset of the OpenAI response object used by the reranker."""

    @property
    def output_text(self) -> str | None:  # pragma: no cover - protocol attribute
        """Return the concatenated textual output returned by the model."""
        ...


class ResponsesClient(Protocol):
    """Protocol for the OpenAI Responses API used by the reranker."""

    def create(
        self,
        *,
        model: str,
        input_items: ResponseInput,
        text: ResponseTextConfigParam | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ResponseLike:  # pragma: no cover - interface definition
        """Invoke the Responses API and return a response payload."""
        ...


class ResponsesClientProvider(Protocol):
    """Minimal client exposing a ``responses`` attribute."""

    @property
    def responses(self) -> ResponsesClient:  # pragma: no cover - interface definition
        """Return the Responses API client."""
        ...


class RerankerError(RuntimeError):
    """Raised when reranking cannot be completed."""


@dataclass(slots=True)
class CrossEncoderReranker:
    """Apply a cross-encoder model to re-score retrieval candidates."""

    client: ResponsesClientProvider
    model: str
    top_k: int = 100
    text_feature_key: str = "content"
    system_prompt: str = (
        "You are a cross-encoder reranker. Score each candidate document for its relevance to the query on a"
        " scale between 0 and 1 (higher is better). Return JSON using the provided schema."
    )
    _text_config: ResponseTextConfigParam = field(init=False)

    def __post_init__(self) -> None:
        """Validate configuration and prepare structured output schema."""
        if self.top_k <= 0:
            message = "top_k must be a positive integer"
            raise ValueError(message)
        self._text_config: ResponseTextConfigParam = {
            "format": {
                "type": "json_schema",
                "name": "reranker_scores",
                "strict": True,
                "description": "Relevance scores for candidate documents.",
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "scores": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": ["id", "score"],
                                "properties": {
                                    "id": {"type": "string"},
                                    "score": {
                                        "type": "number",
                                        "minimum": 0.0,
                                        "maximum": 1.0,
                                    },
                                },
                            },
                        },
                    },
                    "required": ["scores"],
                },
            },
        }

    def rerank(self, query: str, candidates: t.Sequence[Candidate]) -> list[Candidate]:
        """Return candidates sorted by cross-encoder scores."""
        items = list(candidates)
        if not items:
            return []
        limited = items[: min(self.top_k, len(items))]
        documents = [
            {"id": candidate.path, "text": self._extract_text(candidate)}
            for candidate in limited
        ]
        payload = {
            "query": query,
            "documents": documents,
        }
        response = self.client.responses.create(
            model=self.model,
            input_items=[
                {
                    "role": "system",
                    "type": "message",
                    "content": [
                        {
                            "type": "input_text",
                            "text": self.system_prompt,
                        },
                    ],
                },
                {
                    "role": "user",
                    "type": "message",
                    "content": [
                        {
                            "type": "input_text",
                            "text": json.dumps(payload, ensure_ascii=False),
                        },
                    ],
                },
            ],
            text=self._text_config,
            metadata={"component": "cross_encoder_reranker", "top_k": len(documents)},
        )
        score_map = self._parse_scores(response)
        reranked_top: list[Candidate] = []
        for candidate in limited:
            score = float(score_map.get(candidate.path, 0.0))
            reranked_top.append(
                candidate.model_copy(update={"score_rerank": score}),
            )
        reranked_top.sort(key=lambda candidate: candidate.score_rerank, reverse=True)
        result: list[Candidate] = []
        for index, candidate in enumerate(reranked_top, start=1):
            result.append(candidate.model_copy(update={"rank": index}))
        next_rank = len(result) + 1
        for candidate in items[len(limited) :]:
            rank_value = candidate.rank if candidate.rank is not None else next_rank
            result.append(candidate.model_copy(update={"rank": rank_value}))
            next_rank += 1
        return result

    def _parse_scores(self, response: ResponseLike) -> dict[str, float]:
        output_text = response.output_text or ""
        try:
            raw = json.loads(output_text)
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive guard
            message = "Failed to decode reranker response as JSON"
            raise RerankerError(message) from exc
        if not isinstance(raw, dict):
            message = "Reranker response must be a JSON object"
            raise RerankerError(message)
        data = t.cast("dict[str, Any]", raw)
        scores = data.get("scores")
        if not isinstance(scores, list):
            message = "Reranker response missing 'scores' array"
            raise RerankerError(message)
        parsed: dict[str, float] = {}
        iterable_scores = t.cast("t.Iterable[object]", scores)
        for entry in iterable_scores:
            if not isinstance(entry, dict):
                continue
            entry_dict = t.cast("dict[str, Any]", entry)
            identifier = entry_dict.get("id")
            score = entry_dict.get("score")
            if isinstance(identifier, str) and isinstance(score, (int, float)):
                parsed[identifier] = float(score)
        if not parsed:
            message = "Reranker response did not include any valid scores"
            raise RerankerError(message)
        return parsed

    def _extract_text(self, candidate: Candidate) -> str:
        text = candidate.features.get(self.text_feature_key)
        if not isinstance(text, str):
            message = (
                f"Candidate '{candidate.path}' is missing text feature '{self.text_feature_key}'"
            )
            raise RerankerError(message)
        return text
