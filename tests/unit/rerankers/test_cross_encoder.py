"""Unit tests for the cross-encoder reranker."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest

from code_agent_experiments.domain.models import Candidate
from code_agent_experiments.rerankers import CrossEncoderReranker, RerankerError


def _candidate(path: str, *, content: str, rank: int | None = None) -> Candidate:
    return Candidate(
        path=path,
        score_keyword=0.0,
        score_dense=0.0,
        score_rerank=0.0,
        rank=rank,
        features={"content": content},
    )


@dataclass
class _DummyResponse:
    payload: dict[str, Any]

    @property
    def output_text(self) -> str:
        return json.dumps(self.payload)


class _DummyResponsesClient:
    def __init__(self, scores: dict[str, float]) -> None:
        self._scores = scores
        self.calls: list[dict[str, Any]] = []

    def create(
        self,
        *,
        model: str,
        input_items: Any,
        text: Any | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> _DummyResponse:
        self.calls.append({
            "model": model,
            "input_items": input_items,
            "text": text,
            "metadata": metadata,
        })
        return _DummyResponse(
            payload={
                "scores": [
                    {"id": identifier, "score": score}
                    for identifier, score in self._scores.items()
                ],
            },
        )


class _DummyClient:
    def __init__(self, responses_client: _DummyResponsesClient) -> None:
        self._responses = responses_client

    @property
    def responses(self) -> _DummyResponsesClient:
        return self._responses


def test_cross_encoder_rerank_orders_candidates_by_score() -> None:
    """Candidates should be reordered to follow descending rerank scores."""
    responses_client = _DummyResponsesClient({"b.py": 0.9, "a.py": 0.2})
    reranker = CrossEncoderReranker(
        client=_DummyClient(responses_client),
        model="gpt-4.1-mini",
        top_k=2,
    )
    candidates = [
        _candidate("a.py", content="alpha"),
        _candidate("b.py", content="bravo"),
        _candidate("c.py", content="charlie"),
    ]

    reranked = reranker.rerank("find bravo", candidates)

    assert [candidate.path for candidate in reranked[:2]] == ["b.py", "a.py"]
    assert reranked[0].score_rerank >= reranked[1].score_rerank >= reranked[2].score_rerank
    payload = json.loads(responses_client.calls[0]["input_items"][1]["content"][0]["text"])
    assert len(payload["documents"]) == 2
    assert reranked[2].path == "c.py"


def test_cross_encoder_rerank_requires_text_feature() -> None:
    """An informative error is raised when candidate text is unavailable."""
    responses_client = _DummyResponsesClient({})
    reranker = CrossEncoderReranker(
        client=_DummyClient(responses_client),
        model="gpt-4.1-mini",
        top_k=1,
    )
    candidates = [
        Candidate(
            path="missing.py",
            score_keyword=0.0,
            score_dense=0.0,
            score_rerank=0.0,
            rank=None,
            features={},
        ),
    ]

    with pytest.raises(RerankerError):
        reranker.rerank("query", candidates)

