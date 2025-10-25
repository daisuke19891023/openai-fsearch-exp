"""Tests for domain models round-trip serialization."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pydantic import BaseModel

from code_agent_experiments.domain.models import (
    Candidate,
    ChunkingConfig,
    Metrics,
    ModelConfig,
    ReportSummary,
    RetrievalRecord,
    RerankerConfig,
    RunConfig,
    Scenario,
    ToolCall,
    ToolLimits,
    VectorBackendConfig,
)


def _sample_run_config() -> RunConfig:
    return RunConfig(
        id="run-001",
        model=ModelConfig(name="gpt-4.1-mini", temperature=0.1, max_output_tokens=1024),
        tools_enabled=["ripgrep", "vector.faiss", "fd"],
        chunking=ChunkingConfig(type="fixed", size_tokens=500, overlap_tokens=50, languages=["python", "ts"]),
        vector=VectorBackendConfig(backend="faiss", index_path=".index/faiss", metric="cosine"),
        reranker=RerankerConfig(name="bge-reranker-v2-m3", top_k=75),
        tool_limits=ToolLimits(max_calls=12, per_call_timeout_sec=25),
        seed=13,
        replicate=2,
    )


def _sample_scenario() -> Scenario:
    return Scenario(
        id="scenario-001",
        repo_path="./repos/sample",
        query="Where is the ValueError raised?",
        language="python",
        ground_truth_files=["src/module.py", "tests/test_module.py"],
        metadata={"issue": "SWE-1"},
    )


def _sample_tool_call() -> ToolCall:
    return ToolCall(
        name="ripgrep",
        args={"pattern": "ValueError", "root": "./repos/sample"},
        started_at=datetime(2024, 7, 1, 12, 0, 0, tzinfo=UTC),
        ended_at=datetime(2024, 7, 1, 12, 0, 1, tzinfo=UTC),
        success=True,
        stderr_preview="",
        stdout_preview="path/to/file.py:42: raise ValueError",
        token_in=120,
        token_out=35,
        latency_ms=980,
    )


def _sample_candidate() -> Candidate:
    return Candidate(
        path="src/module.py",
        score_keyword=0.85,
        score_dense=0.72,
        score_rerank=0.9,
        rank=1,
        features={"hits": 3},
    )


def _sample_metrics(run_id: str, scenario_id: str) -> Metrics:
    return Metrics(
        scenario_id=scenario_id,
        run_id=run_id,
        strategy="rg+faiss",
        recall_at_k={5: 1.0, 10: 1.0, 20: 1.0},
        ndcg_at_k={5: 0.95, 10: 0.97, 20: 0.99},
        mrr=1.0,
        tool_calls=4,
        wall_ms=1345,
        cost_usd=0.12,
    )


@pytest.mark.parametrize(
    "instance",
    [
        _sample_run_config(),
        _sample_scenario(),
        _sample_tool_call(),
        _sample_candidate(),
        RetrievalRecord(
            scenario_id="scenario-001",
            run_id="run-001",
            strategy="rg+faiss",
            candidates=[_sample_candidate()],
            tool_calls=[_sample_tool_call()],
            elapsed_ms=2050,
        ),
        _sample_metrics(run_id="run-001", scenario_id="scenario-001"),
        ReportSummary(
            run_id="run-001",
            per_scenario=[_sample_metrics(run_id="run-001", scenario_id="scenario-001")],
            aggregates={"mean_recall@10": 0.9},
        ),
    ],
)
def test_models_round_trip(instance: BaseModel) -> None:
    """Each model should survive model_dump/model_validate round-trips."""
    payload = instance.model_dump()
    restored = type(instance).model_validate(payload)
    assert restored.model_dump() == payload

