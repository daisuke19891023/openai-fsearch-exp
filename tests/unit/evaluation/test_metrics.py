"""Tests for retrieval metric calculations."""

from __future__ import annotations

import math

from code_agent_experiments.domain.models import Candidate, RetrievalRecord, Scenario
from code_agent_experiments.evaluation.metrics import aggregate_metrics, compute_retrieval_metrics


def _make_record(run_id: str, strategy: str) -> RetrievalRecord:
    return RetrievalRecord(
        scenario_id="scenario-1",
        run_id=run_id,
        strategy=strategy,
        candidates=[
            Candidate(path="a.py", rank=1),
            Candidate(path="c.py", rank=2),
            Candidate(path="b.py", rank=3),
            Candidate(path="d.py", rank=4),
        ],
        tool_calls=[],
        elapsed_ms=1200,
    )


def test_compute_retrieval_metrics_matches_reference_values() -> None:
    """Ensure metric calculations match a known ranking example."""
    scenario = Scenario(
        id="scenario-1",
        repo_path="/workspace/repo",
        query="Fix bug",
        ground_truth_files=["a.py", "b.py"],
    )
    record = _make_record("run-123", "keyword")

    metrics = compute_retrieval_metrics(scenario, record, k_values=[1, 2, 3])

    assert math.isclose(metrics.recall_at_k[1], 0.5, rel_tol=1e-6)
    assert math.isclose(metrics.recall_at_k[2], 0.5, rel_tol=1e-6)
    assert math.isclose(metrics.recall_at_k[3], 1.0, rel_tol=1e-6)

    assert math.isclose(metrics.ndcg_at_k[1], 1.0, rel_tol=1e-6)
    ideal_dcg_k2 = 1.0 + 1.0 / math.log2(3)
    assert math.isclose(metrics.ndcg_at_k[2], 1.0 / ideal_dcg_k2, rel_tol=1e-6)
    assert math.isclose(
        metrics.ndcg_at_k[3], (1.0 + 1.0 / math.log2(4)) / ideal_dcg_k2, rel_tol=1e-6,
    )

    assert math.isclose(metrics.mrr, 1.0, rel_tol=1e-6)
    assert metrics.tool_calls == 0
    assert metrics.wall_ms == 1200


def test_aggregate_metrics_reports_means() -> None:
    """Average metrics across scenarios for aggregate reporting."""
    scenario = Scenario(
        id="scenario-1",
        repo_path="/workspace/repo",
        query="Fix bug",
        ground_truth_files=["a.py", "b.py"],
    )
    record = _make_record("run-123", "keyword")
    metrics_one = compute_retrieval_metrics(scenario, record, k_values=[1, 3])
    metrics_one = metrics_one.model_copy(update={"cost_usd": 0.25})

    metrics_two = compute_retrieval_metrics(
        Scenario(
            id="scenario-2",
            repo_path="/workspace/repo",
            query="Fix bug",
            ground_truth_files=["does_not_exist.py"],
        ),
        RetrievalRecord(
            scenario_id="scenario-2",
            run_id="run-123",
            strategy="keyword",
            candidates=[Candidate(path="x.py", rank=1)],
            tool_calls=[],
            elapsed_ms=500,
        ),
        k_values=[1, 3],
    )

    summary = aggregate_metrics([metrics_one, metrics_two])

    assert summary["scenario_count"] == 2

    avg_recall = summary["avg_recall_at_k"]
    avg_ndcg = summary["avg_ndcg_at_k"]
    assert isinstance(avg_recall, dict)
    assert isinstance(avg_ndcg, dict)
    assert math.isclose(
        avg_recall[1],
        (metrics_one.recall_at_k[1] + metrics_two.recall_at_k[1]) / 2,
        rel_tol=1e-6,
    )
    assert math.isclose(
        avg_ndcg[3],
        (metrics_one.ndcg_at_k[3] + metrics_two.ndcg_at_k[3]) / 2,
        rel_tol=1e-6,
    )
    mean_mrr = summary["mean_mrr"]
    assert isinstance(mean_mrr, float)
    assert math.isclose(mean_mrr, (metrics_one.mrr + metrics_two.mrr) / 2, rel_tol=1e-6)
    total_tool_calls = summary["total_tool_calls"]
    assert isinstance(total_tool_calls, int)
    assert total_tool_calls == metrics_one.tool_calls + metrics_two.tool_calls
    total_wall_ms = summary["total_wall_ms"]
    assert isinstance(total_wall_ms, int)
    assert total_wall_ms == metrics_one.wall_ms + metrics_two.wall_ms
    total_cost = summary["total_cost_usd"]
    assert total_cost is not None
    assert isinstance(total_cost, (int, float))
    assert math.isclose(float(total_cost), 0.25, rel_tol=1e-6)
