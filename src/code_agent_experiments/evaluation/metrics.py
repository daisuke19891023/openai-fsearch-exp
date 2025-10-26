"""Metrics calculations for retrieval experiments."""

from __future__ import annotations

import math
from statistics import fmean
from typing import Iterable, Sequence

from code_agent_experiments.domain.models import Metrics, RetrievalRecord, Scenario


def _sorted_candidates(record: RetrievalRecord) -> list[tuple[int, str]]:
    """Return (rank, path) pairs ordered by rank or insertion order."""
    ranked: list[tuple[int, str]] = []
    for index, candidate in enumerate(record.candidates):
        rank = candidate.rank if candidate.rank is not None else index + 1
        ranked.append((rank, candidate.path))
    ranked.sort(key=lambda item: item[0])
    return ranked


def _recall_at_k(
    ranked_paths: Sequence[str],
    ground_truth: set[str],
    k: int,
) -> float:
    if not ground_truth:
        return 0.0
    top_k = ranked_paths[:k]
    hits = sum(1 for path in top_k if path in ground_truth)
    return hits / len(ground_truth)


def _dcg_at_k(ranked_paths: Sequence[str], ground_truth: set[str], k: int) -> float:
    dcg = 0.0
    for idx, path in enumerate(ranked_paths[:k]):
        relevance = 1.0 if path in ground_truth else 0.0
        if relevance:
            dcg += relevance / math.log2(idx + 2)
    return dcg


def _ndcg_at_k(ranked_paths: Sequence[str], ground_truth: set[str], k: int) -> float:
    if not ground_truth:
        return 0.0
    dcg = _dcg_at_k(ranked_paths, ground_truth, k)
    ideal_relevances = [1.0] * min(k, len(ground_truth))
    idcg = 0.0
    for idx, rel in enumerate(ideal_relevances):
        idcg += rel / math.log2(idx + 2)
    if idcg == 0:
        return 0.0
    return dcg / idcg


def _mrr(ranked_paths: Sequence[str], ground_truth: set[str]) -> float:
    for idx, path in enumerate(ranked_paths):
        if path in ground_truth:
            return 1.0 / (idx + 1)
    return 0.0


def compute_retrieval_metrics(
    scenario: Scenario,
    record: RetrievalRecord,
    k_values: Iterable[int] = (5, 10, 20),
) -> Metrics:
    """Compute retrieval metrics for a scenario given its retrieval record."""
    ranked = _sorted_candidates(record)
    ranked_paths = [path for _, path in ranked]
    ground_truth = set(scenario.ground_truth_files)
    sorted_k = sorted({int(k) for k in k_values})

    recall = {k: _recall_at_k(ranked_paths, ground_truth, k) for k in sorted_k}
    ndcg = {k: _ndcg_at_k(ranked_paths, ground_truth, k) for k in sorted_k}
    mrr = _mrr(ranked_paths, ground_truth)

    return Metrics(
        scenario_id=scenario.id,
        run_id=record.run_id,
        strategy=record.strategy,
        k_values=sorted_k,
        recall_at_k=recall,
        ndcg_at_k=ndcg,
        mrr=mrr,
        tool_calls=len(record.tool_calls),
        wall_ms=record.elapsed_ms,
        cost_usd=None,
    )


def aggregate_metrics(per_scenario: Sequence[Metrics]) -> dict[str, float | int | dict[int, float] | None]:
    """Aggregate per-scenario metrics into run-level statistics."""
    if not per_scenario:
        return {
            "scenario_count": 0,
            "avg_recall_at_k": {},
            "avg_ndcg_at_k": {},
            "mean_mrr": 0.0,
            "total_tool_calls": 0,
            "total_wall_ms": 0,
            "total_cost_usd": None,
        }

    k_values = sorted({k for metrics in per_scenario for k in metrics.k_values})
    avg_recall = {
        k: fmean(metrics.recall_at_k.get(k, 0.0) for metrics in per_scenario)
        for k in k_values
    }
    avg_ndcg = {
        k: fmean(metrics.ndcg_at_k.get(k, 0.0) for metrics in per_scenario)
        for k in k_values
    }
    mean_mrr = fmean(metrics.mrr for metrics in per_scenario)
    total_tool_calls = sum(metrics.tool_calls for metrics in per_scenario)
    total_wall_ms = sum(metrics.wall_ms for metrics in per_scenario)

    costs = [metrics.cost_usd for metrics in per_scenario if metrics.cost_usd is not None]
    total_cost = float(sum(costs)) if costs else 0.0

    return {
        "scenario_count": len(per_scenario),
        "avg_recall_at_k": avg_recall,
        "avg_ndcg_at_k": avg_ndcg,
        "mean_mrr": mean_mrr,
        "total_tool_calls": total_tool_calls,
        "total_wall_ms": total_wall_ms,
        "total_cost_usd": total_cost if costs else None,
    }
