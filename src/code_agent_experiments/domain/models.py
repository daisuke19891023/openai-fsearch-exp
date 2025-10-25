"""Pydantic domain models for Code Agent Experiments."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

Datetime = datetime

ToolName = Literal[
    "ripgrep",
    "grep",
    "fd",
    "find",
    "vector.faiss",
    "vector.pgvector",
    "vector.chroma",
    "ast.tree_sitter",
    "ast.python",
    "lsp.mcp",
]


class ChunkingConfig(BaseModel):
    """Configuration for chunking source code into retrieval units."""

    type: Literal["fixed", "ast"] = "fixed"
    size_tokens: int = 400
    overlap_tokens: int = 80
    languages: list[str] = Field(default_factory=lambda: ["python"])


class VectorBackendConfig(BaseModel):
    """Configuration for vector retrieval backends."""

    backend: Literal["faiss", "pgvector", "chroma"] = "faiss"
    index_path: str | None = None
    connection_url: str | None = None
    metric: Literal["cosine", "l2", "ip"] = "cosine"


class RerankerConfig(BaseModel):
    """Optional reranker configuration."""

    name: str | None = None
    top_k: int = 100


class ModelConfig(BaseModel):
    """Configuration for the LLM used in a run."""

    name: str
    temperature: float = 0.0
    max_output_tokens: int = 2048


class ToolLimits(BaseModel):
    """Limits imposed on tool usage for a run."""

    max_calls: int = 16
    per_call_timeout_sec: int = 20


class RunConfig(BaseModel):
    """Experiment run configuration."""

    id: str
    model: ModelConfig
    tools_enabled: list[ToolName]
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    vector: VectorBackendConfig | None = None
    reranker: RerankerConfig | None = None
    tool_limits: ToolLimits = Field(default_factory=ToolLimits)
    seed: int = 7
    replicate: int = 1


class Scenario(BaseModel):
    """Single evaluation scenario definition."""

    id: str
    repo_path: str
    query: str
    language: str | None = "python"
    ground_truth_files: list[str]
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolCall(BaseModel):
    """Telemetry for a single tool invocation."""

    name: ToolName
    args: dict[str, Any]
    started_at: Datetime
    ended_at: Datetime | None = None
    success: bool | None = None
    stderr_preview: str | None = None
    stdout_preview: str | None = None
    token_in: int = 0
    token_out: int = 0
    latency_ms: int | None = None


class Candidate(BaseModel):
    """A retrieved candidate file with associated scores."""

    path: str
    score_keyword: float = 0.0
    score_dense: float = 0.0
    score_rerank: float = 0.0
    rank: int | None = None
    features: dict[str, Any] = Field(default_factory=dict)


class RetrievalRecord(BaseModel):
    """Full retrieval trace for a scenario/run combination."""

    scenario_id: str
    run_id: str
    strategy: str
    candidates: list[Candidate]
    tool_calls: list[ToolCall]
    elapsed_ms: int


class Metrics(BaseModel):
    """Aggregate retrieval metrics."""

    scenario_id: str
    run_id: str
    strategy: str
    k_values: list[int] = Field(default_factory=lambda: [5, 10, 20])
    recall_at_k: dict[int, float]
    ndcg_at_k: dict[int, float]
    mrr: float
    tool_calls: int
    wall_ms: int
    cost_usd: float | None = None


class ReportSummary(BaseModel):
    """Aggregated metrics for an experiment run."""

    run_id: str
    per_scenario: list[Metrics]
    aggregates: dict[str, Any]
