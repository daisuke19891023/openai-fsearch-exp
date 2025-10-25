# Code Agent Experiments MVP Specification

## 1. Product Overview
- **Goal:** Provide an experimental framework to evaluate how LLM-based coding agents discover relevant source files for bug fixes.
- **Scope (MVP):** Focus on candidate file discovery using multiple search strategies (shell tools, vector indices, hybrid flows) and measure recall-oriented metrics. Automated fixes, large-scale indexing, and distributed execution are out of scope.
- **Primary Capabilities:**
  - Compare search tools (ripgrep, grep, fd, find) and vector stores (FAISS, pgvector, Chroma).
  - Experiment with chunking strategies (fixed-size tokens, AST boundaries, future LSP support).
  - Observe model prompting, tool selection behavior, and collect reproducible traces.
  - Persist experiment artefacts for evaluation and reporting.

## 2. System Architecture
- **CLI Runner (Typer):** Entry point exposing commands for initialization, indexing, querying, scenario execution, evaluation, reporting, benchmarking shell tools, and optional MCP connectivity checks.
- **Experiment Orchestrator:** Coordinates Runs, manages agent interactions, and invokes tool bridges.
- **Agent Runner:** Interfaces with OpenAI Responses API / Agents SDK, registering Python tool bridges and optional rerankers.
- **Tool Bridge:** Wraps shell, vector, AST, and optional LSP tooling; enforces subprocess safety (timeouts, binary checks).
- **Storage Layer:** Uses SQLite plus Parquet/JSON for artefacts, and per-backend directories for indices.
- **Observability:** Persist all tool calls and LLM traces; expose telemetry via SQLite and saved logs.

## 3. Domain Model Summary
All domain entities are defined with Pydantic v2 models under `domain.models`.
- `ToolName` literal enumerates supported tool identifiers.
- Configurations: `ChunkingConfig`, `VectorBackendConfig`, `RerankerConfig`, `ModelConfig`, `ToolLimits`, `RunConfig`.
- Scenario data: `Scenario` holds repository path, natural language query, and ground-truth files.
- Execution telemetry: `ToolCall`, `Candidate`, `RetrievalRecord`, `Metrics`, `ReportSummary`.
- Tool I/O schemas live in `domain.tool_schemas` and mirror the MVP definitions for ripgrep, fd, and vector search.

## 4. CLI Surface
Implemented via Typer with the following commands:
- `cae init` – bootstrap project files and environment templates.
- `cae index` – generate repository indices with configurable backends and chunking.
- `cae query` – run single interactive agent query against a repo.
- `cae run-scenarios` – execute batches of scenarios with replication.
- `cae eval` – aggregate metrics from stored retrieval results.
- `cae report` – produce Markdown/HTML experiment summaries.
- `cae bench tools` – micro-benchmark shell tools.
- `cae mcp check` – optional LSP bridge validation.

## 5. Retrieval Pipeline
1. **Candidate Generation:** Use `fd` to scope files and `ripgrep` for textual hits.
2. **Semantic Completion:** Query vector indices (FAISS/pgvector/Chroma) to augment candidates.
3. **Reranking (optional):** Apply cross-encoder rerankers (e.g., BGE) on top-k candidates.
4. **Verification (optional):** Use AST or LSP lookups for structural confirmation.

## 6. Indexing & Embeddings
- Utilize OpenAI Embeddings with chunkers (`fixed` or `ast` strategies) to produce token windows or syntax-aware spans.
- Persist metadata (path, language, symbol, span, hash) alongside embeddings.
- Store indices under `.index/` subdirectories according to backend requirements.

## 7. Evaluation & Reporting
- Metrics: Recall@K, nDCG@K, MRR, tool-call counts, wall-clock times, optional USD cost estimates.
- Artefacts: Store `RetrievalRecord` JSON files per scenario and aggregated `Metrics` JSONL per run.
- Reporting: Generate Markdown/HTML summaries with per-scenario and aggregate breakdowns.

## 8. Logging & Safety
- Enforce subprocess safety (timeouts, argument sanitization, binary presence checks).
- Clamp tool outputs to reasonable limits (e.g., ≤5k lines or 1k files) to avoid overloads.
- Keep MCP/LSP interactions read-only during MVP.

## 9. Project Configuration
- Python 3.11+, managed via `uv`.
- Dependencies per MVP template: OpenAI SDKs, Pydantic, Typer, Rich, Orjson, FAISS/Chroma/pgvector, tree-sitter, numpy, pandas, jinja2.
- Dev extras include pytest, pytest-cov, ruff, mypy.
- `pyproject.toml` defines CLI script entry point (Typer) and tool configurations (ruff, mypy).

## 10. Roadmap & Tasks
The MVP is divided into epics (EP0–EP11) covering bootstrap, domain models, shell tools, chunkers, vector backends, rerankers, agent integration, evaluation, orchestration, SWE-bench importers, and operational safeguards. Each epic contains detailed task IDs with definitions of done and acceptance criteria.

