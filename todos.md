# Implementation Tasks

## EP0 – Project Bootstrap
- [ ] T0-1: Initialize uv project and align `pyproject.toml`, add README and env templates.
- [ ] T0-2: Configure lint/type/test tooling via Nox, ensure sanity test coverage.

## EP1 – Domain Model Foundation
- [x] T1-1: Implement `domain.models` per MVP specification with round-trip tests.
- [x] T1-2: Build YAML/ENV config loader for run and scenario definitions.

## EP2 – Shell Tool Wrappers
- [x] T2-1: Implement ripgrep wrapper returning structured matches.
- [x] T2-2: Implement fd wrapper with gitignore-aware behavior.
- [x] T2-3: Provide grep/find wrappers for baseline comparisons.

## EP3 – Chunking & Embeddings
- [ ] T3-1: Implement fixed-token chunker.
- [ ] T3-2: Implement tree-sitter AST chunker.
- [ ] T3-3: Implement OpenAI embeddings client with batching and retries.

## EP4 – Vector Backends
- [ ] T4-1: Build FAISS backend with persistence.
- [ ] T4-2: Integrate Chroma backend (optional).
- [ ] T4-3: Integrate pgvector backend (optional).

## EP5 – Reranking (Optional)
- [ ] T5-1: Add cross-encoder reranker pipeline stage.

## EP6 – Responses API Integration
- [ ] T6-1: Register tool schemas and handle tool calls via Responses API.
- [ ] T6-2: Support iterative loops with tool limit enforcement and logging.

## EP7 – Agents SDK Integration (Optional)
- [ ] T7-1: Expose tools via Agents SDK with trace visualization.

## EP8 – Evaluation Suite
- [ ] T8-1: Implement metrics calculations (Recall@K, nDCG@K, MRR).
- [ ] T8-2: Generate Markdown/HTML reports.

## EP9 – Experiment Orchestration
- [ ] T9-1: Implement scenario runner with persistence.
- [ ] T9-2: Provide baseline experiment profiles.

## EP10 – SWE-bench Importer
- [ ] T10-1: Ingest SWE-bench scenarios into internal format.

## EP11 – Operational Safety
- [ ] T11-1: Add binary/version checks and graceful errors for missing tools.
- [ ] T11-2: Enforce output caps to avoid runaway subprocess output.
