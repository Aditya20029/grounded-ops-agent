# Build specification

This file records the specification this project was built against, so reviewers
can see the intended scope and judge the implementation against it. Design
decisions made where the spec left something open are recorded in the README and
in `docs/architecture.md`.

## 1. Mission

A grounded operations assistant that answers questions over a company's
structured and unstructured operational records. It combines retrieval (for the
qualitative "why") with live analytics tools exposed over MCP (for the
quantitative "how much" and "when"), and grounds every answer with inline
citations back to source records. An agent loop interleaves retrieval and tool
calls under hard guardrails: maximum steps, cycle detection, a per-request token
budget, tool timeouts, and read-only tools.

Headline demo query: _"What was the average resolution time for P1 incidents last
quarter, and what were the top 3 recurring root causes?"_ — the agent computes
the metric through an MCP analytics tool, retrieves the relevant postmortems,
returns a grounded answer with inline citations and a sources list, and exposes a
step-by-step tool-call trace.

## 2. Design principles and invariants

1. **One embedding model at a time.** Query and document embeddings come from the
   same model; the active model id and dimension are stamped into the index
   metadata and asserted at query time.
2. **The two vector stores stay in sync.** pgvector is the source of truth; FAISS
   is a derived index, always rebuildable from pgvector.
3. **Tools are read-only.** No writes, no arbitrary SQL, no non-whitelisted
   identifiers.
4. **Retrieved content and tool output are untrusted data, not instructions.**
5. **Ingestion is idempotent.** Upsert on `chunk_id`; never duplicate.
6. **Graded evaluation is reproducible.** Seeded generator, temperature-0 graded
   runs, recorded models + date.
7. **No secrets in the repo.** Env-driven, validated at startup.

## 3. Phase 0: repository and GitHub

New local project `grounded-ops-agent`, git, GitHub repo (public), MIT `LICENSE`,
thorough `.gitignore`, `.env.example` with every variable documented, README
skeleton. Tooling: `pyproject.toml` for ruff/mypy/pytest; a `Makefile` with
`install up down migrate seed ingest run mcp test test-unit test-integration lint
typecheck eval smoke`. CI workflow. First commit + push.

## 4. Domain and data (swappable)

Fictional SaaS operations, seeded with Faker. Unstructured corpus to `data/seed/`:
~500 support tickets, ~60 incident postmortems (Markdown), ~30 runbooks. Structured
tables in Postgres: `incidents` (severity P1..P4, service, opened_at, resolved_at,
root_cause_category), `metrics_daily` (~1 year of daily rows), `customers`, `slas`,
`tickets`. Seeding is idempotent (truncate-and-reload or upsert behind a flag).

## 5. Architecture

See `docs/architecture.md` for the diagram and component map.

## 6. Component specifications

- **6.1 Data model & migrations** — Alembic (or versioned `init.sql`). First
  migration: `CREATE EXTENSION IF NOT EXISTS vector`, structured tables, and
  `chunks(chunk_id pk, doc_id, source_type, title, chunk_index, char_start,
  char_end, content, embedding vector(<DIM>), embedding_model, created_at)`. HNSW
  index with `vector_cosine_ops`. pgvector-enabled compose image + healthcheck.
- **6.2 Embeddings & providers** — `EmbeddingProvider` (HF sentence-transformers
  `BAAI/bge-small-en-v1.5` default 384d; OpenAI `text-embedding-3-small` 1536d).
  `LLMProvider` (OpenAI + Anthropic) with `complete()`, `stream()`, native
  tool-calling, and `count_tokens()`. Model names from config. Retry/backoff on
  all external calls.
- **6.3 Ingestion** — loaders per source type; token-aware chunker with overlap
  (target 500 tokens, 80 overlap), header-aware Markdown splitting; per-chunk
  metadata; embed + stamp `embedding_model`; upsert keyed on `chunk_id`; then
  build/update FAISS.
- **6.4 Vector stores** — `VectorStore` interface; pgvector canonical (HNSW,
  cosine, hybrid SQL filters); FAISS derived (in-memory, id-map persisted,
  rebuildable, concurrent-rebuild lock). Benchmark recall@k + p50/p95 latency.
- **6.5 Retrieval** — dense (pgvector + FAISS), keyword (Postgres FTS / BM25), RRF
  (`score(d) = sum 1/(k + rank)`, k=60), optional cross-encoder reranker
  (`BAAI/bge-reranker-base`, off by default). Single implementation shared by the
  orchestrator seed and the `search_records` tool.
- **6.6 Tool adapter** — MCP JSON-schema -> OpenAI function format and Anthropic
  tool format; normalize tool results into one internal shape. Unit-tested both
  directions.
- **6.7 MCP analytics server** — standalone process, read-only tools over stdio
  (local) and streamable HTTP (networked): `list_schema`, `query_metrics`,
  `get_timeseries`, `aggregate`, `get_record`, `search_records`. Identifier
  whitelist (`ALLOWED = {table: {columns, group_by_columns}}`,
  `ALLOWED_AGG = {count,sum,avg,min,max}`); only values are bound; row cap +
  input-size limits.
- **6.8 MCP client** — thin typed client; lists/calls tools; manages server
  lifecycle (spawn/shutdown for stdio); handles tool errors and timeouts.
- **6.9 Orchestrator & guardrails** — seed retrieval then bounded tool loop.
  `MAX_AGENT_STEPS=6`, cycle detection on hashed `(tool, normalized_args)`,
  `PER_REQUEST_TOKEN_BUDGET=30000` (pre-step estimate + reconcile actual usage),
  `TOOL_TIMEOUT_SECONDS=15`, `MAX_TOOL_RETRIES=2`, cost accounting, structured
  trace.
- **6.10 Citation grounding** — per-request registry with stable indices; labelled
  delimited chunks; `[n]` citations; `sources` array of `{index, chunk_id, doc_id,
  title, snippet, char_span}`; post-validation strips hallucinated citations;
  faithfulness check (LLM judge rubric or NLI).
- **6.11 Trust boundaries & security** — untrusted data delimited in prompts;
  read-only whitelisted tools; input-size limits; no secrets in code or logs.
- **6.12 FastAPI surface & SSE** — `POST /chat` (SSE), `POST /ingest`,
  `POST /search`, `GET /health`. SSE events: `status`, `tool_call`, `tool_result`,
  `token`, `sources`, `trace`, `done`, `error`. Pydantic v2 models, async, asyncpg
  pool, request id everywhere, structured errors.

## 7. Tech stack

Python 3.11+, FastAPI + uvicorn, Pydantic v2 + pydantic-settings, Postgres +
pgvector, SQLAlchemy + asyncpg + Alembic, faiss-cpu, sentence-transformers,
OpenAI + Anthropic SDKs, MCP Python SDK, pytest/ruff/mypy, Docker, GitHub
Actions. Pin versions only after checking current releases; use current model
names; do not bake model strings or prices into code.

## 8-19

Repository structure, build phases (each ending in a passing build + commit),
quality bar (full type hints, structured logging, retry/backoff, no secrets),
testing strategy (offline fakes; unit + integration markers; ~80% on core), CI
(pgvector service; mocked providers), pricing config (per-1M-token), evaluation
& metrics (gold set; recall@k / nDCG@k / MRR / citation precision-recall /
faithfulness; temperature-0 graded; p50/p95 latency; cost/query), observability +
optional frontend, configuration & environment (pydantic-settings, fail-fast),
working agreement (phase by phase, conventional commits, verify library/SDK/model
details from docs), README requirements, and the v1 definition of done.

The full original prompt text is preserved in the project's commit history and
issue tracker; this document captures the requirements the code is held to.
