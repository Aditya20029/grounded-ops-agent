# grounded-ops-agent

> A grounded operations assistant that answers questions over a company's
> operational records by combining **retrieval** (the qualitative "why") with
> **live analytics tools over MCP** (the quantitative "how much" and "when"),
> and grounds every answer with **inline citations** back to source records,
> under hard agent guardrails.

[![CI](https://github.com/Aditya20029/grounded-ops-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/Aditya20029/grounded-ops-agent/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)

> **Status:** v1 complete. Built in numbered phases; the app is runnable after
> every phase, with green CI throughout. A minimal web UI ships in `frontend/`.

<!-- TODO(demo): record demo.gif of the headline query and drop it here. -->
<!-- ![demo](docs/assets/demo.gif) -->
<!-- TODO(screenshots): add docs/assets/ui.png (answer + sources + trace). -->

---

## What it does

The headline query it supports end to end:

> **"What was the average resolution time for P1 incidents last quarter, and what
> were the top 3 recurring root causes?"**

The agent computes the metric through an MCP analytics tool, retrieves the
relevant postmortems, returns a grounded answer with inline citations and a
sources list, and exposes a step-by-step tool-call trace.

## Architecture

```mermaid
flowchart TD
    U[User query] --> API[FastAPI /chat SSE]
    API --> AG[Agent orchestrator + guardrails]
    AG --> SEED[Initial hybrid retrieval seed]
    SEED --> RET[Retrieval module]
    AG --> ADP[Tool adapter: MCP schema to provider tools]
    ADP --> MC[MCP client]
    MC --> MS[MCP analytics server, separate process]
    MS --> PG[(Postgres: pgvector + structured tables)]
    RET --> PG
    RET --> FS[FAISS index + id map on disk]
    AG --> LLM[LLM provider: OpenAI or Claude]
    LLM --> AG
    AG --> REG[Citation registry + post-validation]
    REG --> API
    API --> U
```

Full design rationale: [docs/architecture.md](docs/architecture.md). The original
build specification: [docs/BUILD_PROMPT.md](docs/BUILD_PROMPT.md).

## Features

- **Hybrid retrieval** over pgvector (dense) + Postgres full-text (keyword), fused
  with Reciprocal Rank Fusion, with an optional cross-encoder reranker.
- **Two vector stores by design** — pgvector as the canonical, filterable,
  persistent source of truth; FAISS as a low-latency in-memory path and a
  benchmark baseline. The eval harness reports recall@k and p50/p95 latency for
  both. ([Why two stores?](#why-two-vector-stores))
- **MCP analytics tools** — a standalone read-only MCP server exposing
  `query_metrics`, `get_timeseries`, `aggregate`, `get_record`, `search_records`,
  and `list_schema`, with **identifier whitelisting** that closes the SQL
  identifier-injection vector.
- **Agentic loop with hard guardrails** — max steps, cycle detection, a
  per-request token budget, tool timeouts, retries, and read-only tools.
  ([How guardrails work](#how-guardrails-work))
- **Citation grounding** — a per-request citation registry with stable indices, a
  `sources` array on every answer, post-validation that strips hallucinated
  citations, and a faithfulness check.
- **Provider-agnostic** — Anthropic Claude or OpenAI behind one `LLMProvider`;
  sentence-transformers or OpenAI behind one `EmbeddingProvider`. Model names and
  prices live in config, never in code.
- **Reproducible evaluation** — seeded synthetic data, a gold set, and a metrics
  table (retrieval, citation, faithfulness, latency, cost).
- **Production hygiene** — Pydantic v2 models, async handlers, structured JSON
  logging with request ids, retry/backoff on every external call, no secrets in
  the repo, and CI that runs with no API keys.

## Quickstart

Prerequisites: Docker (for Postgres+pgvector), Python 3.11+, and optionally
`make`. From a clean clone:

```bash
make install     # editable install with dev extras
make up          # start Postgres+pgvector, wait until healthy
make migrate     # apply database migrations
make seed        # generate the synthetic corpus + load structured tables
make ingest      # chunk, embed, upsert into pgvector, build FAISS
make run         # serve the API + web UI on http://127.0.0.1:8000
make smoke       # end-to-end smoke test of the headline query
make eval        # print the evaluation metrics table
```

Then open **http://127.0.0.1:8000/** for the web UI, or `POST /chat`, `/search`,
`/ingest` directly (`/docs` for the OpenAPI page). `make mcp` runs the MCP
analytics server standalone (for the networked `streamable-http` transport).

No API keys are needed to develop or to run the test suite: the default
embedding model (`BAAI/bge-small-en-v1.5`) runs locally on CPU, and tests use
deterministic offline fakes. An LLM provider key is required only to actually
generate answers; copy `.env.example` to `.env` and fill it in.

> **Windows without `make`:** every target maps to a single command shown in the
> [Makefile](Makefile); run those directly (e.g. `docker compose up -d --wait`,
> `python -m alembic upgrade head`, `python scripts/seed_db.py`, ...).

## Example queries

<!-- Sample outputs below illustrate the response shape; they will be replaced
     with captured runs once the agent loop (Phase 7) and API (Phase 8) land. -->

1. **"What was the average resolution time for P1 incidents last quarter, and what
   were the top 3 recurring root causes?"** — computes the metric via the
   `query_metrics` / `aggregate` MCP tools, retrieves the relevant postmortems,
   and returns a grounded answer with `[n]` citations plus a `sources` list and a
   tool-call trace.
2. **"Show the weekly trend of P1 incidents for the `payments` service this
   year."** — uses `get_timeseries`, grounded against the incident records.
3. **"Why did the checkout outage on the payments service happen, and how was it
   remediated?"** — retrieval-only, grounded against the postmortem with inline
   citations.
4. **"Which customers are on an enterprise SLA and had a P1 in the last 30 days?"**
   — hybrid SQL + vector retrieval with structured filters.

## Evaluation

`make eval` runs the harness over a 30-item gold set
([src/app/eval/gold.jsonl](src/app/eval/gold.jsonl), regenerable with
`python scripts/build_gold.py`) and prints a Markdown table of retrieval metrics
(recall@k, nDCG@k, MRR), citation precision/recall, faithfulness, p50/p95 latency,
and average cost per query, plus a pgvector-vs-FAISS recall/latency comparison.
Numbers are reproducible (seeded data, pinned judge model, temperature 0) and the
report is stamped with the models and date.

Relevance is scored at the document level (robust to chunk-boundary changes).
Faithfulness (LLM-as-judge, 3-point rubric) runs only when an LLM key is
configured; with the offline `fake`/`echo` providers the harness still produces
reproducible retrieval and citation numbers with **no key and no model download**.

> _Run `make eval` from a clean clone to generate the table below. Example shape
> (replace with your run's output, which is printed with the exact models + date):_

| Metric | Value |
| --- | --- |
| recall@5 | `make eval` |
| nDCG@5 | `make eval` |
| MRR | `make eval` |
| Citation precision / recall | `make eval` |
| Faithfulness (3-pt rubric) | `make eval` (needs an LLM key) |
| p50 / p95 latency (retrieval) | `make eval` |
| Avg cost / query | `make eval` |
| FAISS vs pgvector (recall@5, p50/p95) | `make eval` |

## Why two vector stores

pgvector is the **canonical** store: it lives in the same Postgres as the
structured operational data, enabling hybrid SQL + vector queries (semantic
search filtered by `priority`/`status`/`service`/`created_at`, joined to
structured tables), with cosine distance, an HNSW index, transactional
consistency, and persistence. FAISS is a **derived** in-memory index for a
low-latency semantic path and as a benchmark baseline (Flat/IVF/HNSW) for recall
and latency. FAISS stores vectors only, so a `faiss-id -> chunk_id` mapping is
persisted alongside it and rebuilt from pgvector if missing or stale. The eval
benchmark comparing the two is the reason both exist.

## How guardrails work

One guaranteed initial hybrid retrieval seeds context; then a bounded loop lets
the model call tools via native tool-calling and finally answer. Bounds:
`MAX_AGENT_STEPS` (hard stop), cycle detection on hashed `(tool, args)`, a
`PER_REQUEST_TOKEN_BUDGET` enforced by pre-step estimation and reconciled against
actual usage, `TOOL_TIMEOUT_SECONDS` + `MAX_TOOL_RETRIES`, per-request cost
accounting, and a structured tool-call trace. Retrieved content and tool output
are treated as untrusted **data** (delimited, never executed as instructions).

## Tech stack

Python 3.11+ · FastAPI + uvicorn · Pydantic v2 + pydantic-settings ·
PostgreSQL + pgvector · SQLAlchemy 2 (async) + asyncpg + Alembic · faiss-cpu ·
sentence-transformers · Anthropic + OpenAI SDKs · MCP Python SDK · pytest /
ruff / mypy · Docker Compose · GitHub Actions.

## Configuration

All configuration is environment-driven and validated at startup
(fail-fast). See [.env.example](.env.example) for every variable with a comment.
At least one LLM provider key is required; the OpenAI embedding option requires
its key. Prices in [config/pricing.json](config/pricing.json) are configurable
and may be out of date.

## Development

```bash
make lint        # ruff check + format check
make typecheck   # mypy (strict-ish)
make test        # full suite
make test-unit   # offline unit tests (no DB, no keys)
make test-integration  # needs Postgres+pgvector
```

CI runs lint, type check, unit tests, then migrations + integration tests against
a pgvector service — with no provider API keys (providers are faked in tests).

## Deploy notes

- **Database:** any Postgres 16 with the `pgvector` extension (the compose file
  uses `pgvector/pgvector:pg16`). Point `DATABASE_URL` at it and run `make migrate`.
- **MCP transport:** local development spawns the analytics server over **stdio**
  per request. For a deployment, run the server as a long-lived process with
  `MCP_TRANSPORT=streamable-http` (`make mcp`) and set `MCP_SERVER_URL`; the client
  connects over HTTP instead of spawning a subprocess.
- **Providers/keys:** set `LLM_PROVIDER`/`LLM_MODEL` and the matching key. For a
  zero-key demo, use `EMBEDDING_PROVIDER=fake` and `LLM_PROVIDER=fake` (offline
  hashing embeddings + a deterministic echo model) so the whole pipeline runs from
  a clean clone; analytics tool planning needs a real LLM.
- **Web UI:** `frontend/index.html` is served at `/` by the app (dependency-free
  vanilla JS that consumes the SSE stream); the API is also at `/docs`.
- **Observability:** structured JSON logs with a request id per request; per-request
  token and USD cost are returned in the `done` event. An OpenTelemetry/Langfuse
  hook can be added behind a flag (off by default).

## Roadmap

- [x] **Phase 0/1** — repo, tooling, docker-compose, CI, README.
- [x] **Phase 2** — data model, migrations, seeded synthetic dataset.
- [x] **Phase 3** — embeddings + idempotent ingestion.
- [x] **Phase 4** — vector stores, hybrid retrieval, FAISS-vs-pgvector benchmark.
- [x] **Phase 5** — generation + citation grounding.
- [x] **Phase 6** — MCP analytics server + client + tool adapter.
- [x] **Phase 7** — agent orchestrator + guardrails.
- [x] **Phase 8** — FastAPI surface + SSE protocol.
- [x] **Phase 9** — evaluation harness (`make eval`).
- [x] **Phase 10** — minimal frontend + deploy notes.
- [ ] **Next** — capture demo GIF + screenshots; optional cross-encoder rerank by
  default; richer hybrid SQL filters (join chunks to incidents by service/severity).

## License

[MIT](LICENSE).
