# grounded-ops-agent  --  developer task runner.
# Recipes are intentionally single, cross-platform commands so they work under
# GNU make on Linux/macOS (CI) and Windows (cmd.exe). For Windows without make,
# every command below can be run directly; see README "Quickstart".

PYTHON ?= python

.PHONY: help install up down migrate seed ingest run mcp \
        test test-unit test-integration lint format typecheck eval smoke clean

help:
	@echo "Targets: install up down migrate seed ingest run mcp test test-unit test-integration lint format typecheck eval smoke clean"

install:               ## Install the package (editable) with dev extras
	$(PYTHON) -m pip install -e ".[dev]"

up:                    ## Start Postgres+pgvector and wait until healthy
	docker compose up -d --wait

down:                  ## Stop and remove containers
	docker compose down

migrate:               ## Apply database migrations
	$(PYTHON) -m alembic upgrade head

seed:                  ## Generate synthetic corpus + load structured tables (idempotent)
	$(PYTHON) scripts/seed_db.py

ingest:                ## Chunk + embed the corpus into pgvector, then build FAISS
	$(PYTHON) scripts/ingest.py

run:                   ## Run the FastAPI app
	$(PYTHON) -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

mcp:                   ## Run the MCP analytics server (stdio by default)
	$(PYTHON) -m app.mcp_server

test:                  ## Run the full test suite
	$(PYTHON) -m pytest

test-unit:             ## Unit tests only (no external deps)
	$(PYTHON) -m pytest -m "not integration"

test-integration:      ## Integration tests (need Postgres+pgvector)
	$(PYTHON) -m pytest -m integration

lint:                  ## Lint + format check
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

format:                ## Auto-format and apply safe lint fixes
	$(PYTHON) -m ruff format .
	$(PYTHON) -m ruff check --fix .

typecheck:             ## Static type check
	$(PYTHON) -m mypy src

eval:                  ## Run the evaluation harness and print the metrics table
	$(PYTHON) scripts/run_eval.py

smoke:                 ## End-to-end smoke test of the headline query
	$(PYTHON) scripts/smoke.py

clean:                 ## Remove caches and the derived FAISS index
	$(PYTHON) -c "import shutil,glob,os; [shutil.rmtree(p,ignore_errors=True) for p in ['.pytest_cache','.mypy_cache','.ruff_cache','htmlcov']]"
