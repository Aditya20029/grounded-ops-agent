"""Integration test: ingest then retrieve via hybrid pgvector and FAISS."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import text

from app.core.db import session_scope
from app.core.settings import get_settings
from app.ingestion.indexer import ingest_documents
from app.ingestion.loaders import SourceDoc
from app.ingestion.store import PgChunkWriter
from app.llm.embeddings import HashingEmbeddingProvider
from app.retrieval.pgvector_store import PgVectorStore
from app.retrieval.service import RetrievalService

pytest.importorskip("faiss")

from app.retrieval.faiss_store import FaissStore

pytestmark = pytest.mark.integration

_DOCS = [
    SourceDoc(
        "DOC-PAY",
        "postmortem",
        "Payments outage",
        "# Summary\n\npayments gateway timeout caused by a deployment regression in checkout",
        markdown=True,
    ),
    SourceDoc(
        "DOC-AUTH",
        "postmortem",
        "Auth outage",
        "# Summary\n\nauthentication login failures from a capacity shortfall in the auth service",
        markdown=True,
    ),
]


async def _truncate_chunks() -> None:
    async with session_scope() as session:
        await session.execute(text("TRUNCATE chunks"))


async def test_hybrid_pgvector_and_faiss(db_engine: object, tmp_path: Path) -> None:
    await _truncate_chunks()
    settings = get_settings()
    provider = HashingEmbeddingProvider(settings.embedding_dim)
    await ingest_documents(_DOCS, provider, PgChunkWriter())

    pg_store = PgVectorStore()
    service = RetrievalService(provider=provider, pg_store=pg_store, rrf_k=settings.rrf_k)
    hybrid = await service.retrieve("payments gateway timeout deployment", top_k=3)
    assert any(r.doc_id == "DOC-PAY" for r in hybrid)

    # FAISS rebuilds from pgvector and serves the same corpus.
    faiss_store = FaissStore(settings.embedding_dim, index_path=tmp_path / "idx.faiss")
    await faiss_store.ensure(pg_store, provider.model_name)
    assert faiss_store.ready and faiss_store.size > 0

    query_vector = provider.embed_query("authentication login capacity shortfall")
    faiss_results = await faiss_store.search(query_vector, top_k=3)
    assert any(r.doc_id == "DOC-AUTH" for r in faiss_results)
