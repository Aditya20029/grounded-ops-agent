"""Integration test: ingestion writes chunks to pgvector and is idempotent.

Uses the offline hashing embedding provider so it needs no model download or API
key; the vector dimension matches the migration's ``vector(<dim>)`` column.
"""

from __future__ import annotations

import pytest
from sqlalchemy import text

from app.core.db import session_scope
from app.core.settings import get_settings
from app.ingestion.indexer import ingest_documents
from app.ingestion.loaders import SourceDoc
from app.ingestion.store import PgChunkWriter
from app.llm.embeddings import HashingEmbeddingProvider

pytestmark = pytest.mark.integration


async def _truncate_chunks() -> None:
    async with session_scope() as session:
        await session.execute(text("TRUNCATE chunks"))


async def test_ingest_writes_and_is_idempotent(db_engine: object) -> None:
    await _truncate_chunks()

    provider = HashingEmbeddingProvider(get_settings().embedding_dim)
    writer = PgChunkWriter()
    docs = [
        SourceDoc("DOC-A", "ticket", "A", "payments latency spike. " * 60, markdown=False),
        SourceDoc(
            "DOC-B",
            "postmortem",
            "B",
            "# Root cause\n\ndeployment regression. " * 40,
            markdown=True,
        ),
    ]

    first = await ingest_documents(docs, provider, writer)
    count_first = await writer.count()
    assert count_first == first.chunks > 0

    # Re-ingesting the same corpus upserts in place: no duplicate rows.
    second = await ingest_documents(docs, provider, writer)
    count_second = await writer.count()
    assert count_second == count_first == second.chunks
