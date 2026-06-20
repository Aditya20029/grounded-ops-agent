"""Unit tests: ingestion is idempotent (no duplicate chunks) and prunes orphans."""

from __future__ import annotations

import pytest

from app.ingestion.indexer import ingest_documents
from app.ingestion.loaders import SourceDoc
from tests.fakes.providers import FakeEmbeddingProvider, InMemoryChunkWriter


def _docs() -> list[SourceDoc]:
    return [
        SourceDoc("DOC-1", "ticket", "Ticket one", "word here " * 400, markdown=False),
        SourceDoc(
            "DOC-2",
            "runbook",
            "Runbook two",
            "# Heading\n\n" + ("alpha beta gamma. " * 200),
            markdown=True,
        ),
    ]


@pytest.mark.unit
async def test_ingest_twice_yields_no_duplicates() -> None:
    provider = FakeEmbeddingProvider(64)
    writer = InMemoryChunkWriter()

    first = await ingest_documents(_docs(), provider, writer, batch_size=8)
    count_after_first = await writer.count()

    second = await ingest_documents(_docs(), provider, writer, batch_size=8)
    count_after_second = await writer.count()

    assert count_after_first == count_after_second
    assert first.chunks == second.chunks == count_after_first
    assert first.chunks > 2  # multiple chunks were produced

    for row in writer.rows:
        assert row.embedding_model == provider.model_name
        assert len(row.embedding) == 64
        assert row.chunk_id.startswith(row.doc_id + "::")


@pytest.mark.unit
async def test_orphans_pruned_when_document_shrinks() -> None:
    provider = FakeEmbeddingProvider(32)
    writer = InMemoryChunkWriter()

    big = [SourceDoc("DOC-1", "ticket", "T", "sentence here. " * 300, markdown=False)]
    await ingest_documents(big, provider, writer, batch_size=8)
    assert await writer.count() > 1

    small = [SourceDoc("DOC-1", "ticket", "T", "tiny", markdown=False)]
    await ingest_documents(small, provider, writer, batch_size=8)
    assert await writer.count() == 1  # stale chunks were pruned
