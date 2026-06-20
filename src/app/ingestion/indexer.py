"""Ingestion orchestration: chunk -> embed -> idempotent upsert into the store.

Embedding runs in batches off the event loop (the providers are synchronous and
CPU-bound). Chunk ids are stable (``{doc_id}::{chunk_index}``) so re-running
ingestion upserts in place; orphaned chunks from a now-shorter document are
pruned. This is the single ingestion path; the FAISS index is built from
pgvector afterwards (Phase 4).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime

from app.core.logging import get_logger
from app.ingestion.chunker import chunk_document
from app.ingestion.loaders import SourceDoc
from app.ingestion.store import ChunkRow, ChunkWriter
from app.llm.embeddings import EmbeddingProvider

logger = get_logger(__name__)


@dataclass(frozen=True)
class IngestStats:
    documents: int
    chunks: int
    embedding_model: str


async def ingest_documents(
    docs: list[SourceDoc],
    provider: EmbeddingProvider,
    writer: ChunkWriter,
    *,
    batch_size: int = 64,
) -> IngestStats:
    """Chunk, embed, and upsert all documents; return ingestion stats."""
    pairs = [
        (doc, piece) for doc in docs for piece in chunk_document(doc.text, markdown=doc.markdown)
    ]
    texts = [piece.content for _, piece in pairs]

    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        vectors.extend(await asyncio.to_thread(provider.embed_documents, batch))

    created_at = datetime.now(UTC)
    rows_by_doc: dict[str, list[ChunkRow]] = {}
    for (doc, piece), vector in zip(pairs, vectors, strict=True):
        rows_by_doc.setdefault(doc.doc_id, []).append(
            ChunkRow(
                chunk_id=f"{doc.doc_id}::{piece.chunk_index}",
                doc_id=doc.doc_id,
                source_type=doc.source_type,
                title=doc.title,
                chunk_index=piece.chunk_index,
                char_start=piece.char_start,
                char_end=piece.char_end,
                content=piece.content,
                embedding=vector,
                embedding_model=provider.model_name,
                created_at=created_at,
            )
        )

    total = 0
    for doc in docs:
        rows = rows_by_doc.get(doc.doc_id, [])
        await writer.upsert_chunks(rows)
        await writer.delete_orphans(doc.doc_id, keep=len(rows))
        total += len(rows)

    stats = IngestStats(documents=len(docs), chunks=total, embedding_model=provider.model_name)
    logger.info(
        "ingest.complete",
        extra={
            "documents": stats.documents,
            "chunks": stats.chunks,
            "embedding_model": stats.embedding_model,
        },
    )
    return stats
