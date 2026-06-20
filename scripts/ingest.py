"""Chunk, embed, and upsert the seed corpus into pgvector (idempotent).

Usage:
    python scripts/ingest.py     # after `make seed`

Re-running upserts on chunk_id and prunes orphans, so it never duplicates.
The FAISS index is built from pgvector in Phase 4.
"""

from __future__ import annotations

import asyncio

from app.core.db import dispose_engine
from app.core.logging import configure_logging, get_logger
from app.core.settings import get_settings
from app.ingestion.indexer import ingest_documents
from app.ingestion.loaders import load_corpus
from app.ingestion.store import PgChunkWriter
from app.llm.embeddings import get_embedding_provider

logger = get_logger("ingest")


async def _main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)

    docs = load_corpus()
    if not docs:
        logger.warning(
            "ingest.no_documents",
            extra={"hint": "run `python scripts/seed_db.py` first to generate data/seed/"},
        )
        return

    provider = get_embedding_provider(settings)
    writer = PgChunkWriter()
    try:
        stats = await ingest_documents(docs, provider, writer)
        total = await writer.count()
        logger.info(
            "ingest.done",
            extra={
                "documents": stats.documents,
                "chunks_written": stats.chunks,
                "chunks_in_db": total,
                "embedding_model": stats.embedding_model,
            },
        )
    finally:
        await dispose_engine()


def main() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    main()
