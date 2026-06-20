"""Chunk persistence behind a small writer interface.

Decoupling the writer from the DB lets ingestion be unit-tested for idempotency
with an in-memory writer (no Postgres), while production uses ``PgChunkWriter``
which upserts on ``chunk_id`` and prunes orphaned chunks.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, runtime_checkable

from sqlalchemy import delete, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.core.db import session_scope
from app.db.models import Chunk


@dataclass(frozen=True)
class ChunkRow:
    """A fully-formed chunk ready to persist."""

    chunk_id: str
    doc_id: str
    source_type: str
    title: str
    chunk_index: int
    char_start: int
    char_end: int
    content: str
    embedding: list[float]
    embedding_model: str
    created_at: datetime


@runtime_checkable
class ChunkWriter(Protocol):
    """Upsert chunks and prune orphans for a document (idempotent ingestion)."""

    async def upsert_chunks(self, rows: list[ChunkRow]) -> None: ...

    async def delete_orphans(self, doc_id: str, keep: int) -> int: ...

    async def count(self) -> int: ...


def _row_dict(row: ChunkRow) -> dict[str, object]:
    return {
        "chunk_id": row.chunk_id,
        "doc_id": row.doc_id,
        "source_type": row.source_type,
        "title": row.title,
        "chunk_index": row.chunk_index,
        "char_start": row.char_start,
        "char_end": row.char_end,
        "content": row.content,
        "embedding": row.embedding,
        "embedding_model": row.embedding_model,
        "created_at": row.created_at,
    }


class PgChunkWriter:
    """pgvector-backed writer: upsert on ``chunk_id``, prune orphans by index."""

    async def upsert_chunks(self, rows: list[ChunkRow]) -> None:
        if not rows:
            return
        async with session_scope() as session:
            stmt = pg_insert(Chunk).values([_row_dict(r) for r in rows])
            # created_at is preserved on conflict; everything else is refreshed.
            update_cols = {
                c.name: stmt.excluded[c.name]
                for c in Chunk.__table__.columns
                if c.name not in ("chunk_id", "created_at")
            }
            stmt = stmt.on_conflict_do_update(index_elements=["chunk_id"], set_=update_cols)
            await session.execute(stmt)

    async def delete_orphans(self, doc_id: str, keep: int) -> int:
        async with session_scope() as session:
            result = await session.execute(
                delete(Chunk).where(Chunk.doc_id == doc_id, Chunk.chunk_index >= keep)
            )
            return int(getattr(result, "rowcount", 0) or 0)

    async def count(self) -> int:
        async with session_scope() as session:
            result = await session.execute(select(func.count()).select_from(Chunk))
            return int(result.scalar_one())
