"""pgvector-backed retrieval: the canonical, persistent, filterable store.

Dense search uses cosine distance over the HNSW index; keyword search uses the
Postgres full-text GIN index. Both support metadata filters (hybrid SQL +
vector). ``load_all`` and ``stored_embedding_model`` support building/validating
the derived FAISS index.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import Select, func, select

from app.core.db import session_scope
from app.db.models import Chunk
from app.retrieval.types import RetrievedChunk, SearchFilters
from app.retrieval.vector_store import VectorStore

_FIELDS = (
    Chunk.chunk_id,
    Chunk.doc_id,
    Chunk.source_type,
    Chunk.title,
    Chunk.content,
    Chunk.char_start,
    Chunk.char_end,
)


@dataclass(frozen=True)
class StoredChunk:
    """A chunk plus its embedding, used to (re)build the FAISS index."""

    chunk_id: str
    doc_id: str
    source_type: str
    title: str
    content: str
    char_start: int
    char_end: int
    embedding: list[float]


def _apply_filters(stmt: Select, filters: SearchFilters | None) -> Select:
    if filters is None:
        return stmt
    if filters.source_types is not None:
        stmt = stmt.where(Chunk.source_type.in_(filters.source_types))
    if filters.doc_ids is not None:
        stmt = stmt.where(Chunk.doc_id.in_(filters.doc_ids))
    return stmt


class PgVectorStore(VectorStore):
    """Canonical pgvector retrieval backend."""

    async def search(
        self,
        query_vector: list[float],
        top_k: int,
        filters: SearchFilters | None = None,
    ) -> list[RetrievedChunk]:
        distance = Chunk.embedding.cosine_distance(query_vector).label("distance")
        stmt: Select = select(*_FIELDS, distance).order_by(distance).limit(top_k)
        stmt = _apply_filters(stmt, filters)
        async with session_scope() as session:
            rows = (await session.execute(stmt)).all()
        return [
            RetrievedChunk(
                chunk_id=r.chunk_id,
                doc_id=r.doc_id,
                source_type=r.source_type,
                title=r.title,
                content=r.content,
                char_start=r.char_start,
                char_end=r.char_end,
                score=1.0 - float(r.distance),  # cosine similarity
                retriever="pgvector",
            )
            for r in rows
        ]

    async def keyword_search(
        self,
        query: str,
        top_k: int,
        filters: SearchFilters | None = None,
    ) -> list[RetrievedChunk]:
        tsvector = func.to_tsvector("english", Chunk.content)
        tsquery = func.plainto_tsquery("english", query)
        rank = func.ts_rank(tsvector, tsquery).label("rank")
        stmt: Select = (
            select(*_FIELDS, rank)
            .where(tsvector.op("@@")(tsquery))
            .order_by(rank.desc())
            .limit(top_k)
        )
        stmt = _apply_filters(stmt, filters)
        async with session_scope() as session:
            rows = (await session.execute(stmt)).all()
        return [
            RetrievedChunk(
                chunk_id=r.chunk_id,
                doc_id=r.doc_id,
                source_type=r.source_type,
                title=r.title,
                content=r.content,
                char_start=r.char_start,
                char_end=r.char_end,
                score=float(r.rank),
                retriever="keyword",
            )
            for r in rows
        ]

    async def stored_embedding_model(self) -> str | None:
        """The embedding model stamped on stored chunks (None if empty)."""
        async with session_scope() as session:
            result = await session.execute(select(Chunk.embedding_model).limit(1))
            return result.scalar_one_or_none()

    async def load_all(self) -> list[StoredChunk]:
        """Load every chunk with its embedding (for building the FAISS index)."""
        async with session_scope() as session:
            rows = (await session.execute(select(Chunk))).scalars().all()
        return [
            StoredChunk(
                chunk_id=c.chunk_id,
                doc_id=c.doc_id,
                source_type=c.source_type,
                title=c.title,
                content=c.content,
                char_start=c.char_start,
                char_end=c.char_end,
                embedding=[float(x) for x in c.embedding],
            )
            for c in rows
        ]
