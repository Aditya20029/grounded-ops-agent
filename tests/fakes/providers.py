"""Deterministic, offline test doubles for providers and the chunk store."""

from __future__ import annotations

from app.ingestion.store import ChunkRow
from app.llm.embeddings import HashingEmbeddingProvider


class FakeEmbeddingProvider(HashingEmbeddingProvider):
    """Deterministic embeddings for tests (no model download, no network)."""

    def __init__(self, dim: int = 384) -> None:
        super().__init__(dim)
        self.model_name = f"fake-embed-{dim}"


class InMemoryChunkWriter:
    """In-memory ChunkWriter for unit-testing idempotent ingestion."""

    def __init__(self) -> None:
        self._rows: dict[str, ChunkRow] = {}

    async def upsert_chunks(self, rows: list[ChunkRow]) -> None:
        for row in rows:
            self._rows[row.chunk_id] = row

    async def delete_orphans(self, doc_id: str, keep: int) -> int:
        victims = [
            cid
            for cid, row in self._rows.items()
            if row.doc_id == doc_id and row.chunk_index >= keep
        ]
        for cid in victims:
            del self._rows[cid]
        return len(victims)

    async def count(self) -> int:
        return len(self._rows)

    @property
    def rows(self) -> list[ChunkRow]:
        return list(self._rows.values())
