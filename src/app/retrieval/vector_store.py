"""Abstract vector store so either backend (pgvector, FAISS) can serve a query."""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.retrieval.types import RetrievedChunk, SearchFilters


class VectorStore(ABC):
    """A dense semantic search backend over the chunk corpus."""

    @abstractmethod
    async def search(
        self,
        query_vector: list[float],
        top_k: int,
        filters: SearchFilters | None = None,
    ) -> list[RetrievedChunk]:
        """Return the ``top_k`` nearest chunks to ``query_vector``."""
