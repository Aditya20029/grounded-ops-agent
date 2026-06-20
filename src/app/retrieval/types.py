"""Shared retrieval data types."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk returned by a retriever, with its score and provenance."""

    chunk_id: str
    doc_id: str
    source_type: str
    title: str
    content: str
    char_start: int
    char_end: int
    score: float
    retriever: str  # "pgvector" | "faiss" | "keyword" | "rrf" | "reranker"


@dataclass(frozen=True)
class SearchFilters:
    """Optional metadata filters applied to retrieval (hybrid SQL + vector)."""

    source_types: tuple[str, ...] | None = None
    doc_ids: tuple[str, ...] | None = None

    def allows(self, source_type: str, doc_id: str) -> bool:
        """In-memory filter predicate (used by the FAISS path)."""
        if self.source_types is not None and source_type not in self.source_types:
            return False
        return self.doc_ids is None or doc_id in self.doc_ids

    @property
    def is_empty(self) -> bool:
        return self.source_types is None and self.doc_ids is None
