"""The single retrieval implementation: hybrid dense + keyword, fused with RRF.

Both the agent orchestrator's initial seed retrieval and the MCP
``search_records`` tool call this, so retrieval behaviour never diverges. The
query embedding model is asserted against the model stamped on stored chunks
(the "one embedding model at a time" invariant) before any search runs.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from app.core.errors import EmbeddingMismatchError
from app.core.settings import Settings
from app.llm.embeddings import EmbeddingProvider, get_embedding_provider
from app.retrieval.faiss_store import FaissStore
from app.retrieval.pgvector_store import PgVectorStore
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.rrf import reciprocal_rank_fusion
from app.retrieval.types import RetrievedChunk, SearchFilters


@dataclass
class RetrievalService:
    """Hybrid retrieval over pgvector (or FAISS) + keyword, fused with RRF."""

    provider: EmbeddingProvider
    pg_store: PgVectorStore
    faiss_store: FaissStore | None = None
    rrf_k: int = 60
    reranker: CrossEncoderReranker | None = None

    async def _assert_model_matches(self) -> None:
        stored = await self.pg_store.stored_embedding_model()
        if stored is not None and stored != self.provider.model_name:
            raise EmbeddingMismatchError(
                f"Query model '{self.provider.model_name}' does not match the model "
                f"stored in the index ('{stored}'). Re-embed the corpus or switch back."
            )

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int = 8,
        filters: SearchFilters | None = None,
        use_faiss: bool = False,
    ) -> list[RetrievedChunk]:
        await self._assert_model_matches()
        query_vector = await asyncio.to_thread(self.provider.embed_query, query)

        pool = max(top_k * 4, 25)
        use_faiss_now = use_faiss and self.faiss_store is not None and self.faiss_store.ready
        dense_store = self.faiss_store if use_faiss_now else self.pg_store
        assert dense_store is not None  # narrowed by use_faiss_now

        dense = await dense_store.search(query_vector, pool, filters)
        keyword = await self.pg_store.keyword_search(query, pool, filters)
        fused = reciprocal_rank_fusion([dense, keyword], k=self.rrf_k)

        if self.reranker is not None:
            return await asyncio.to_thread(self.reranker.rerank, query, fused, top_k)
        return fused[:top_k]


def build_retrieval_service(
    settings: Settings, *, faiss_store: FaissStore | None = None
) -> RetrievalService:
    """Construct a RetrievalService from settings (reranker wired if enabled)."""
    reranker = CrossEncoderReranker() if settings.reranker_enabled else None
    return RetrievalService(
        provider=get_embedding_provider(settings),
        pg_store=PgVectorStore(),
        faiss_store=faiss_store,
        rrf_k=settings.rrf_k,
        reranker=reranker,
    )
