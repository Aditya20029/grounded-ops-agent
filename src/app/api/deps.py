"""Shared, cached API dependencies.

The embedding model is loaded once and reused across requests (loading a
sentence-transformers model per request would be prohibitively slow).
"""

from __future__ import annotations

from functools import lru_cache

from app.core.settings import get_settings
from app.llm.embeddings import EmbeddingProvider, get_embedding_provider
from app.retrieval.pgvector_store import PgVectorStore
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.service import RetrievalService


@lru_cache(maxsize=1)
def shared_embedding_provider() -> EmbeddingProvider:
    return get_embedding_provider(get_settings())


@lru_cache(maxsize=1)
def shared_retrieval_service() -> RetrievalService:
    settings = get_settings()
    return RetrievalService(
        provider=shared_embedding_provider(),
        pg_store=PgVectorStore(),
        rrf_k=settings.rrf_k,
        reranker=CrossEncoderReranker() if settings.reranker_enabled else None,
    )
