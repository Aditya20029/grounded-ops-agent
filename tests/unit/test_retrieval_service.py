"""Unit tests for the hybrid RetrievalService (fakes only, no DB)."""

from __future__ import annotations

import pytest

from app.core.errors import EmbeddingMismatchError
from app.retrieval.service import RetrievalService
from app.retrieval.types import RetrievedChunk
from tests.fakes.providers import FakeEmbeddingProvider


def _chunk(chunk_id: str, score: float, retriever: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id="d",
        source_type="ticket",
        title="t",
        content="body",
        char_start=0,
        char_end=1,
        score=score,
        retriever=retriever,
    )


class _FakePgStore:
    """Duck-typed PgVectorStore replacement for offline tests."""

    def __init__(
        self,
        model: str | None,
        dense: list[RetrievedChunk],
        keyword: list[RetrievedChunk],
    ) -> None:
        self._model = model
        self._dense = dense
        self._keyword = keyword

    async def stored_embedding_model(self) -> str | None:
        return self._model

    async def search(self, qv: list[float], k: int, filters: object = None) -> list[RetrievedChunk]:
        return self._dense[:k]

    async def keyword_search(self, q: str, k: int, filters: object = None) -> list[RetrievedChunk]:
        return self._keyword[:k]


@pytest.mark.unit
async def test_fuses_dense_and_keyword() -> None:
    provider = FakeEmbeddingProvider(8)
    dense = [_chunk("1", 0.9, "pgvector"), _chunk("2", 0.8, "pgvector")]
    keyword = [_chunk("2", 5.0, "keyword"), _chunk("3", 4.0, "keyword")]
    store = _FakePgStore(provider.model_name, dense, keyword)
    service = RetrievalService(provider=provider, pg_store=store, rrf_k=60)  # type: ignore[arg-type]

    results = await service.retrieve("q", top_k=3)
    assert {c.chunk_id for c in results} == {"1", "2", "3"}
    assert all(c.retriever == "rrf" for c in results)
    # "2" is in both lists -> highest fused score -> ranked first.
    assert results[0].chunk_id == "2"


@pytest.mark.unit
async def test_model_mismatch_raises() -> None:
    provider = FakeEmbeddingProvider(8)
    store = _FakePgStore("a-different-model", [], [])
    service = RetrievalService(provider=provider, pg_store=store, rrf_k=60)  # type: ignore[arg-type]
    with pytest.raises(EmbeddingMismatchError):
        await service.retrieve("q", top_k=3)


@pytest.mark.unit
async def test_empty_corpus_returns_nothing() -> None:
    provider = FakeEmbeddingProvider(8)
    store = _FakePgStore(None, [], [])
    service = RetrievalService(provider=provider, pg_store=store, rrf_k=60)  # type: ignore[arg-type]
    assert await service.retrieve("q", top_k=3) == []
