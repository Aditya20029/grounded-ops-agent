"""Unit tests for the FAISS store: id mapping, persistence, filters.

Skipped when faiss is not installed; runs in CI (which installs faiss-cpu).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("faiss")

from app.retrieval.faiss_store import FaissStore
from app.retrieval.pgvector_store import StoredChunk
from app.retrieval.types import SearchFilters


def _norm(vec: list[float]) -> list[float]:
    arr = np.asarray(vec, dtype="float32")
    return (arr / np.linalg.norm(arr)).tolist()


def _chunk(chunk_id: str, vec: list[float], source: str = "ticket") -> StoredChunk:
    return StoredChunk(
        chunk_id=chunk_id,
        doc_id=chunk_id,
        source_type=source,
        title=f"title {chunk_id}",
        content=f"content {chunk_id}",
        char_start=0,
        char_end=1,
        embedding=_norm(vec),
    )


@pytest.mark.unit
async def test_build_and_search_returns_nearest(tmp_path: Path) -> None:
    chunks = [
        _chunk("a", [1, 0, 0, 0]),
        _chunk("b", [0, 1, 0, 0]),
        _chunk("c", [0, 0, 1, 0]),
    ]
    store = FaissStore(4, index_path=tmp_path / "idx.faiss", index_type="flat")
    n = await store.build(chunks, "fake-model")
    assert n == 3
    assert store.ready and store.size == 3

    results = await store.search(_norm([1, 0, 0, 0]), top_k=2)
    assert results[0].chunk_id == "a"
    assert results[0].retriever == "faiss"


@pytest.mark.unit
async def test_persistence_round_trip(tmp_path: Path) -> None:
    chunks = [_chunk("a", [1, 0, 0, 0]), _chunk("b", [0, 1, 0, 0])]
    writer = FaissStore(4, index_path=tmp_path / "idx.faiss")
    await writer.build(chunks, "model-x")

    reader = FaissStore(4, index_path=tmp_path / "idx.faiss")
    assert reader._load() is True
    assert reader.size == 2
    results = await reader.search(_norm([0, 1, 0, 0]), top_k=1)
    assert results[0].chunk_id == "b"


@pytest.mark.unit
async def test_filters_restrict_results(tmp_path: Path) -> None:
    chunks = [
        _chunk("a", [1, 0, 0, 0], source="ticket"),
        _chunk("b", [0.95, 0.05, 0, 0], source="runbook"),
    ]
    store = FaissStore(4, index_path=tmp_path / "idx.faiss")
    await store.build(chunks, "model-x")
    results = await store.search(
        _norm([1, 0, 0, 0]), top_k=5, filters=SearchFilters(source_types=("runbook",))
    )
    assert {r.chunk_id for r in results} == {"b"}
