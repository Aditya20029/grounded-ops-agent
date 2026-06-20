"""FAISS-backed retrieval: a derived, in-memory, low-latency index.

FAISS stores vectors only, so a side mapping from FAISS internal id to
``chunk_id`` (plus enough chunk metadata to return results without touching the
database) is persisted alongside the index. On startup the index and mapping are
loaded; if the files are missing or the stamped embedding model differs, the
index is rebuilt from pgvector (the source of truth). Concurrent rebuilds are
guarded by a lock; concurrent reads are fine.

Embedding vectors are L2-normalized by the providers, so inner-product search
(``METRIC_INNER_PRODUCT``) is equivalent to cosine similarity. Flat, IVF, and
HNSW index types are supported so the benchmark can compare them.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from app.core.logging import get_logger
from app.retrieval.types import RetrievedChunk, SearchFilters
from app.retrieval.vector_store import VectorStore

if TYPE_CHECKING:
    from app.retrieval.pgvector_store import PgVectorStore, StoredChunk

logger = get_logger(__name__)

DEFAULT_INDEX_DIR = Path(__file__).resolve().parents[3] / "data" / "index"
_META_KEYS = ("doc_id", "source_type", "title", "content", "char_start", "char_end")


class FaissStore(VectorStore):
    """In-memory FAISS index with a persisted chunk-id mapping."""

    def __init__(
        self,
        dim: int,
        *,
        index_path: Path | None = None,
        index_type: str = "flat",
    ) -> None:
        self.dim = dim
        self.index_type = index_type
        self.index_path = index_path or (DEFAULT_INDEX_DIR / "chunks.faiss")
        self.map_path = self.index_path.parent / (self.index_path.name + ".map.json")
        self._index: Any | None = None
        self._id_map: list[str] = []
        self._meta: dict[str, dict[str, Any]] = {}
        self._embedding_model: str | None = None
        self._lock = asyncio.Lock()

    @property
    def ready(self) -> bool:
        return self._index is not None

    @property
    def size(self) -> int:
        return len(self._id_map)

    # -- building -------------------------------------------------------------

    def _new_index(self, n: int) -> Any:
        import faiss

        index: Any
        if self.index_type == "hnsw":
            index = faiss.IndexHNSWFlat(self.dim, 32, faiss.METRIC_INNER_PRODUCT)
        elif self.index_type == "ivf":
            quantizer = faiss.IndexFlatIP(self.dim)
            nlist = max(1, min(100, n // 39 or 1))
            index = faiss.IndexIVFFlat(quantizer, self.dim, nlist, faiss.METRIC_INNER_PRODUCT)
        else:
            index = faiss.IndexFlatIP(self.dim)
        return index

    def _build_sync(self, vectors: np.ndarray) -> Any:
        index = self._new_index(len(vectors))
        if not index.is_trained:
            index.train(vectors)
        index.add(vectors)
        return index

    async def build(self, chunks: list[StoredChunk], embedding_model: str) -> int:
        """Build the index from chunks and persist it. Returns the chunk count."""
        async with self._lock:
            if chunks:
                vectors = np.asarray([c.embedding for c in chunks], dtype="float32")
                self._index = await asyncio.to_thread(self._build_sync, vectors)
            else:
                import faiss

                self._index = faiss.IndexFlatIP(self.dim)
            self._id_map = [c.chunk_id for c in chunks]
            self._meta = {c.chunk_id: {k: getattr(c, k) for k in _META_KEYS} for c in chunks}
            self._embedding_model = embedding_model
            await asyncio.to_thread(self._persist)
            logger.info(
                "faiss.built",
                extra={
                    "chunks": len(chunks),
                    "index_type": self.index_type,
                    "embedding_model": embedding_model,
                },
            )
            return len(chunks)

    async def rebuild_from_pg(self, pg_store: PgVectorStore, embedding_model: str) -> int:
        """Rebuild the index from pgvector (the source of truth)."""
        chunks = await pg_store.load_all()
        return await self.build(chunks, embedding_model)

    async def ensure(self, pg_store: PgVectorStore, embedding_model: str) -> None:
        """Load from disk; rebuild from pgvector if missing or model-mismatched."""
        if self._load():
            # _load only succeeds when the on-disk dimension matches, so only the
            # stamped embedding model can still be stale here.
            if self._embedding_model == embedding_model:
                return
            logger.info(
                "faiss.stale_rebuild",
                extra={"on_disk": self._embedding_model, "active": embedding_model},
            )
        await self.rebuild_from_pg(pg_store, embedding_model)

    # -- persistence ----------------------------------------------------------

    def _persist(self) -> None:
        import faiss

        assert self._index is not None
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(self.index_path))
        self.map_path.write_text(
            json.dumps(
                {
                    "embedding_model": self._embedding_model,
                    "dim": self.dim,
                    "index_type": self.index_type,
                    "id_map": self._id_map,
                    "meta": self._meta,
                },
            ),
            encoding="utf-8",
        )

    def _load(self) -> bool:
        if not self.index_path.exists() or not self.map_path.exists():
            return False
        import faiss

        sidecar = json.loads(self.map_path.read_text(encoding="utf-8"))
        if sidecar.get("dim") != self.dim:
            return False
        self._index = faiss.read_index(str(self.index_path))
        self._id_map = sidecar["id_map"]
        self._meta = sidecar["meta"]
        self._embedding_model = sidecar.get("embedding_model")
        self.index_type = sidecar.get("index_type", self.index_type)
        return True

    # -- search ---------------------------------------------------------------

    async def search(
        self,
        query_vector: list[float],
        top_k: int,
        filters: SearchFilters | None = None,
    ) -> list[RetrievedChunk]:
        index = self._index
        if index is None or self.size == 0:
            return []
        query = np.asarray([query_vector], dtype="float32")
        # Over-fetch when filtering so enough survive the in-memory predicate.
        fetch = top_k * 5 if filters and not filters.is_empty else top_k
        fetch = min(fetch, self.size)
        scores, indices = await asyncio.to_thread(index.search, query, fetch)

        out: list[RetrievedChunk] = []
        for score, idx in zip(scores[0], indices[0], strict=True):
            if idx < 0:
                continue
            chunk_id = self._id_map[int(idx)]
            meta = self._meta[chunk_id]
            if filters is not None and not filters.allows(meta["source_type"], meta["doc_id"]):
                continue
            out.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    doc_id=meta["doc_id"],
                    source_type=meta["source_type"],
                    title=meta["title"],
                    content=meta["content"],
                    char_start=meta["char_start"],
                    char_end=meta["char_end"],
                    score=float(score),
                    retriever="faiss",
                )
            )
            if len(out) >= top_k:
                break
        return out
