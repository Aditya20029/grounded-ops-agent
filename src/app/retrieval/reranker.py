"""Optional cross-encoder reranker (off by default; slow on CPU).

Enabled via ``RERANKER_ENABLED``. Loads ``BAAI/bge-reranker-base`` lazily so the
dependency is only paid when actually used.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from app.retrieval.types import RetrievedChunk


class CrossEncoderReranker:
    """Re-score (query, chunk) pairs with a cross-encoder and re-order."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-base") -> None:
        self.model_name = model_name
        self._model: Any | None = None

    def _load(self) -> Any:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        if not chunks:
            return chunks
        model = self._load()
        scores = model.predict([(query, c.content) for c in chunks])
        ranked = sorted(zip(chunks, scores, strict=True), key=lambda pair: pair[1], reverse=True)
        out = [replace(chunk, score=float(score), retriever="reranker") for chunk, score in ranked]
        return out[:top_k] if top_k is not None else out
