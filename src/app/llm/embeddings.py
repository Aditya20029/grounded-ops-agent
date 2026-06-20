"""Embedding providers behind one interface.

Implementations:
- ``HuggingFaceEmbeddingProvider`` (sentence-transformers, default, CPU, no key)
- ``OpenAIEmbeddingProvider`` (optional, needs a key)
- ``HashingEmbeddingProvider`` (deterministic, offline; powers keyless CI and
  the test fakes, and lets the demo run with no model download)

Heavy/optional dependencies (``sentence_transformers``, ``openai``) are imported
lazily inside the providers, so importing this module is cheap and never pulls
torch. Query and document embeddings always come from the same model (the "one
embedding model at a time" invariant); the active model id and dimension are
exposed so callers can stamp and assert them.
"""

from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod

import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.errors import EmbeddingMismatchError, ProviderError
from app.core.settings import Settings

# bge-* retrieval models recommend a short instruction prefix for queries.
_BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
_WORD = re.compile(r"[a-z0-9]+")


class EmbeddingProvider(ABC):
    """Common interface for embedding query and document text."""

    model_name: str
    dim: int

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of document chunks."""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""

    def _check_dim(self, vectors: list[list[float]]) -> None:
        if vectors and len(vectors[0]) != self.dim:
            raise EmbeddingMismatchError(
                f"Model '{self.model_name}' returned dimension {len(vectors[0])}, "
                f"expected {self.dim}."
            )


class HuggingFaceEmbeddingProvider(EmbeddingProvider):
    """sentence-transformers provider (model loaded lazily on first use)."""

    def __init__(self, model_name: str, dim: int) -> None:
        self.model_name = model_name
        self.dim = dim
        self._model: object | None = None

    def _load(self) -> object:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        model = self._load()
        vectors = model.encode(  # type: ignore[attr-defined]
            texts, normalize_embeddings=True, convert_to_numpy=True
        )
        out = [v.tolist() for v in vectors]
        self._check_dim(out)
        return out

    def embed_query(self, text: str) -> list[float]:
        prompt = _BGE_QUERY_INSTRUCTION + text if "bge" in self.model_name.lower() else text
        model = self._load()
        vector = model.encode(  # type: ignore[attr-defined]
            [prompt], normalize_embeddings=True, convert_to_numpy=True
        )[0]
        out: list[float] = vector.tolist()
        self._check_dim([out])
        return out


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embeddings provider with retry/backoff on transient failures."""

    def __init__(self, model_name: str, dim: int, api_key: str) -> None:
        self.model_name = model_name
        self.dim = dim
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key)

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, max=30))
    def _create(self, texts: list[str]) -> list[list[float]]:
        try:
            resp = self._client.embeddings.create(model=self.model_name, input=texts)
        except Exception as exc:  # surfaced after retries are exhausted
            raise ProviderError(f"OpenAI embeddings failed: {exc}") from exc
        return [list(d.embedding) for d in resp.data]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out = self._create(texts)
        self._check_dim(out)
        return out

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class HashingEmbeddingProvider(EmbeddingProvider):
    """Deterministic, offline bag-of-words hashing embeddings (L2-normalized).

    Uses a stable hash (not Python's salted ``hash``) so vectors are identical
    across processes and runs. Good enough to exercise retrieval mechanics
    end to end without any model download or API key.
    """

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.model_name = f"hashing-bow-{dim}"

    def _embed_one(self, text: str) -> list[float]:
        vec = np.zeros(self.dim, dtype=np.float32)
        for token in _WORD.findall(text.lower()):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(digest, "big") % self.dim
            vec[idx] += 1.0
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_one(text)


def get_embedding_provider(settings: Settings) -> EmbeddingProvider:
    """Construct the embedding provider selected by settings."""
    if settings.embedding_provider == "huggingface":
        return HuggingFaceEmbeddingProvider(settings.embedding_model, settings.embedding_dim)
    if settings.embedding_provider == "openai":
        if not settings.openai_api_key:  # defensive; settings already validates this
            raise ProviderError("OPENAI_API_KEY is required for OpenAI embeddings.")
        return OpenAIEmbeddingProvider(
            settings.embedding_model, settings.embedding_dim, settings.openai_api_key
        )
    return HashingEmbeddingProvider(settings.embedding_dim)
