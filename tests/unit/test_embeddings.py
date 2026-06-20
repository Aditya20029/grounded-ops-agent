"""Unit tests for embedding providers and the model/dimension assertion."""

from __future__ import annotations

import math

import pytest

from app.core.errors import EmbeddingMismatchError
from app.core.settings import Settings
from app.llm.embeddings import HashingEmbeddingProvider, get_embedding_provider


@pytest.mark.unit
def test_hashing_is_deterministic_and_normalized() -> None:
    provider = HashingEmbeddingProvider(384)
    v1 = provider.embed_query("payments service latency spike")
    v2 = provider.embed_query("payments service latency spike")
    assert v1 == v2
    assert len(v1) == 384
    norm = math.sqrt(sum(x * x for x in v1))
    assert math.isclose(norm, 1.0, rel_tol=1e-5)


@pytest.mark.unit
def test_hashing_distinguishes_text() -> None:
    provider = HashingEmbeddingProvider(256)
    assert provider.embed_query("alpha tokens") != provider.embed_query("beta tokens")


@pytest.mark.unit
def test_embed_documents_shape() -> None:
    provider = HashingEmbeddingProvider(128)
    out = provider.embed_documents(["a b c", "d e f", ""])
    assert len(out) == 3
    assert all(len(v) == 128 for v in out)


@pytest.mark.unit
def test_dimension_mismatch_raises() -> None:
    provider = HashingEmbeddingProvider(384)
    with pytest.raises(EmbeddingMismatchError):
        provider._check_dim([[0.0] * 5])


@pytest.mark.unit
def test_factory_returns_hashing_for_fake(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "fake")
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    provider = get_embedding_provider(settings)
    assert isinstance(provider, HashingEmbeddingProvider)
    assert provider.dim == settings.embedding_dim
