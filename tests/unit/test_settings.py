"""Unit tests for configuration loading and fail-fast validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.core.settings import Settings


@pytest.mark.unit
def test_defaults_load(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.embedding_dim == 384
    assert s.embedding_model == "BAAI/bge-small-en-v1.5"
    assert s.max_agent_steps == 6
    assert s.per_request_token_budget == 30_000


@pytest.mark.unit
def test_anthropic_provider_requires_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)  # type: ignore[call-arg]


@pytest.mark.unit
def test_openai_embedding_requires_openai_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")  # LLM provider satisfied
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-small")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)  # type: ignore[call-arg]


@pytest.mark.unit
def test_unknown_embedding_model_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setenv("EMBEDDING_MODEL", "not-a-real-model")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)  # type: ignore[call-arg]


@pytest.mark.unit
def test_openai_embedding_dim(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setenv("OPENAI_API_KEY", "ok")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-small")
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.embedding_dim == 1536
