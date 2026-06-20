"""Shared pytest configuration.

Default environment variables are installed in ``pytest_configure`` (which runs
before any test module is imported) so the offline unit suite runs with no real
``.env`` and no provider keys. Integration tests that need a real database read
``DATABASE_URL`` (already defaulted to the docker-compose DSN).
"""

from __future__ import annotations

import os

import pytest

from app.core.settings import Settings, get_settings

_DEFAULT_ENV = {
    "DATABASE_URL": "postgresql+asyncpg://ops:ops@localhost:5432/ops",
    "EMBEDDING_PROVIDER": "huggingface",
    "EMBEDDING_MODEL": "BAAI/bge-small-en-v1.5",
    "LLM_PROVIDER": "anthropic",
    "LLM_MODEL": "claude-opus-4-7",
    "ANTHROPIC_API_KEY": "test-anthropic-key",
}


def pytest_configure(config: pytest.Config) -> None:
    for key, value in _DEFAULT_ENV.items():
        os.environ.setdefault(key, value)


@pytest.fixture
def settings() -> Settings:
    """Fresh settings built from the current environment."""
    get_settings.cache_clear()
    return get_settings()
