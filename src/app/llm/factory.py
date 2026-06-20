"""Select and construct the configured LLM provider."""

from __future__ import annotations

from app.core.errors import ConfigError
from app.core.settings import Settings
from app.llm.base import LLMProvider


def get_llm_provider(settings: Settings) -> LLMProvider:
    """Build the LLM provider selected by settings (lazy provider imports)."""
    if settings.llm_provider == "anthropic":
        if not settings.anthropic_api_key:
            raise ConfigError("ANTHROPIC_API_KEY is required for the Anthropic provider.")
        from app.llm.anthropic_provider import AnthropicProvider

        return AnthropicProvider(
            settings.llm_model, settings.anthropic_api_key, thinking=settings.llm_thinking
        )
    if settings.llm_provider == "openai":
        if not settings.openai_api_key:
            raise ConfigError("OPENAI_API_KEY is required for the OpenAI provider.")
        from app.llm.openai_provider import OpenAIProvider

        return OpenAIProvider(settings.llm_model, settings.openai_api_key)

    # "fake": deterministic, offline provider so the full pipeline runs keyless.
    from app.llm.echo_provider import EchoLLMProvider

    return EchoLLMProvider()
