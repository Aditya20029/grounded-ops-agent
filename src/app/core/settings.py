"""Application configuration, loaded and validated once at startup.

All configuration comes from environment variables (or a local ``.env``). The
settings object fails fast with a clear message when something required is
missing or inconsistent, so the app never starts in a half-configured state.
No secrets live in code.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal, Self

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Known embedding models and their vector dimensions. The active model's
# dimension is stamped into the migration and the index metadata; query and
# index dimensions are asserted to match at retrieval time.
EMBEDDING_DIMS: dict[str, int] = {
    "BAAI/bge-small-en-v1.5": 384,
    "all-MiniLM-L6-v2": 384,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
}

EmbeddingProviderName = Literal["huggingface", "openai", "fake"]
LLMProviderName = Literal["anthropic", "openai", "fake"]
MCPTransport = Literal["stdio", "streamable-http"]


class Settings(BaseSettings):
    """Validated runtime configuration. See ``.env.example`` for documentation."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Database
    database_url: str
    db_pool_size: int = 10

    # Embeddings
    embedding_provider: EmbeddingProviderName = "huggingface"
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    # LLM provider
    llm_provider: LLMProviderName = "anthropic"
    llm_model: str = "claude-opus-4-7"
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None

    # Retrieval
    reranker_enabled: bool = False
    rrf_k: int = Field(default=60, ge=1)
    retrieval_top_k: int = Field(default=8, ge=1)

    # MCP analytics server
    mcp_transport: MCPTransport = "stdio"
    mcp_server_url: str = "http://127.0.0.1:8848/mcp"
    mcp_max_result_rows: int = Field(default=1000, ge=1)

    # Agent guardrails
    max_agent_steps: int = Field(default=6, ge=1)
    per_request_token_budget: int = Field(default=30_000, ge=1000)
    tool_timeout_seconds: int = Field(default=15, ge=1)
    max_tool_retries: int = Field(default=2, ge=0)
    max_output_tokens: int = Field(default=2000, ge=256)
    llm_thinking: bool = False

    # Evaluation
    eval_judge_model: str = "claude-opus-4-7"
    data_seed: int = 1337

    # Server / app
    app_host: str = "127.0.0.1"
    app_port: int = 8000
    log_level: str = "INFO"
    max_query_chars: int = Field(default=4000, ge=1)
    max_document_chars: int = Field(default=200_000, ge=1)

    @property
    def embedding_dim(self) -> int:
        """Vector dimension of the active embedding model."""
        return EMBEDDING_DIMS[self.embedding_model]

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        # The active embedding model must have a known dimension, since it is
        # baked into the migration and asserted at retrieval time.
        if self.embedding_model not in EMBEDDING_DIMS:
            known = ", ".join(sorted(EMBEDDING_DIMS))
            raise ValueError(
                f"Unknown EMBEDDING_MODEL '{self.embedding_model}'. "
                f"Known models (add others to EMBEDDING_DIMS): {known}"
            )

        # At least one usable LLM provider key for the selected provider.
        if self.llm_provider == "anthropic" and not self.anthropic_api_key:
            raise ValueError("LLM_PROVIDER=anthropic requires ANTHROPIC_API_KEY to be set.")
        if self.llm_provider == "openai" and not self.openai_api_key:
            raise ValueError("LLM_PROVIDER=openai requires OPENAI_API_KEY to be set.")

        # OpenAI embeddings require the OpenAI key regardless of LLM provider.
        if self.embedding_provider == "openai" and not self.openai_api_key:
            raise ValueError("EMBEDDING_PROVIDER=openai requires OPENAI_API_KEY to be set.")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton (validated on first access).

    Field values are populated from the environment / ``.env`` by
    pydantic-settings, so no constructor arguments are required.
    """
    return Settings()
