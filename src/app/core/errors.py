"""Typed application errors.

Every external failure is mapped to one of these so the API layer can return a
structured error response with a stable ``code`` instead of leaking provider or
driver internals.
"""

from __future__ import annotations


class AppError(Exception):
    """Base class for all application errors.

    Attributes:
        message: Human-readable description (safe to surface to clients).
        code: Stable machine-readable code used in structured error responses.
    """

    code: str = "app_error"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class ConfigError(AppError):
    """Invalid or missing configuration discovered at startup."""

    code = "config_error"


class ValidationAppError(AppError):
    """Caller-supplied input failed validation (e.g. exceeds size limits)."""

    code = "validation_error"


class NotFoundError(AppError):
    """A requested record or resource does not exist."""

    code = "not_found"


class RetrievalError(AppError):
    """Retrieval or vector-store operation failed."""

    code = "retrieval_error"


class EmbeddingMismatchError(RetrievalError):
    """Query embedding model/dimension does not match the stored index."""

    code = "embedding_mismatch"


class ProviderError(AppError):
    """An LLM or embedding provider call failed."""

    code = "provider_error"


class ToolError(AppError):
    """An MCP tool call failed, timed out, or was rejected by the whitelist."""

    code = "tool_error"


class GuardrailError(AppError):
    """An agent guardrail (step cap, token budget, cycle) halted execution."""

    code = "guardrail_error"
