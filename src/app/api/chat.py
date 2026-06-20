"""Chat endpoint: the agent loop streamed over Server-Sent Events.

A multi-step agent loop cannot token-stream during planning, so the SSE protocol
carries lifecycle events: ``status``, ``tool_call``, ``tool_result``, ``token``
(final-answer chunks), ``sources``, ``trace``, ``done``, and ``error``.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import asdict, is_dataclass
from typing import Any

from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse

from app.agent.orchestrator import Orchestrator, mcp_executor
from app.api.deps import shared_retrieval_service
from app.core.errors import ValidationAppError
from app.core.logging import get_logger
from app.core.settings import get_settings
from app.llm.factory import get_llm_provider
from app.mcp_client.client import MCPClient
from app.retrieval.types import SearchFilters
from app.schemas.chat import ChatRequest

router = APIRouter(tags=["agent"])
logger = get_logger(__name__)


def _default(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    raise TypeError(f"not JSON serializable: {type(obj)!r}")


@router.post("/chat")
async def chat(req: ChatRequest) -> EventSourceResponse:
    settings = get_settings()
    if len(req.question) > settings.max_query_chars:
        raise ValidationAppError(f"question exceeds {settings.max_query_chars} characters")
    filters = SearchFilters(source_types=tuple(req.source_types)) if req.source_types else None

    async def event_stream() -> AsyncIterator[dict[str, str]]:
        try:
            async with MCPClient(settings) as client:
                orch = Orchestrator(
                    llm=get_llm_provider(settings),
                    retrieval=shared_retrieval_service(),
                    executor=mcp_executor(client),
                    settings=settings,
                )
                async for event in orch.events(req.question, top_k=req.top_k, filters=filters):
                    yield {"event": event.type, "data": json.dumps(event.data, default=_default)}
        except Exception as exc:  # surface as a structured error event, then end
            logger.exception("chat.error")
            yield {
                "event": "error",
                "data": json.dumps({"error": "agent_error", "detail": str(exc)}),
            }

    return EventSourceResponse(event_stream())
