"""Chat API route."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status

from ..dependencies import get_rag_service
from ..middleware import chat_rate_limit
from ..schemas import ChatRequest, ChatResponse
from ..services.rag import RagService

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
@chat_rate_limit()
async def chat_endpoint(
    payload: ChatRequest,
    service: RagService = Depends(get_rag_service),
) -> ChatResponse:
    try:
        return service.handle_chat(payload)
    except (FileNotFoundError, ValueError) as exc:
        logger.exception("Knowledge base unavailable")
        raise HTTPException(
            status_code=status.HTTP_424_FAILED_DEPENDENCY,
            detail=str(exc),
        ) from exc
    except Exception as exc:  # pragma: no cover - safeguard
        logger.exception("Chat endpoint failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to generate answer",
        ) from exc
