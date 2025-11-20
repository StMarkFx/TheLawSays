"""Feedback ingestion endpoint (placeholder)."""

import logging

from fastapi import APIRouter, status

from ..schemas import FeedbackRequest

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/feedback",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit interaction feedback",
)
async def submit_feedback(payload: FeedbackRequest) -> dict:
    # Future: forward to analytics pipeline. For now, just log.
    logger.info(
        "Feedback received: conversation=%s message=%s rating=%s comment=%s",
        payload.conversation_id,
        payload.message_id,
        payload.rating,
        payload.comment,
    )
    return {"status": "received"}
