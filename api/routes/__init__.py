"""API route registrations."""

from fastapi import APIRouter

from .chat import router as chat_router
from .feedback import router as feedback_router
from .health import router as health_router


def get_api_router() -> APIRouter:
    router = APIRouter()
    router.include_router(health_router, prefix="/v1", tags=["health"])
    router.include_router(chat_router, prefix="/v1", tags=["chat"])
    router.include_router(feedback_router, prefix="/v1", tags=["feedback"])
    return router
