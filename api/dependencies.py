"""FastAPI dependency wiring."""

from functools import lru_cache

from .config import Settings, get_settings
from .services.rag import RagService


@lru_cache()
def get_rag_service() -> RagService:
    settings: Settings = get_settings()
    return RagService(settings=settings)
