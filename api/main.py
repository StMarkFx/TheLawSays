"""FastAPI entrypoint for TheLawSays backend."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .dependencies import get_rag_service
from .routes import get_api_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info("Starting TheLawSays backend in %s mode", settings.environment)
    # Warm up resources so first request is fast (skip during tests to avoid heavy loading)
    if settings.environment != "test":
        get_rag_service()
    yield
    logger.info("Shutting down TheLawSays backend")


def create_app() -> FastAPI:
    settings = get_settings()
    # Parse allowed origins from comma-separated string
    allowed_origins = [origin.strip() for origin in settings.allow_origins.split(",") if origin.strip()]

    app = FastAPI(title="TheLawSays API", version="1.0.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(get_api_router())
    return app


app = create_app()
