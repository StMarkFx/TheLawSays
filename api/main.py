"""FastAPI entrypoint for TheLawSays backend."""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .dependencies import get_rag_service
from .routes import get_api_router

# Configure logging to output to stdout (for Railway logs)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown events."""
    settings = get_settings()
    logger.info("Starting TheLawSays backend in %s mode", settings.environment)
    
    # Warm up resources so first request is fast (skip during tests to avoid heavy loading)
    if settings.environment != "test":
        try:
            logger.info("Initializing RAG service...")
            get_rag_service()
            logger.info("RAG service initialized successfully")
        except FileNotFoundError as exc:
            logger.error(
                "Knowledge base files not found: %s. "
                "Please ensure documents.json, legal_index.faiss, and bm25_index.pkl exist in the repository root.",
                exc
            )
            # Continue startup - service will fail on first request with clear error
        except MemoryError:
            logger.error(
                "Out of memory while loading knowledge base. "
                "Consider lazy-loading or increasing available memory."
            )
            # Continue startup - service will fail on first request
        except Exception as exc:
            logger.exception("Failed to initialize RAG service during startup: %s", exc)
            # Continue startup - service will fail on first request with clear error
    
    yield
    
    logger.info("Shutting down TheLawSays backend")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
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
