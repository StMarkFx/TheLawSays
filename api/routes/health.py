"""Healthcheck endpoints."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/health", summary="Service healthcheck")
async def health() -> dict:
    return {"status": "ok"}
