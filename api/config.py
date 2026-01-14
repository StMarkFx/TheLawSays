"""Configuration handling for the FastAPI backend."""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from thelawsays_core.settings import DEFAULT_ALPHA, DEFAULT_OPENAI_MODEL, DEFAULT_TOP_K


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    openai_api_key: Optional[str] = Field(
        default=None,
        validation_alias="OPENAI_API_KEY",
    )
    openai_model: str = Field(
        default=DEFAULT_OPENAI_MODEL,
        validation_alias="OPENAI_MODEL",
    )
    retrieval_top_k: int = Field(
        default=DEFAULT_TOP_K,
        validation_alias="RETRIEVAL_TOP_K",
    )
    retrieval_alpha: float = Field(
        default=DEFAULT_ALPHA,
        validation_alias="RETRIEVAL_ALPHA",
    )
    allow_origins: str = Field(
        default="http://localhost:3000",
        validation_alias="ALLOW_ORIGINS",
    )
    environment: str = Field(
        default="development",
        validation_alias="ENVIRONMENT",
    )
    enable_moderation: bool = Field(
        default=True,
        validation_alias="ENABLE_MODERATION",
    )
    # Rate limiting settings
    rate_limit_chat_requests: int = Field(
        default=10,
        validation_alias="RATE_LIMIT_CHAT_REQUESTS",
    )
    rate_limit_chat_window: int = Field(
        default=60,
        validation_alias="RATE_LIMIT_CHAT_WINDOW",
    )
    rate_limit_feedback_requests: int = Field(
        default=5,
        validation_alias="RATE_LIMIT_FEEDBACK_REQUESTS",
    )
    rate_limit_feedback_window: int = Field(
        default=3600,  # 1 hour in seconds
        validation_alias="RATE_LIMIT_FEEDBACK_WINDOW",
    )
    # Security settings
    trusted_hosts: str = Field(
        default="thelawsays.com,localhost",
        validation_alias="TRUSTED_HOSTS",
    )
    enable_https_redirect: bool = Field(
        default=True,
        validation_alias="ENABLE_HTTPS_REDIRECT",
    )


@lru_cache()
def get_settings() -> Settings:
    return Settings()
