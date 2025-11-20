"""Configuration handling for the FastAPI backend."""

from __future__ import annotations

from functools import lru_cache
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings

from thelawsays_core.settings import DEFAULT_ALPHA, DEFAULT_OPENAI_MODEL, DEFAULT_TOP_K


class Settings(BaseSettings):
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    openai_model: str = Field(default=DEFAULT_OPENAI_MODEL, env="OPENAI_MODEL")
    retrieval_top_k: int = Field(default=DEFAULT_TOP_K, env="RETRIEVAL_TOP_K")
    retrieval_alpha: float = Field(default=DEFAULT_ALPHA, env="RETRIEVAL_ALPHA")
    allow_origins: str = Field(default="http://localhost:3000", env="ALLOW_ORIGINS")
    environment: str = Field(default="development", env="ENVIRONMENT")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
