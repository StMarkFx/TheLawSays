"""Pydantic schemas for API input/output."""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChunkSchema(BaseModel):
    id: int
    source: str
    jurisdiction: str
    text: str
    score: Optional[float] = None
    meta: dict = Field(default_factory=dict)


class ChatRequest(BaseModel):
    message: str
    history: List[Message] = Field(default_factory=list)
    jurisdiction: Optional[str] = None
    top_k: Optional[int] = None

    @model_validator(mode="before")
    @classmethod
    def _support_conversation_field(cls, values: dict) -> dict:
        history = values.get("history")
        conversation = values.pop("conversation", None)
        if history is None and conversation is not None:
            values["history"] = conversation
        return values

    @field_validator("jurisdiction")
    @classmethod
    def empty_to_none(cls, value: Optional[str]) -> Optional[str]:
        if value and not value.strip():
            return None
        return value


class ChatResponse(BaseModel):
    answer: str
    chunks: List[ChunkSchema] = Field(default_factory=list)
    retrieval_used: bool
    metadata: dict = Field(default_factory=dict)


class FeedbackRequest(BaseModel):
    conversation_id: str
    message_id: Optional[str] = None
    rating: Literal["thumbs_up", "thumbs_down"]
    comment: Optional[str] = None
