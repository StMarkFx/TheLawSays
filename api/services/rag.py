"""RAG service orchestrating intent detection, retrieval, and generation."""

from __future__ import annotations

from typing import List, Optional

from fastapi import HTTPException, status

from thelawsays_core import (
    IntentDecision,
    IntentDetector,
    KnowledgeBase,
    build_conversational_prompt,
    build_no_results_prompt,
    build_rag_prompt,
    load_knowledge_base,
)
from thelawsays_core.data import Chunk
from thelawsays_core.openai_utils import create_openai_client, generate_completion
from thelawsays_core.security import is_suspicious_input, validate_and_sanitize_output
from thelawsays_core.settings import DEFAULT_TOP_K
from ..config import Settings
from ..schemas import ChatRequest, ChatResponse, ChunkSchema


class RagService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.knowledge_base: KnowledgeBase = load_knowledge_base()
        self.client = create_openai_client(settings.openai_api_key)
        self.intent_detector = IntentDetector(client=self.client, model=settings.openai_model)

    def _moderate_input(self, text: str) -> None:
        if not self.settings.enable_moderation or not self.client:
            return
        try:
            response = self.client.moderations.create(input=text)
            if response.results[0].flagged:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Your request has been flagged as inappropriate.",
                )
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to moderate request.",
            ) from exc

    def handle_chat(self, payload: ChatRequest) -> ChatResponse:
        history = [msg.model_dump() for msg in payload.history]

        if is_suspicious_input(payload.message, history):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Your request contains suspicious input.",
            )
        self._moderate_input(payload.message)
        decision = self.intent_detector.classify(payload.message, history)
        top_k = payload.top_k or self.settings.retrieval_top_k or DEFAULT_TOP_K
        jurisdiction = payload.jurisdiction

        chunks = self._maybe_retrieve(decision, payload.message, top_k, jurisdiction)

        if not decision.retrieval_required:
            prompt = build_conversational_prompt(payload.message, history)
        elif not chunks:
            prompt = build_no_results_prompt(payload.message, history)
        else:
            prompt = build_rag_prompt(payload.message, chunks, jurisdiction, history)

        answer = generate_completion(
            client=self.client,
            prompt=prompt,
            model=self.settings.openai_model,
        )

        answer = validate_and_sanitize_output(answer)

        response_chunks = [ChunkSchema(**chunk.__dict__) for chunk in chunks] if chunks else []

        metadata = {
            "jurisdiction": jurisdiction,
            "intent_reason": decision.reason,
            "intent_label": decision.label,
        }

        return ChatResponse(
            answer=answer,
            chunks=response_chunks,
            retrieval_used=decision.retrieval_required and bool(chunks),
            metadata=metadata,
        )

    def _maybe_retrieve(
        self,
        decision: IntentDecision,
        query: str,
        top_k: int,
        jurisdiction: Optional[str],
    ) -> List[Chunk]:
        if not decision.retrieval_required:
            return []
        chunks = self.knowledge_base.hybrid_retrieve(
            query=query,
            top_k=top_k,
            jurisdiction=jurisdiction,
            alpha=self.settings.retrieval_alpha,
        )
        return chunks
