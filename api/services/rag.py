"""RAG service orchestrating intent detection, retrieval, and generation."""

from __future__ import annotations

import uuid
from typing import List, Optional

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
from thelawsays_core.settings import DEFAULT_TOP_K

from ..config import Settings
from ..schemas import ChatRequest, ChatResponse, ChunkSchema


class RagService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.knowledge_base: KnowledgeBase = load_knowledge_base()
        self.client = create_openai_client(settings.openai_api_key)
        self.intent_detector = IntentDetector(client=self.client, model=settings.openai_model)

    def handle_chat(self, payload: ChatRequest) -> ChatResponse:
        decision = self.intent_detector.classify(payload.message)
        top_k = payload.top_k or self.settings.retrieval_top_k or DEFAULT_TOP_K
        jurisdiction = payload.jurisdiction

        chunks = self._maybe_retrieve(decision, payload.message, top_k, jurisdiction)

        if not decision.retrieval_required:
            prompt = build_conversational_prompt(payload.message)
        elif not chunks:
            prompt = build_no_results_prompt(payload.message)
        else:
            prompt = build_rag_prompt(payload.message, chunks, jurisdiction)

        answer = generate_completion(
            client=self.client,
            prompt=prompt,
            model=self.settings.openai_model,
        )

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
            conversation_id=str(uuid.uuid4()),
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
