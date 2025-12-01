"""Shared utilities for TheLawSays RAG applications."""

from __future__ import annotations

try:  # pragma: no cover - environment-specific shim
    import huggingface_hub
except Exception:  # pragma: no cover - optional dependency
    huggingface_hub = None
else:  # pragma: no cover - execute only when library is present
    if huggingface_hub and not hasattr(huggingface_hub, "cached_download"):
        from huggingface_hub import hf_hub_download

        huggingface_hub.cached_download = hf_hub_download  # type: ignore[attr-defined]

from .intent import IntentDecision, IntentDetector
from .pipeline import KnowledgeBase, load_knowledge_base
from .prompts import build_conversational_prompt, build_no_results_prompt, build_rag_prompt

__all__ = [
    "IntentDecision",
    "IntentDetector",
    "KnowledgeBase",
    "load_knowledge_base",
    "build_conversational_prompt",
    "build_no_results_prompt",
    "build_rag_prompt",
]
