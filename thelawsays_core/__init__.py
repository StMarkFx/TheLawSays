"""Shared utilities for TheLawSays RAG applications."""

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
