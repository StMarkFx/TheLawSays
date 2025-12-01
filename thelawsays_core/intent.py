"""Intent detection logic to decide when to trigger retrieval."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from openai import OpenAI

from .settings import DEFAULT_OPENAI_MODEL

EXPLICIT_RETRIEVAL_TRIGGERS = (
    "cite",
    "citation",
    "quote",
    "section",
    "law says",
    "according to",
    "legal basis",
)

CONVERSATIONAL_PATTERNS = [
    re.compile(r"^(hi|hello|hey|good (morning|afternoon|evening|day))$", re.I),
    re.compile(r"^(thanks?|thank you|appreciate it|thanks a lot|thanks so much)$", re.I),
    re.compile(r"^(okay?|ok|sure|yes|no|alright|cool|nice)$", re.I),
    re.compile(r"^(bye|goodbye|see you|ciao|later)$", re.I),
    re.compile(r"(who (are you|created you|made you|built you|is your creator))", re.I),
    re.compile(r"(what (can you do|do you do|are you|do you know))", re.I),
    re.compile(r"(help|menu|options|start|demo)$", re.I),
    re.compile(r"(tell me about yourself|about you)$", re.I),
]

LEGAL_SYNONYMS = (
    "arrest",
    "police",
    "court",
    "judge",
    "legal",
    "rights",
    "tenancy",
    "landlord",
    "tenant",
    "employment",
    "dismiss",
    "terminate",
    "fine",
    "penalty",
    "crime",
    "fraud",
    "contract",
    "business",
    "tax",
    "marriage",
    "divorce",
    "inheritance",
    "labour",
    "labor",
)


@dataclass
class IntentDecision:
    label: str
    retrieval_required: bool
    reason: str


class IntentDetector:
    """Decides whether a user query should trigger the RAG pipeline."""

    def __init__(
        self,
        client: Optional[OpenAI],
        model: str = DEFAULT_OPENAI_MODEL,
    ) -> None:
        self.client = client
        self.model = model
        self._cache: Dict[str, IntentDecision] = {}

    def classify(self, query: str, history: Optional[List[Dict[str, str]]] = None) -> IntentDecision:
        normalised = query.strip()
        if not normalised:
            return IntentDecision(label="conversational", retrieval_required=False, reason="empty-query")

        lowered = normalised.lower()

        for trigger in EXPLICIT_RETRIEVAL_TRIGGERS:
            if trigger in lowered:
                decision = IntentDecision("legal_lookup", True, f"explicit-trigger:{trigger}")
                self._cache[lowered] = decision
                return decision

        cached = self._cache.get(lowered)
        if cached:
            return cached

        if self._matches_conversational_patterns(lowered):
            decision = IntentDecision("conversational", False, "heuristic-conversational")
            self._cache[lowered] = decision
            return decision

        if self.client:
            try:
                decision = self._classify_with_openai(normalised, history)
                self._cache[lowered] = decision
                return decision
            except Exception:
                # fall back to heuristics if API fails
                pass

        # Heuristic fallback: treat anything moderately long or containing legal synonyms as legal lookup
        if len(lowered) < 15 and not any(term in lowered for term in LEGAL_SYNONYMS):
            decision = IntentDecision("conversational", False, "length-heuristic")
            self._cache[lowered] = decision
            return decision

        decision = IntentDecision("legal_lookup", True, "default-legal")
        self._cache[lowered] = decision
        return decision

    def _matches_conversational_patterns(self, lowered: str) -> bool:
        return any(pattern.search(lowered) for pattern in CONVERSATIONAL_PATTERNS)

    def _classify_with_openai(
        self, query: str, history: Optional[List[Dict[str, str]]] = None
    ) -> IntentDecision:
        history = history or []
        system_prompt = """
You are an expert intent classifier for a Nigerian legal chatbot. Your task is to classify the user's LATEST message based on the provided chat history.

- If the message seeks legal information, asks about laws, rights, legal procedures, or requires legal context to answer accurately, classify it as `legal_lookup`.
- If the message is a greeting, small talk, a thank you, or a meta-question about the chatbot itself, classify it as `conversational`.

Consider the full conversation context. A follow-up question like "what is the penalty for that?" after a legal discussion is a `legal_lookup`.

Respond with ONLY the label (`legal_lookup` or `conversational`).

--- EXAMPLES ---
- User: "hello there" -> conversational
- User: "who are you?" -> conversational
- User: "tell me something cool about the law" -> conversational
- User: "what does the law say about tenant rights in Lagos?" -> legal_lookup
- User: "thanks for the info" -> conversational
- User (after a discussion on rental law): "and what about eviction notices?" -> legal_lookup
- User: "can you cite the relevant section?" -> legal_lookup
- User: "what's an interesting legal case?" -> conversational
"""
        messages = [{"role": "system", "content": system_prompt.strip()}]
        messages.extend(history)
        messages.append({"role": "user", "content": query})

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            max_tokens=5,
            messages=messages,
        )
        label = (response.choices[0].message.content or "").strip().lower()
        if label not in {"legal_lookup", "conversational"}:
            label = "legal_lookup"
        retrieval_required = label == "legal_lookup"
        return IntentDecision(label=label, retrieval_required=retrieval_required, reason="openai-classifier")
