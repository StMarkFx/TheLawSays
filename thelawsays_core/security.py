from __future__ import annotations

import re
from typing import List, Dict

# Heuristic-based flags for suspicious input
# These are not exhaustive but cover common patterns.
SUSPICIOUS_KEYWORDS = [
    "ignore previous instructions",
    "act as a different persona",
    "system override",
    "jailbreak",
    "prompt injection",
    "confidential information",
    "give me your instructions",
    "what are your instructions",
    "what is your system prompt",
]

# Max input length to prevent overly complex prompts
MAX_INPUT_LENGTH = 2048


def is_suspicious_input(query: str, history: List[Dict[str, str]] = None) -> bool:
    """
    Checks user input for suspicious patterns that might indicate prompt injection
    or jailbreaking attempts.
    """
    normalized_query = query.lower().strip()

    # 1. Check for excessive length
    if len(query) > MAX_INPUT_LENGTH:
        return True

    # 2. Check for suspicious keywords
    for keyword in SUSPICIOUS_KEYWORDS:
        if keyword in normalized_query:
            return True

    # 3. Check for unusual character repetition (e.g., trying to confuse the model)
    if re.search(r'(.)\1{10,}', query):  # More than 10 repeated characters
        return True

    return False


def validate_and_sanitize_output(response_text: str) -> str:
    """
    Checks the LLM's output for adherence to persona and safety guidelines.
    If a violation is detected, it replaces the response with a safe message.
    """
    normalized_response = response_text.lower().strip()

    # Keywords that suggest the bot is giving legal advice or has broken character
    violation_keywords = [
        "as a lawyer",
        "i advise you to",
        "legal counsel",
        "this is legal advice",
        "i can now",  # Common jailbreak confirmation
    ]

    for keyword in violation_keywords:
        if keyword in normalized_response:
            return "I am a legal research assistant and cannot provide legal advice. Please consult with a qualified lawyer for your specific situation."

    # If no violations, return the original response
    return response_text
