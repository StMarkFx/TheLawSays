"""Helpers for interacting with the OpenAI Python SDK."""

from __future__ import annotations

from typing import Optional

from openai import OpenAI

from .settings import DEFAULT_OPENAI_MODEL


def create_openai_client(api_key: Optional[str]) -> Optional[OpenAI]:
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


def generate_completion(
    client: Optional[OpenAI],
    prompt: str,
    model: str = DEFAULT_OPENAI_MODEL,
    max_tokens: int = 500,
    temperature: float = 0.2,
) -> str:
    if not client:
        return "Connect an OpenAI API key to enable AI-generated answers."
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:  # pragma: no cover - defensive user facing message
        return f"OpenAI error: {exc}"
