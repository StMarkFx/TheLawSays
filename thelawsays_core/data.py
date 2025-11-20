"""Dataclasses and typed helpers shared across the codebase."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class Chunk:
    id: int
    text: str
    source: str
    jurisdiction: str
    meta: Dict[str, str]
    score: Optional[float] = None

    @property
    def jurisdiction_lower(self) -> str:
        return self.jurisdiction.lower()
