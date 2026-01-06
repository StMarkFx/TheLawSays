"""Centralised configuration constants for shared RAG components."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_PATH = BASE_DIR / "data" / "documents.json"
FAISS_PATH = BASE_DIR / "data" / "indices" / "legal_index.faiss"
BM25_PATH = BASE_DIR / "data" / "indices" / "bm25_index.pkl"

DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_TOP_K = 5
DEFAULT_ALPHA = 0.65
