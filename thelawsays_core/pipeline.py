"""Knowledge base loading and retrieval helpers reused by Streamlit and FastAPI."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from dataclasses import replace
from typing import Dict, Iterable, List, Optional, Tuple

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from .data import Chunk
from .settings import BM25_PATH, DOCS_PATH, FAISS_PATH, DEFAULT_ALPHA


class KnowledgeBase:
    """Container around the hybrid retrieval resources."""

    def __init__(
        self,
        docs: List[Chunk],
        index: faiss.Index,
        embedder: SentenceTransformer,
        bm25: BM25Okapi,
    ) -> None:
        self.docs = docs
        self.index = index
        self.embedder = embedder
        self.bm25 = bm25

    def hybrid_retrieve(
        self,
        query: str,
        top_k: int,
        jurisdiction: Optional[str],
        alpha: float = DEFAULT_ALPHA,
    ) -> List[Chunk]:
        candidate_scores: Dict[int, float] = {}

        def jurisdiction_matches(chunk: Chunk, selected: Optional[str]) -> bool:
            if selected is None:
                return True
            return chunk.jurisdiction_lower == selected.lower()

        pool = min(top_k * 5, len(self.docs))
        if pool:
            query_vec = self.embedder.encode([query], normalize_embeddings=True)
            query_vec = np.asarray(query_vec, dtype="float32")
            sims, indices = self.index.search(query_vec, pool)
            for idx, sim in zip(indices[0], sims[0]):
                if idx < 0 or idx >= len(self.docs):
                    continue
                chunk = self.docs[idx]
                if not jurisdiction_matches(chunk, jurisdiction):
                    continue
                candidate_scores[idx] = candidate_scores.get(idx, 0.0) + alpha * float(sim)

        tokens = [token for token in query.lower().split() if token]
        if tokens:
            bm25_scores = np.array(self.bm25.get_scores(tokens), dtype="float32")
            max_score = float(bm25_scores.max()) if bm25_scores.size else 0.0
            if max_score > 0:
                for idx, score in enumerate(bm25_scores):
                    if score <= 0:
                        continue
                    chunk = self.docs[idx]
                    if not jurisdiction_matches(chunk, jurisdiction):
                        continue
                    norm_score = float(score / max_score)
                    candidate_scores[idx] = candidate_scores.get(idx, 0.0) + (1.0 - alpha) * norm_score

        if jurisdiction and not candidate_scores:
            # fallback without jurisdiction filter
            return self.hybrid_retrieve(query, top_k, jurisdiction=None, alpha=alpha)

        ranked = sorted(candidate_scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
        results: List[Chunk] = []
        for idx, score in ranked:
            chunk = replace(self.docs[idx], score=float(score))
            results.append(chunk)
        return results


def _validate_paths(paths: Iterable[Path]) -> Tuple[Path, ...]:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing knowledge base files. Run `python build_index.py` first.\n"
            f"Missing: {', '.join(missing)}"
        )
    return tuple(paths)


def _load_documents(raw_docs: List[Dict]) -> List[Chunk]:
    if not raw_docs:
        raise ValueError("documents.json did not contain any chunks.")

    chunks: List[Chunk] = []
    for idx, doc in enumerate(raw_docs):
        chunk = Chunk(
            id=doc.get("id", idx),
            text=doc.get("text", ""),
            source=doc.get("source", "Unknown"),
            jurisdiction=doc.get("jurisdiction", "Unknown"),
            meta=doc.get("meta", {}),
        )
        chunks.append(chunk)
    return chunks


def load_knowledge_base(
    docs_path: Path = DOCS_PATH,
    faiss_path: Path = FAISS_PATH,
    bm25_path: Path = BM25_PATH,
) -> KnowledgeBase:
    _validate_paths((docs_path, faiss_path, bm25_path))

    with docs_path.open("r", encoding="utf-8") as docs_file:
        raw_docs = json.load(docs_file)
    docs = _load_documents(raw_docs)

    index = faiss.read_index(str(faiss_path))
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    with bm25_path.open("rb") as bm25_file:
        bm25 = pickle.load(bm25_file)

    return KnowledgeBase(docs, index, embedder, bm25)
