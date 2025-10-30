# app.py
from __future__ import annotations

import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import faiss
import numpy as np
import streamlit as st
from openai import OpenAI
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

DOCS_PATH = Path("documents.json")
FAISS_PATH = Path("legal_index.faiss")
BM25_PATH = Path("bm25_index.pkl")
OPENAI_MODEL = "gpt-4o-mini"

st.set_page_config(page_title="TheLawSays", page_icon=":scales:", layout="centered")
st.markdown("<h1 style='text-align: center;'>:scales: TheLawSays</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; color: #666;'>Federal and Lagos laws | Cited | Reduced hallucinations</p>",
    unsafe_allow_html=True,
)
st.info(":warning: For research only | Not legal advice")


@st.cache_resource(show_spinner=False)
def load_pipeline() -> tuple[List[Dict], faiss.Index, SentenceTransformer, BM25Okapi]:
    missing = [str(path) for path in (DOCS_PATH, FAISS_PATH, BM25_PATH) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing knowledge base files. Run `python build_index.py` first.\n"
            f"Missing: {', '.join(missing)}"
        )

    with DOCS_PATH.open("r", encoding="utf-8") as docs_file:
        docs = json.load(docs_file)
    if not isinstance(docs, list) or not docs:
        raise ValueError("documents.json did not contain any chunks.")

    index = faiss.read_index(str(FAISS_PATH))
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    with BM25_PATH.open("rb") as bm25_file:
        bm25 = pickle.load(bm25_file)

    # Normalise document ids and cached lowercase jurisdictions for quicker filters
    for idx, doc in enumerate(docs):
        doc.setdefault("id", idx)
        doc["jurisdiction_lower"] = doc.get("jurisdiction", "").lower()

    return docs, index, embedder, bm25


try:
    docs, index, embedder, bm25 = load_pipeline()
except Exception as exc:
    st.error(f"Failed to load the knowledge base: {exc}")
    st.stop()


from openai import OpenAI

openai_key = st.secrets.get("OPENAI_API_KEY")
openai_client = OpenAI(api_key=openai_key) if openai_key else None

if not openai_client:
    st.warning("OpenAI API key missing. Add `OPENAI_API_KEY` to .streamlit/secrets.toml to enable answers.")


def detect_jurisdiction(query: str) -> Optional[str]:
    lowered = query.lower()
    if "lagos" in lowered:
        return "Lagos"
    if any(token in lowered for token in ("federal", "nigeria", "abuja", "nationwide")):
        return "Federal"
    return None


def jurisdiction_matches(doc: Dict, jurisdiction: Optional[str]) -> bool:
    if jurisdiction is None:
        return True
    return doc.get("jurisdiction_lower") == jurisdiction.lower()


def hybrid_retrieve(query: str, top_k: int, jurisdiction: Optional[str], alpha: float = 0.65) -> List[Dict]:
    candidate_scores: Dict[int, float] = defaultdict(float)

    # Vector search (cosine similarity because embeddings are normalised)
    pool = min(top_k * 5, len(docs))
    if pool:
        query_vec = embedder.encode([query], normalize_embeddings=True)
        query_vec = np.asarray(query_vec, dtype="float32")
        sims, indices = index.search(query_vec, pool)
        for doc_idx, sim in zip(indices[0], sims[0]):
            if doc_idx < 0 or doc_idx >= len(docs):
                continue
            doc = docs[doc_idx]
            if not jurisdiction_matches(doc, jurisdiction):
                continue
            candidate_scores[doc_idx] += alpha * float(sim)

    # Lexical search (BM25)
    tokens = [token for token in query.lower().split() if token]
    if tokens:
        bm25_scores = np.array(bm25.get_scores(tokens), dtype="float32")
        if bm25_scores.size and bm25_scores.max() > 0:
            normaliser = float(bm25_scores.max())
            for doc_idx, score in enumerate(bm25_scores):
                if score <= 0:
                    continue
                doc = docs[doc_idx]
                if not jurisdiction_matches(doc, jurisdiction):
                    continue
                candidate_scores[doc_idx] += (1.0 - alpha) * float(score / normaliser)

    if jurisdiction and not candidate_scores:
        # Soft fallback to avoid empty answers when a strict filter misses
        return hybrid_retrieve(query, top_k, jurisdiction=None, alpha=alpha)

    ranked = sorted(candidate_scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
    return [docs[idx] for idx, _ in ranked]


def build_prompt(query: str, chunks: List[Dict], jurisdiction: Optional[str]) -> str:
    if jurisdiction:
        clarification = f"The user is asking about {jurisdiction} law."
    else:
        clarification = "Clarify whether the user needs Federal or Lagos law if it is not explicit."

    context_blocks = []
    for idx, chunk in enumerate(chunks, start=1):
        source = chunk.get("source", "Unknown").replace(".pdf", "")
        section = chunk.get("meta", {}).get("title", "Unknown")
        juris = chunk.get("jurisdiction", "Unknown")
        excerpt = chunk.get("text", "")[:1200]
        context_blocks.append(
            f"**Source {idx}: {source} ({juris})** | **Section:** {section}\n> {excerpt}"
        )

    context = "\n\n".join(context_blocks)

    prompt = f"""
You are a Nigerian lawyer. Answer only using the excerpts provided.

{clarification}

Quote law like:
> According to Section 88 of the Criminal Law of Lagos State: "exact quotation"

--- EXCERPTS ---
{context}
--- QUESTION ---
{query}

Answer:
"""
    return prompt.strip()


def openai_generate(prompt: str) -> str:
    if not openai_client:
        return "Connect an OpenAI API key to enable AI-generated answers."
    try:
        response = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        return f"OpenAI error: {exc}"


with st.sidebar:
    st.header("Search options")
    scope_choice = st.radio("Jurisdiction filter", ("Auto", "Federal", "Lagos"), index=0)
    top_k = st.slider("Max excerpts", min_value=3, max_value=8, value=4, step=1)


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


if query := st.chat_input("Ask about Nigerian laws..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    selected_jurisdiction = None if scope_choice == "Auto" else scope_choice
    inferred_jurisdiction = selected_jurisdiction or detect_jurisdiction(query)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving relevant excerpts..."):
            chunks = hybrid_retrieve(query, top_k=top_k, jurisdiction=inferred_jurisdiction)

        if not chunks:
            answer = "I could not find relevant excerpts. Try rephrasing the question or selecting Auto scope."
        else:
            with st.spinner("Drafting answer..."):
                answer = openai_generate(build_prompt(query, chunks, inferred_jurisdiction))

        st.write(answer)

        if chunks:
            with st.expander("Show retrieved excerpts"):
                for idx, chunk in enumerate(chunks, start=1):
                    st.markdown(
                        f"**Excerpt {idx}** -- {chunk.get('source', 'Unknown')} ({chunk.get('jurisdiction', 'Unknown')})"
                    )
                    st.markdown(f"> {chunk.get('text', '')}")

        st.session_state.messages.append({"role": "assistant", "content": answer})


st.markdown("---")
st.markdown("**Last updated:** 2025-10-30 | Built with Tika, OCR, MiniLM, FAISS, OpenAI")
