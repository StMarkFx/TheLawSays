"""
Legacy Streamlit demo for TheLawSays.

This interface remains for quick local experimentation. The production-ready
experience will live in the FastAPI + Next.js stack (see ``upgrade.md``).

To switch back to this version at any time:
1. Ensure ``documents.json``, ``legal_index.faiss`` and ``bm25_index.pkl`` exist.
2. Add your OpenAI key to ``.streamlit/secrets.toml`` as ``OPENAI_API_KEY``.
3. Run ``streamlit run app.py``.

The FastAPI + Next.js deployment does not depend on this file, so feel free to
comment or customise it for demos without affecting production code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import streamlit as st

from thelawsays_core import (
    IntentDetector,
    build_conversational_prompt,
    build_no_results_prompt,
    build_rag_prompt,
    load_knowledge_base,
)
from thelawsays_core.data import Chunk
from thelawsays_core.openai_utils import create_openai_client, generate_completion
from thelawsays_core.pipeline import KnowledgeBase
from thelawsays_core.settings import DEFAULT_ALPHA, DEFAULT_TOP_K


st.set_page_config(
    page_title="TheLawSays - Nigerian Legal AI",
    page_icon="⚖",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown(
    '<h1 style="text-align: center;">⚖ TheLawSays — Nigerian Legal AI Assistant</h1>',
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_resources() -> KnowledgeBase:
    return load_knowledge_base()


try:
    knowledge_base = load_resources()
except Exception as exc:  # pragma: no cover - executed in UI only
    st.error(f"Failed to load the knowledge base: {exc}")
    st.stop()


openai_key = st.secrets.get("OPENAI_API_KEY")
openai_client = create_openai_client(openai_key)

if not openai_client:
    st.warning("OpenAI API key missing. Add `OPENAI_API_KEY` to .streamlit/secrets.toml to enable answers.")

intent_detector = IntentDetector(client=openai_client)


def detect_jurisdiction(query: str) -> Optional[str]:
    lowered = query.lower()
    if "lagos" in lowered:
        return "Lagos"
    if any(token in lowered for token in ("federal", "nigeria", "abuja", "nationwide")):
        return "Federal"
    return None


def render_sidebar() -> Dict[str, Optional[str]]:
    with st.sidebar:
        st.markdown("### Search Configuration")

        jurisdiction_choice = st.radio(
            "**Jurisdiction Filter**",
            ["Auto-detect", "Federal Law", "Lagos State"],
            index=0,
            help="Filter search results by legal jurisdiction. Auto-detect analyses your query.",
        )

        top_k = st.slider(
            "**Source Depth**",
            min_value=3,
            max_value=8,
            value=DEFAULT_TOP_K,
            step=1,
            help="Number of legal excerpts to analyse (more sources = slower but potentially better answers).",
        )

        st.markdown("---")
        st.markdown("### About TheLawSays")
        st.markdown(
            "Empower yourself with Nigerian law at your fingertips. Ask any question about Federal or Lagos State "
            "statutes and get instant, verifiable answers backed by cited sections."
        )
        st.markdown("### Built by St. Mark Adebayo")
        st.markdown(
            "St. Mark is an AI/ML Engineer focused on democratizing access to legal information through responsible AI."
        )

        if openai_client:
            st.success("AI Engine: Connected")
        else:
            st.error("AI Engine: API Key Missing")

        return {
            "jurisdiction_option": jurisdiction_choice,
            "top_k": top_k,
        }


sidebar_state = render_sidebar()


def map_jurisdiction(selected: str) -> Optional[str]:
    mapping = {
        "Auto-detect": None,
        "Federal Law": "Federal",
        "Lagos State": "Lagos",
    }
    return mapping.get(selected)


if "messages" not in st.session_state:
    st.session_state.messages: List[Dict[str, str]] = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


def maybe_retrieve(decision_label: str, jurisdiction: Optional[str], query: str, top_k: int) -> List[Chunk]:
    if decision_label != "legal_lookup":
        return []
    return knowledge_base.hybrid_retrieve(query=query, top_k=top_k, jurisdiction=jurisdiction, alpha=DEFAULT_ALPHA)


if query := st.chat_input("Find out what the law says about..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    selected_jurisdiction = map_jurisdiction(sidebar_state["jurisdiction_option"])
    inferred_jurisdiction = selected_jurisdiction or detect_jurisdiction(query)

    decision = intent_detector.classify(query)

    with st.chat_message("assistant"):
        chunks: List[Chunk] = []
        if decision.retrieval_required:
            with st.spinner("Retrieving relevant excerpts..."):
                chunks = maybe_retrieve(decision.label, inferred_jurisdiction, query, sidebar_state["top_k"])

        if not decision.retrieval_required:
            prompt = build_conversational_prompt(query)
        elif not chunks:
            prompt = build_no_results_prompt(query)
        else:
            prompt = build_rag_prompt(query, chunks, inferred_jurisdiction)

        with st.spinner("Drafting answer..."):
            answer = generate_completion(openai_client, prompt)

        st.write(answer)

        if chunks:
            with st.expander("Show retrieved sources"):
                for idx, chunk in enumerate(chunks, start=1):
                    st.markdown(f"**Excerpt {idx}** — {chunk.source} ({chunk.jurisdiction})")
                    st.markdown(f"> {chunk.text}")

        st.session_state.messages.append({"role": "assistant", "content": answer})


st.markdown(
    """
<div style="text-align: center; color: #6c757d; font-size: 0.8rem; margin: 1rem 0;">
This is for research and educational purposes only. It is not legal advice and should not be used as a substitute for
professional legal counsel. Always consult qualified lawyers for specific legal situations.
<a href="https://github.com/StMarkFx/TheLawSays" target="_blank" style="color: #007bff;">View on GitHub</a>
</div>
""",
    unsafe_allow_html=True,
)
