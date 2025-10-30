# build_index.py
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from hashlib import md5
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import faiss
import numpy as np
import pytesseract
import spacy
from pdf2image import convert_from_path
from pdf2image.exceptions import PDFInfoNotInstalledError
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from tika import parser
import shutil


BASE_DIR = Path("laws")
OUTPUT_DOCS = Path("documents.json")
OUTPUT_FAISS = Path("legal_index.faiss")
OUTPUT_BM25 = Path("bm25_index.pkl")

MIN_CHARS_PER_CHUNK = 50
MIN_WORDS_PER_CHUNK = 8
FAISS_SEARCH_METRIC = "ip"  # inner product on normalised vectors

JAVA_AVAILABLE = shutil.which("java") is not None
POPPLER_AVAILABLE = shutil.which("pdftoppm") is not None


@dataclass
class Chunk:
    text: str
    source: str
    jurisdiction: str
    meta: Dict[str, str]


def load_models() -> Tuple[spacy.Language, SentenceTransformer]:
    nlp = spacy.load("en_core_web_sm", disable=["ner"])
    if "parser" not in nlp.pipe_names:
        if "senter" in nlp.pipe_names:
            pass
        else:
            nlp.add_pipe("sentencizer")
    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    return nlp, model


def ocr_pdf(pdf_path: Path) -> str:
    if not POPPLER_AVAILABLE:
        raise RuntimeError(
            "Poppler (pdftoppm) is required for OCR but was not found on PATH. "
            "Install Poppler and ensure the bin directory is added to PATH."
        )
    print(f"[OCR] Falling back to OCR for {pdf_path.name}")
    try:
        images = convert_from_path(str(pdf_path), dpi=200)
    except PDFInfoNotInstalledError as err:
        raise RuntimeError(
            "Poppler is required for pdf2image. Install it and ensure `pdftoppm` is available."
        ) from err
    return "\n".join(pytesseract.image_to_string(img) for img in images)


def clean_text(text: str) -> str:
    return " ".join(text.split())


def extract_text(pdf_path: Path) -> Tuple[str, Dict[str, str]]:
    metadata: Dict[str, str] = {}
    text: str = ""
    if JAVA_AVAILABLE:
        try:
            parsed = parser.from_file(str(pdf_path)) or {}
            metadata = parsed.get("metadata", {}) or {}
            text = (parsed.get("content") or "").strip()
        except Exception as exc:
            print(f"[WARN] Tika failed for {pdf_path.name}: {exc}")
    else:
        print("[WARN] Java runtime not found. Skipping Tika extraction and using OCR/text fallback.")

    if not text:
        text = ocr_pdf(pdf_path)

    return clean_text(text), metadata


def chunk_text(nlp: spacy.Language, text: str) -> Iterable[str]:
    doc = nlp(text)
    for span in doc.sents:
        trimmed = span.text.strip()
        if len(trimmed) < MIN_CHARS_PER_CHUNK:
            continue
        if len(trimmed.split()) < MIN_WORDS_PER_CHUNK:
            continue
        yield trimmed


def iter_pdfs() -> Iterable[Tuple[str, Path]]:
    for jurisdiction_dir in ["federal", "lagos"]:
        dir_path = BASE_DIR / jurisdiction_dir
        if not dir_path.exists():
            continue
        for pdf_path in sorted(dir_path.glob("*.pdf")):
            yield jurisdiction_dir.capitalize(), pdf_path


def build_documents(nlp: spacy.Language) -> List[Chunk]:
    documents: List[Chunk] = []
    seen_hashes = set()
    for jurisdiction, pdf_path in iter_pdfs():
        print(f"[LOAD] Processing {pdf_path.relative_to(BASE_DIR)}")
        text, metadata = extract_text(pdf_path)
        if not text:
            print(f"[SKIP] No text extracted from {pdf_path.name}")
            continue
        for chunk in chunk_text(nlp, text):
            key = md5(chunk[:512].encode("utf-8", errors="ignore")).hexdigest()
            if key in seen_hashes:
                continue
            seen_hashes.add(key)
            documents.append(
                Chunk(
                    text=chunk,
                    source=pdf_path.name,
                    jurisdiction=jurisdiction,
                    meta={
                        "title": metadata.get("dc:title", "Unknown"),
                        "creator": metadata.get("dc:creator", "Unknown"),
                        "date": metadata.get("Creation-Date", "Unknown"),
                    },
                )
            )
    return documents


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    embeddings = embeddings.astype("float32")
    if FAISS_SEARCH_METRIC == "ip":
        index: faiss.Index = faiss.IndexFlatIP(embeddings.shape[1])
    else:
        index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index


def main() -> None:
    if not BASE_DIR.exists():
        raise SystemExit("No 'laws/' directory found. Add PDFs before building the index.")

    nlp, model = load_models()
    documents = build_documents(nlp)
    if not documents:
        raise SystemExit("No chunks were generated. Check your PDFs or OCR setup.")

    print(f"[INFO] Chunked {len(documents)} passages. Embedding...")
    texts = [d.text for d in documents]
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=(FAISS_SEARCH_METRIC == "ip"),
    )
    index = build_faiss_index(np.array(embeddings))
    faiss.write_index(index, str(OUTPUT_FAISS))

    tokenized = [d.text.lower().split() for d in documents]
    bm25 = BM25Okapi(tokenized)
    with OUTPUT_BM25.open("wb") as bm25_file:
        pickle.dump(bm25, bm25_file)

    serialisable_docs = [
        {
            "id": idx,
            "text": doc.text,
            "source": doc.source,
            "jurisdiction": doc.jurisdiction,
            "meta": doc.meta,
        }
        for idx, doc in enumerate(documents)
    ]
    with OUTPUT_DOCS.open("w", encoding="utf-8") as docs_file:
        json.dump(serialisable_docs, docs_file, indent=2)

    print(f"[DONE] Saved {len(documents)} chunks to {OUTPUT_DOCS}")
    print(f"[DONE] FAISS index: {OUTPUT_FAISS}")
    print(f"[DONE] BM25 index: {OUTPUT_BM25}")


if __name__ == "__main__":
    main()
