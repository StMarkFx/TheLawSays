# build_index.py
from __future__ import annotations

import json
import pickle
import time
import signal
import sys
from dataclasses import dataclass
from hashlib import md5
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional
import logging
import argparse

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
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import functools


BASE_DIR = Path("laws")
OUTPUT_DOCS = Path("data/documents.json")
OUTPUT_FAISS = Path("data/indices/legal_index.faiss")
OUTPUT_BM25 = Path("data/indices/bm25_index.pkl")

MIN_CHARS_PER_CHUNK = 50
MIN_WORDS_PER_CHUNK = 8
FAISS_SEARCH_METRIC = "ip"  # inner product on normalised vectors

JAVA_AVAILABLE = shutil.which("java") is not None
POPPLER_AVAILABLE = shutil.which("pdftoppm") is not None

# Progress and checkpointing files
PROGRESS_FILE = Path("progress.json")
FAILED_PDFS_FILE = Path("failed_pdfs.json")
DOCUMENTS_PARTIAL = Path("documents_partial.json")
EMBEDDINGS_PARTIAL = Path("embeddings_partial.npy")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/logs/build_index.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class ProgressState:
    last_pdf: str
    processed_pdfs: int
    total_pdfs: int
    documents_count: int
    timestamp: float
    jurisdiction_filter: Optional[str] = None


def save_progress(state: ProgressState) -> None:
    """Save current progress to file."""
    try:
        with PROGRESS_FILE.open('w') as f:
            json.dump({
                'last_pdf': state.last_pdf,
                'processed_pdfs': state.processed_pdfs,
                'total_pdfs': state.total_pdfs,
                'documents_count': state.documents_count,
                'timestamp': state.timestamp,
                'jurisdiction_filter': state.jurisdiction_filter
            }, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save progress: {e}")


def load_progress() -> Optional[ProgressState]:
    """Load progress from file if it exists."""
    if not PROGRESS_FILE.exists():
        return None
    try:
        with PROGRESS_FILE.open('r') as f:
            data = json.load(f)
        return ProgressState(
            last_pdf=data['last_pdf'],
            processed_pdfs=data['processed_pdfs'],
            total_pdfs=data['total_pdfs'],
            documents_count=data['documents_count'],
            timestamp=data['timestamp'],
            jurisdiction_filter=data.get('jurisdiction_filter')
        )
    except Exception as e:
        logger.warning(f"Failed to load progress: {e}")
        return None


def save_partial_documents(documents: List[Chunk]) -> None:
    """Save partial documents to allow resuming."""
    try:
        serializable_docs = [
            {
                "text": doc.text,
                "source": doc.source,
                "jurisdiction": doc.jurisdiction,
                "meta": doc.meta,
            }
            for doc in documents
        ]
        # Write to temp file first, then rename for atomicity
        temp_file = DOCUMENTS_PARTIAL.with_suffix('.tmp')
        with temp_file.open("w", encoding="utf-8") as f:
            json.dump(serializable_docs, f, indent=2)
        temp_file.replace(DOCUMENTS_PARTIAL)
    except Exception as e:
        logger.warning(f"Failed to save partial documents: {e}")


def load_partial_documents() -> List[Chunk]:
    """Load partial documents if they exist."""
    if not DOCUMENTS_PARTIAL.exists():
        return []
    try:
        with DOCUMENTS_PARTIAL.open('r', encoding='utf-8') as f:
            data = json.load(f)
        return [
            Chunk(
                text=item['text'],
                source=item['source'],
                jurisdiction=item['jurisdiction'],
                meta=item['meta']
            )
            for item in data
        ]
    except Exception as e:
        logger.warning(f"Failed to load partial documents: {e}")
        return []


def save_partial_embeddings(embeddings: np.ndarray) -> None:
    """Save partial embeddings to allow resuming."""
    try:
        # Write to temp file first, then rename for atomicity
        temp_file = EMBEDDINGS_PARTIAL.with_suffix('.tmp')
        np.save(temp_file, embeddings)
        temp_file.replace(EMBEDDINGS_PARTIAL)
    except Exception as e:
        logger.warning(f"Failed to save partial embeddings: {e}")


def load_partial_embeddings() -> Optional[np.ndarray]:
    """Load partial embeddings if they exist."""
    if not EMBEDDINGS_PARTIAL.exists():
        return None
    try:
        return np.load(EMBEDDINGS_PARTIAL)
    except Exception as e:
        logger.warning(f"Failed to load partial embeddings: {e}")
        return None


def log_failed_pdf(pdf_path: Path, error: str) -> None:
    """Log failed PDF to file."""
    try:
        failed_data = {}
        if FAILED_PDFS_FILE.exists():
            with FAILED_PDFS_FILE.open('r') as f:
                failed_data = json.load(f)

        failed_data[str(pdf_path)] = {
            'error': error,
            'timestamp': time.time()
        }

        with FAILED_PDFS_FILE.open('w') as f:
            json.dump(failed_data, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to log failed PDF: {e}")


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


def timed_convert_from_path(pdf_path: str, dpi: int, max_pages: int, timeout_seconds: int = 120) -> list:
    """Convert PDF to images with timeout protection."""
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="pdf_conversion") as executor:
        future = executor.submit(
            convert_from_path,
            pdf_path,
            dpi=dpi,
            first_page=1,
            last_page=max_pages
        )
        try:
            return future.result(timeout=timeout_seconds)
        except FutureTimeoutError:
            logger.warning(f"PDF conversion timeout after {timeout_seconds}s at {dpi} DPI")
            raise TimeoutError(f"PDF conversion timed out at {dpi} DPI")


def ocr_page_with_timeout(img, timeout_seconds: int = 15) -> str:
    """OCR a single page with shorter timeout."""
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(pytesseract.image_to_string, img)
        try:
            return future.result(timeout=timeout_seconds)
        except FutureTimeoutError:
            logger.warning(f"OCR timeout after {timeout_seconds}s")
            return ""


def ocr_pdf(pdf_path: Path, max_pages: int = 50) -> str:
    """High-performance OCR with smart optimizations and timeout protection."""
    if not POPPLER_AVAILABLE:
        raise RuntimeError(
            "Poppler (pdftoppm) is required for OCR but was not found on PATH. "
            "Install Poppler and ensure the bin directory is added to PATH."
        )

    logger.info(f"[OCR] Processing {pdf_path.name}")

    try:
        # Step 1: Quick page count check - skip very large documents
        from pdf2image import pdfinfo_from_path
        try:
            page_info = pdfinfo_from_path(str(pdf_path))
            total_pages = page_info['Pages']
            logger.info(f"[OCR] Document has {total_pages} pages")

            # Skip documents that are too large (>150 pages)
            if total_pages > 150:
                logger.warning(f"[OCR] Skipping {pdf_path.name}: too many pages ({total_pages})")
                return ""

            # Adjust max_pages based on document size
            if total_pages > 100:
                max_pages = min(max_pages, 20)  # Sample fewer pages from very large docs
            elif total_pages > 50:
                max_pages = min(max_pages, 15)
            else:
                max_pages = min(max_pages, total_pages)

        except Exception as e:
            logger.warning(f"[OCR] Could not get page count for {pdf_path.name}: {e}")
            # Continue with default max_pages

        # Step 2: Progressive DPI conversion with timeout
        images = None
        dpi_attempts = [72, 100, 150]  # Start with lowest DPI for speed

        for dpi in dpi_attempts:
            try:
                logger.info(f"[OCR] Attempting conversion at {dpi} DPI (max {max_pages} pages)")
                images = timed_convert_from_path(str(pdf_path), dpi, max_pages, timeout_seconds=90)
                logger.info(f"[OCR] Successfully converted {len(images)} pages at {dpi} DPI")
                break
            except TimeoutError:
                logger.warning(f"[OCR] Conversion timeout at {dpi} DPI, trying higher DPI")
                continue
            except Exception as e:
                logger.warning(f"[OCR] Conversion failed at {dpi} DPI: {e}")
                continue

        if images is None:
            logger.error(f"[OCR] All conversion attempts failed for {pdf_path.name}")
            return ""

        # Step 3: Concurrent OCR processing
        logger.info(f"[OCR] Running OCR on {len(images)} pages")

        # Process pages in parallel batches for speed
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(ocr_page_with_timeout, img, 15) for img in images]
            page_texts = []
            for i, future in enumerate(futures, 1):
                try:
                    text = future.result(timeout=20)  # 20 second timeout per page
                    if text.strip():
                        page_texts.append(text)
                    if i % 5 == 0:
                        logger.info(f"[OCR] Processed {i}/{len(images)} pages")
                except Exception as e:
                    logger.warning(f"[OCR] Failed to OCR page {i}: {e}")

        result = "\n".join(page_texts)
        logger.info(f"[OCR] Extracted {len(result)} characters from {pdf_path.name}")
        return result

    except PDFInfoNotInstalledError as err:
        raise RuntimeError(
            "Poppler is required for pdf2image. Install it and ensure `pdftoppm` is available."
        ) from err
    except Exception as e:
        logger.error(f"[OCR] Failed to process {pdf_path.name}: {e}")
        return ""


def clean_text(text: str) -> str:
    return " ".join(text.split())


def extract_text(pdf_path: Path) -> Tuple[str, Dict[str, str]]:
    """Extract text from PDF with robust error handling."""
    metadata: Dict[str, str] = {}
    text: str = ""

    # Try Tika first (fastest method)
    if JAVA_AVAILABLE:
        try:
            logger.info(f"[EXTRACT] Trying Tika for {pdf_path.name}")
            parsed = parser.from_file(str(pdf_path)) or {}
            metadata = parsed.get("metadata", {}) or {}
            text = (parsed.get("content") or "").strip()
            if text:
                logger.info(f"[EXTRACT] Tika extracted {len(text)} chars from {pdf_path.name}")
                return clean_text(text), metadata
        except Exception as exc:
            logger.warning(f"[EXTRACT] Tika failed for {pdf_path.name}: {exc}")
    else:
        logger.info("[EXTRACT] Java not available, skipping Tika")

    # Fall back to OCR if Tika failed or isn't available
    logger.info(f"[EXTRACT] Falling back to OCR for {pdf_path.name}")
    text = ocr_pdf(pdf_path)
    if not text:
        logger.error(f"[EXTRACT] All extraction methods failed for {pdf_path.name}")
        return "", metadata

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


def iter_pdfs(jurisdiction_filter: Optional[str] = None) -> Iterable[Tuple[str, Path]]:
    """Iterate through PDFs with optional jurisdiction filtering."""
    jurisdictions = ["federal", "lagos"]
    if jurisdiction_filter:
        jurisdiction_filter = jurisdiction_filter.lower()
        if jurisdiction_filter in ["federal", "lagos"]:
            jurisdictions = [jurisdiction_filter]
        else:
            logger.warning(f"Unknown jurisdiction filter: {jurisdiction_filter}")

    for jurisdiction_dir in jurisdictions:
        dir_path = BASE_DIR / jurisdiction_dir
        if not dir_path.exists():
            logger.warning(f"Jurisdiction directory not found: {dir_path}")
            continue
        for pdf_path in sorted(dir_path.glob("*.pdf")):
            yield jurisdiction_dir.capitalize(), pdf_path


def build_documents(nlp: spacy.Language, jurisdiction_filter: Optional[str] = None, resume: bool = False) -> List[Chunk]:
    """Build documents with error handling, checkpointing, and progress tracking."""
    documents: List[Chunk] = []
    seen_hashes = set()

    # Load partial documents if resuming
    if resume:
        documents = load_partial_documents()
        seen_hashes = {md5(doc.text[:512].encode("utf-8", errors="ignore")).hexdigest() for doc in documents}
        logger.info(f"[RESUME] Loaded {len(documents)} documents from checkpoint")

    # Count total PDFs for progress tracking
    total_pdfs = sum(1 for _ in iter_pdfs(jurisdiction_filter))
    processed_pdfs = len(documents)  # Approximate based on loaded documents

    logger.info(f"[BUILD] Processing {total_pdfs} PDFs ({'resuming' if resume else 'starting fresh'})")

    progress_state = ProgressState(
        last_pdf="",
        processed_pdfs=processed_pdfs,
        total_pdfs=total_pdfs,
        documents_count=len(documents),
        timestamp=time.time(),
        jurisdiction_filter=jurisdiction_filter
    )

    try:
        for jurisdiction, pdf_path in iter_pdfs(jurisdiction_filter):
            pdf_name = pdf_path.name

            # Skip if already processed (when resuming)
            if resume and any(doc.source == pdf_name for doc in documents):
                logger.info(f"[SKIP] Already processed: {pdf_name}")
                continue

            logger.info(f"[LOAD] Processing {pdf_path.relative_to(BASE_DIR)} ({processed_pdfs + 1}/{total_pdfs})")

            try:
                text, metadata = extract_text(pdf_path)
                if not text:
                    logger.error(f"[SKIP] No text extracted from {pdf_name}")
                    log_failed_pdf(pdf_path, "No text extracted")
                    processed_pdfs += 1
                    continue

                chunks_added = 0
                for chunk in chunk_text(nlp, text):
                    key = md5(chunk[:512].encode("utf-8", errors="ignore")).hexdigest()
                    if key in seen_hashes:
                        continue
                    seen_hashes.add(key)
                    documents.append(
                        Chunk(
                            text=chunk,
                            source=pdf_name,
                            jurisdiction=jurisdiction,
                            meta={
                                "title": metadata.get("dc:title", "Unknown"),
                                "creator": metadata.get("dc:creator", "Unknown"),
                                "date": metadata.get("Creation-Date", "Unknown"),
                            },
                        )
                    )
                    chunks_added += 1

                logger.info(f"[LOAD] Added {chunks_added} chunks from {pdf_name}")
                processed_pdfs += 1

                # Save progress and checkpoint every 5 PDFs
                if processed_pdfs % 5 == 0:
                    progress_state.last_pdf = pdf_name
                    progress_state.processed_pdfs = processed_pdfs
                    progress_state.documents_count = len(documents)
                    progress_state.timestamp = time.time()
                    save_progress(progress_state)
                    save_partial_documents(documents)
                    logger.info(f"[CHECKPOINT] Saved progress: {processed_pdfs}/{total_pdfs} PDFs, {len(documents)} chunks")

            except Exception as e:
                logger.error(f"[ERROR] Failed to process {pdf_name}: {e}")
                log_failed_pdf(pdf_path, str(e))
                processed_pdfs += 1
                continue

        # Final save
        progress_state.last_pdf = pdf_name if 'pdf_name' in locals() else ""
        progress_state.processed_pdfs = processed_pdfs
        progress_state.documents_count = len(documents)
        progress_state.timestamp = time.time()
        save_progress(progress_state)
        save_partial_documents(documents)

        logger.info(f"[BUILD] Completed processing. Total chunks: {len(documents)}")

    except KeyboardInterrupt:
        logger.warning("[BUILD] Interrupted by user. Saving checkpoint...")
        save_progress(progress_state)
        save_partial_documents(documents)
        logger.info(f"[CHECKPOINT] Saved {len(documents)} chunks. Run with --resume to continue.")
        raise

    return documents


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    embeddings = embeddings.astype("float32")
    if FAISS_SEARCH_METRIC == "ip":
        index: faiss.Index = faiss.IndexFlatIP(embeddings.shape[1])
    else:
        index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index


def create_embeddings(documents: List[Chunk], model: SentenceTransformer, resume: bool = False) -> np.ndarray:
    """Create embeddings with checkpointing support."""
    texts = [d.text for d in documents]

    # Try to load partial embeddings if resuming
    if resume:
        partial_embeddings = load_partial_embeddings()
        if partial_embeddings is not None and len(partial_embeddings) == len(texts):
            logger.info(f"[EMBED] Loaded {len(partial_embeddings)} embeddings from checkpoint")
            return partial_embeddings
        elif partial_embeddings is not None:
            logger.warning(f"[EMBED] Partial embeddings count mismatch ({len(partial_embeddings)} vs {len(texts)}), recomputing")

    logger.info(f"[EMBED] Creating embeddings for {len(texts)} chunks")

    try:
        embeddings = model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=(FAISS_SEARCH_METRIC == "ip"),
        )
        embeddings = np.array(embeddings)

        # Save embeddings checkpoint
        save_partial_embeddings(embeddings)
        logger.info("[EMBED] Saved embeddings checkpoint")

        return embeddings

    except KeyboardInterrupt:
        logger.warning("[EMBED] Interrupted, saving partial embeddings...")
        # Note: We can't save partial embeddings during encoding as we don't have them yet
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Build legal document index")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--jurisdiction", choices=["federal", "lagos"], help="Process only specific jurisdiction")
    parser.add_argument("--clean", action="store_true", help="Remove all checkpoint and output files before starting")
    args = parser.parse_args()

    if not BASE_DIR.exists():
        raise SystemExit("No 'laws/' directory found. Add PDFs before building the index.")

    # Clean files if requested
    if args.clean:
        logger.info("[CLEAN] Removing checkpoint and output files...")
        for file in [PROGRESS_FILE, FAILED_PDFS_FILE, DOCUMENTS_PARTIAL, EMBEDDINGS_PARTIAL,
                     OUTPUT_DOCS, OUTPUT_FAISS, OUTPUT_BM25]:
            if file.exists():
                file.unlink()
                logger.info(f"[CLEAN] Removed {file}")

    # Check for resume
    if args.resume and not PROGRESS_FILE.exists():
        logger.warning("[RESUME] No checkpoint found, starting fresh")
        args.resume = False

    logger.info(f"[START] Building index (jurisdiction: {args.jurisdiction or 'all'}, resume: {args.resume})")

    # Load models
    logger.info("[LOAD] Loading NLP and embedding models...")
    nlp, model = load_models()

    # Build documents with checkpointing
    documents = build_documents(nlp, args.jurisdiction, args.resume)
    if not documents:
        raise SystemExit("No chunks were generated. Check your PDFs or OCR setup.")

    # Create embeddings with checkpointing
    embeddings = create_embeddings(documents, model, args.resume)

    # Build FAISS index
    logger.info("[INDEX] Building FAISS index...")
    index = build_faiss_index(embeddings)
    faiss.write_index(index, str(OUTPUT_FAISS))

    # Build BM25 index
    logger.info("[INDEX] Building BM25 index...")
    tokenized = [d.text.lower().split() for d in documents]
    bm25 = BM25Okapi(tokenized)
    with OUTPUT_BM25.open("wb") as bm25_file:
        pickle.dump(bm25, bm25_file)

    # Save documents
    logger.info("[SAVE] Saving documents...")
    serializable_docs = [
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
        json.dump(serializable_docs, docs_file, indent=2)

    # Clean up checkpoint files on successful completion
    logger.info("[CLEANUP] Cleaning up checkpoint files...")
    for file in [PROGRESS_FILE, DOCUMENTS_PARTIAL, EMBEDDINGS_PARTIAL]:
        if file.exists():
            file.unlink()

    logger.info(f"[DONE] Index built successfully!")
    logger.info(f"  Documents: {OUTPUT_DOCS} ({len(documents)} chunks)")
    logger.info(f"  FAISS index: {OUTPUT_FAISS}")
    logger.info(f"  BM25 index: {OUTPUT_BM25}")

    if FAILED_PDFS_FILE.exists():
        logger.warning(f"  Failed PDFs logged to: {FAILED_PDFS_FILE}")


if __name__ == "__main__":
    main()
