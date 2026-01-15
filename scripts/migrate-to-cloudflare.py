#!/usr/bin/env python3
"""
Migration script to move TheLawSays knowledge base into Cloudflare Vectorize (and optionally D1).

This script:
1. Loads document metadata from documents.json
2. Generates embeddings for all text chunks
3. Writes vectors to a Vectorize NDJSON file for upload (streamed)
4. Writes a D1 schema file for optional metadata storage

Usage:
    python scripts/migrate-to-cloudflare.py

Requirements:
- documents.json in data/
- Cloudflare credentials configured
- Wrangler CLI authenticated
"""

import json
import sys
from pathlib import Path
from typing import List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from sentence_transformers import SentenceTransformer
    from thelawsays_core.data import Chunk
except ImportError as e:
    print(f"Missing required packages: {e}")
    print("Install with: pip install sentence-transformers")
    sys.exit(1)


class CloudflareMigrator:
    def __init__(self):
        self.data_dir = Path(__file__).parent.parent / "data"
        self.documents_file = self.data_dir / "documents.json"

        print("Loading embedding model...")
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        self.chunks = self.load_chunks()
        print(f"Loaded {len(self.chunks)} chunks")

    def load_chunks(self) -> List[Chunk]:
        """Load chunks from documents.json."""
        if not self.documents_file.exists():
            raise FileNotFoundError(f"documents.json not found at {self.documents_file}")

        with open(self.documents_file, "r", encoding="utf-8") as f:
            documents_data = json.load(f)

        chunks: List[Chunk] = []
        for doc_data in documents_data:
            if "chunks" in doc_data and isinstance(doc_data.get("chunks"), list):
                doc_id = doc_data.get("id", "doc")
                for i, chunk_data in enumerate(doc_data.get("chunks", [])):
                    chunk = Chunk(
                        id=chunk_data.get("id", f"{doc_id}_{i}"),
                        text=chunk_data["text"],
                        source=doc_data.get("filename", "unknown"),
                        jurisdiction=doc_data.get("jurisdiction", "federal"),
                        meta=chunk_data.get("meta") or chunk_data.get("metadata", {}),
                    )
                    chunks.append(chunk)
                continue

            if "text" in doc_data:
                chunk = Chunk(
                    id=doc_data.get("id", "chunk"),
                    text=doc_data["text"],
                    source=doc_data.get("source", "unknown"),
                    jurisdiction=doc_data.get("jurisdiction", "federal"),
                    meta=doc_data.get("meta", {}),
                )
                chunks.append(chunk)

        return chunks

    def create_d1_schema(self):
        """Generate D1 schema for chunk metadata (optional)."""
        schema = """
        CREATE TABLE IF NOT EXISTS chunks (
            id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            jurisdiction TEXT NOT NULL,
            text TEXT NOT NULL,
            metadata TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_chunks_jurisdiction ON chunks(jurisdiction);
        CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source);
        """

        schema_file = Path(__file__).parent.parent / "d1-schema.sql"
        with open(schema_file, "w", encoding="utf-8") as f:
            f.write(schema)

        print(f"D1 schema written to {schema_file}")
        print("Execute this schema in your D1 database if you plan to use D1 lookups.")

    def write_vectorize_ndjson_streamed(self):
        """Stream embeddings to vectorize.ndjson so the job can resume if interrupted."""
        output_file = Path(__file__).parent.parent / "vectorize.ndjson"
        batch_size = 64

        start_index = 0
        if output_file.exists():
            with open(output_file, "r", encoding="utf-8") as f:
                start_index = sum(1 for _ in f)

        if start_index >= len(self.chunks):
            print("vectorize.ndjson already complete.")
            return

        print(f"Writing vectorize.ndjson starting at index {start_index}...")

        with open(output_file, "a", encoding="utf-8") as f:
            for i in range(start_index, len(self.chunks), batch_size):
                batch = self.chunks[i : i + batch_size]
                texts = [chunk.text for chunk in batch]
                embeddings = self.model.encode(texts, normalize_embeddings=True)

                for j, chunk in enumerate(batch):
                    item = {
                        "id": str(chunk.id),
                        "values": embeddings[j].tolist(),
                        "metadata": {
                            "source": chunk.source,
                            "jurisdiction": chunk.jurisdiction,
                            "text": chunk.text[:1000],
                            "meta": chunk.meta,
                        },
                    }
                    f.write(json.dumps(item, ensure_ascii=True) + "\n")

                print(f"Processed {min(i + batch_size, len(self.chunks))}/{len(self.chunks)} chunks")

        print(f"Vectorize NDJSON written to {output_file}")
        print("Upload with:")
        print("wrangler vectorize insert thelawsays-vectorize --file vectorize.ndjson --batch-size 500")

    def run_migration(self):
        """Run the complete migration process."""
        print("Starting Cloudflare migration...")

        print("Step 1: Creating D1 schema...")
        self.create_d1_schema()

        print("Step 2: Writing Vectorize NDJSON...")
        self.write_vectorize_ndjson_streamed()

        print("Migration preparation complete!")
        print("- Generated vectorize.ndjson for upload")
        print("- Created d1-schema.sql for optional D1 use")

        print("Next steps:")
        print("1. Upload vectors: wrangler vectorize insert thelawsays-vectorize --file vectorize.ndjson --batch-size 500")
        print("2. Update wrangler.toml with correct Vectorize/D1 IDs (already done)")


def main():
    try:
        migrator = CloudflareMigrator()
        migrator.run_migration()
    except Exception as e:
        print(f"Migration failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
