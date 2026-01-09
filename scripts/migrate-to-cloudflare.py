#!/usr/bin/env python3
"""
Migration script to move TheLawSays knowledge base from FAISS to Cloudflare Vectorize + D1.

This script:
1. Loads existing FAISS indices and document metadata
2. Generates embeddings for all text chunks
3. Uploads vectors to Cloudflare Vectorize
4. Stores chunk metadata in Cloudflare D1

Usage:
    python scripts/migrate-to-cloudflare.py

Requirements:
- Existing FAISS indices in data/indices/
- documents.json in data/
- Cloudflare credentials configured
- Wrangler CLI authenticated
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from sentence_transformers import SentenceTransformer
    from thelawsays_core.data import Chunk
    import faiss
except ImportError as e:
    print(f"Missing required packages: {e}")
    print("Install with: pip install sentence-transformers faiss-cpu")
    sys.exit(1)


class CloudflareMigrator:
    def __init__(self):
        self.data_dir = Path(__file__).parent.parent / "data"
        self.indices_dir = self.data_dir / "indices"
        self.documents_file = self.data_dir / "documents.json"

        # Initialize embedding model (same as original)
        print("Loading embedding model...")
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

        # Load existing data
        self.chunks = self.load_chunks()
        print(f"Loaded {len(self.chunks)} chunks")

    def load_chunks(self) -> List[Chunk]:
        """Load chunks from documents.json and FAISS indices."""
        if not self.documents_file.exists():
            raise FileNotFoundError(f"documents.json not found at {self.documents_file}")

        # Load document metadata
        with open(self.documents_file, 'r', encoding='utf-8') as f:
            documents_data = json.load(f)

        chunks = []
        for doc_data in documents_data:
            # Load FAISS index for this document if it exists
            faiss_file = self.indices_dir / f"{doc_data['id']}.faiss"
            if faiss_file.exists():
                try:
                    index = faiss.read_index(str(faiss_file))

                    # Extract chunk texts and metadata
                    for i in range(len(doc_data.get('chunks', []))):
                        chunk_data = doc_data['chunks'][i]

                        chunk = Chunk(
                            id=chunk_data.get('id', f"{doc_data['id']}_{i}"),
                            source=doc_data['filename'],
                            jurisdiction=doc_data.get('jurisdiction', 'federal'),
                            text=chunk_data['text'],
                            embedding=None,  # Will be computed
                            metadata=chunk_data.get('metadata', {})
                        )
                        chunks.append(chunk)

                except Exception as e:
                    print(f"Error loading FAISS index for {doc_data['id']}: {e}")
                    continue

        return chunks

    def generate_embeddings(self) -> List[np.ndarray]:
        """Generate embeddings for all chunks."""
        print("Generating embeddings...")
        texts = [chunk.text for chunk in self.chunks]

        # Generate embeddings in batches to avoid memory issues
        batch_size = 100
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(batch_texts, normalize_embeddings=True)
            embeddings.extend(batch_embeddings)
            print(f"Processed {min(i + batch_size, len(texts))}/{len(texts)} chunks")

        return embeddings

    def create_vectorize_data(self, embeddings: List[np.ndarray]) -> List[Dict[str, Any]]:
        """Prepare data for Vectorize upload."""
        vectorize_data = []

        for i, chunk in enumerate(self.chunks):
            vectorize_data.append({
                'id': chunk.id,
                'values': embeddings[i].tolist(),
                'metadata': {
                    'source': chunk.source,
                    'jurisdiction': chunk.jurisdiction,
                    'text': chunk.text[:1000],  # Limit text length for metadata
                }
            })

        return vectorize_data

    def create_d1_schema(self):
        """Generate D1 schema for chunk metadata."""
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

        # Write schema to file for manual execution
        schema_file = Path(__file__).parent.parent / "d1-schema.sql"
        with open(schema_file, 'w', encoding='utf-8') as f:
            f.write(schema)

        print(f"D1 schema written to {schema_file}")
        print("Execute this schema in your D1 database before running the migration.")

    def generate_migration_script(self, vectorize_data: List[Dict[str, Any]]):
        """Generate JavaScript migration script for uploading to Cloudflare."""
        script_content = f'''// Migration script to upload vectors to Cloudflare Vectorize and D1
// Generated by migrate-to-cloudflare.py

const vectors = {json.dumps(vectorize_data, indent=2)};

export default {{
  async fetch(request, env) {{
    // Upload to Vectorize in batches
    const batchSize = 100;
    let uploaded = 0;

    for (let i = 0; i < vectors.length; i += batchSize) {{
      const batch = vectors.slice(i, i + batchSize);

      try {{
        await env.VECTORIZE_INDEX.upsert(batch);
        uploaded += batch.length;
        console.log(`Uploaded {{uploaded}}/{{vectors.length}} vectors`);
      }} catch (error) {{
        console.error(`Error uploading batch starting at index ${{i}}:`, error);
        return new Response(`Error at batch ${{i}}: ${{error.message}}`, {{ status: 500 }});
      }}
    }}

    // Insert metadata into D1
    for (const vector of vectors) {{
      try {{
        await env.DATABASE.prepare(
          `INSERT OR REPLACE INTO chunks (id, source, jurisdiction, text, metadata)
           VALUES (?, ?, ?, ?, ?)`
        ).bind(
          vector.id,
          vector.metadata.source,
          vector.metadata.jurisdiction,
          vector.metadata.text,
          JSON.stringify(vector.metadata)
        ).run();
      }} catch (error) {{
        console.error(`Error inserting chunk ${{vector.id}}:`, error);
      }}
    }}

    return new Response(`Migration complete! Uploaded ${{uploaded}} vectors and metadata.`);
  }}
}};
'''

        script_file = Path(__file__).parent.parent / "workers" / "migrate-vectors.js"
        with open(script_file, 'w', encoding='utf-8') as f:
            f.write(script_content)

        print(f"Migration script written to {script_file}")
        print("\nTo complete the migration:")
        print("1. Deploy the migration script: wrangler deploy workers/migrate-vectors.js")
        print("2. Run the migration: curl https://your-worker-url")
        print("3. Verify data in Vectorize and D1 dashboards")

    def run_migration(self):
        """Run the complete migration process."""
        print("🚀 Starting Cloudflare migration...")

        # Step 1: Create D1 schema
        print("\n📋 Step 1: Creating D1 schema...")
        self.create_d1_schema()

        # Step 2: Generate embeddings
        print("\n🧮 Step 2: Generating embeddings...")
        embeddings = self.generate_embeddings()

        # Step 3: Prepare Vectorize data
        print("\n📦 Step 3: Preparing Vectorize data...")
        vectorize_data = self.create_vectorize_data(embeddings)

        # Step 4: Generate migration script
        print("\n📝 Step 4: Generating migration script...")
        self.generate_migration_script(vectorize_data)

        print("\n✅ Migration preparation complete!")
        print(f"   - Prepared {len(vectorize_data)} vectors for upload")
        print("   - Generated migration script in workers/migrate-vectors.js")
        print("   - Created D1 schema in d1-schema.sql")

        print("\n📋 Next steps:")
        print("1. Create Vectorize index: wrangler vectorize create thelawsays-vectorize --dimensions=384 --metric=cosine")
        print("2. Execute D1 schema in your database")
        print("3. Deploy migration script: wrangler deploy workers/migrate-vectors.js --name migrate-vectors")
        print("4. Run migration: curl https://migrate-vectors.your-subdomain.workers.dev")
        print("5. Update wrangler.toml with correct Vectorize index and D1 database IDs")


def main():
    try:
        migrator = CloudflareMigrator()
        migrator.run_migration()
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()