graph TD
    A[laws_pdfs/*.pdf] --> B[Tika + OCR]
    B --> C[Chunk + Clean]
    C --> D[documents.json]
    C --> E[BM25 → bm25_index.pkl]
    C --> F[MiniLM → embeddings → FAISS]
    F --> G[legal_index.faiss compressed]
    
    H[User Query] --> I[Hybrid: FAISS + BM25]
    I --> J[Retrieved chunks from documents.json]
    J --> K[Prompt + OpenAI]
    K --> L[Answer with citation]