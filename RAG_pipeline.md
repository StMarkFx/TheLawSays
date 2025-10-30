graph TD
    A[laws_pdfs/*.pdf] --> B[Tika + OCR]
    B --> C[Chunk + Clean]
    C --> D[documents.json]
    C --> E[BM25 → bm25_index.pkl]
    C --> F[MiniLM → embeddings → FAISS]
    F --> G[legal_index.faiss compressed]

    H[User Query] --> I{Conversational Query?}

    I -->|YES "hi", "thanks"| J[Simple Response]
    J --> K[Answer with greeting]

    I -->|NO Legal Question| L[Hybrid: FAISS + BM25]
    L --> M[Retrieved chunks from documents.json]
    M --> N[Prompt + OpenAI + Context]
    N --> O[Answer with legal citations]

    K --> P[No RAG execution]
    O --> Q[RAG execution = citations shown]
