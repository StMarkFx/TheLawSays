# Building RAG Pipelines: A Comprehensive Guide

## Introduction to RAG (Retrieval-Augmented Generation)

Retrieval-Augmented Generation (RAG) is a paradigm that combines the strengths of information retrieval systems with generative AI. Instead of training a large language model on all knowledge, RAG retrieves relevant information from a knowledge base and uses it to generate responses.

RAG is particularly powerful for:
- Domain-specific question answering (like legal, medical, technical docs)
- Reducing hallucinations in AI responses
- Providing cited/explainable answers
- Knowledge bases that change/update over time

## Technologies Used in This Project & Their WHYs

### 1. **Apache Tika + PyTesseract (OCR)**
**WHAT**: Automatic text extraction from PDFs with OCR fallback
**WHY**:
- Legal documents are often PDFs with complex formatting
- OCR ensures no knowledge is lost from scanned/image-based PDFs
- Tika handles native text PDFs efficiently, OCR as fallback

**Alternatives**:
- **PDFMiner/PyPDF2**: Free, no Java required, but less robust for complex PDFs
- **AWS Textract/OCR**: Cloud-based, better accuracy, higher cost
- **Docling**: Meta's new PDF processor, handles tables/charts better

### 2. **spaCy (en_core_web_sm) for Text Processing**
**WHAT**: NLP library for sentence boundary detection and text chunking
**WHY**:
- Accurate sentence splitting crucial for legal document chunking
- Lightweight compared to NLTK
- Fast processing for large document collections

**Alternatives**:
- **NLTK**: More features, but slower and heavier
- **Transformers tokenizers**: Can be used for subword tokenization
- **Rule-based splitting**: Simple but less accurate for legal text

### 3. **Sentence Transformers (MiniLM-L6-v2)**
**WHAT**: Transformer-based sentence embedding model, 384D vectors
**WHY**:
- **Balance of quality vs speed**: Small model (~23MB) but strong performance
- **Efficiency**: Fast inference, suitable for CPU deployment
- **Semantic search**: Captures meaning beyond keywords

**Alternatives based on compute**:
- **Small models (< 100MB)**: all-MiniLM-L6-v2, paraphrase-MiniLM, GIST-small
- **Medium models (100MB-500MB)**: all-MPBNet-base, all-distilroberta-v1
- **Large models (> 500MB)**: all-roberta-large, text2vec-large (better quality)
- **Multilingual**: Newer models like multilingual-e5-large for non-English docs

### 4. **FAISS (Facebook AI Similarity Search)**
**WHAT**: Efficient similarity search and clustering for dense vectors
**WHY**:
- **Speed**: Optimized for high-dimensional vector search (IndexFlatIP for L2 distance)
- **Scalability**: Can handle millions of vectors efficiently
- **Algorithm**: Inner product search for normalized embeddings = cosine similarity
- **Memory efficient**: Compressed indices for production

**Alternatives**:
- **Pinecone/Weaviate**: Managed vector databases with metadata filtering
- **Chroma**: Open-source, Python-native
- **Qdrant**: Rust-based, high performance
- **Milvus/Zilliz**: Enterprise-scale vector search

### 5. **BM25 (Okapi BM25)**
**WHAT**: Advanced TF-IDF algorithm with term frequency saturation
**WHY**:
- **Lexical search**: Catches exact matches and keyword-based queries
- **Complementary to embeddings**: Semantic gaps filled by keyword matching
- **Established effectiveness**: Proven in information retrieval literature

**Alternatives**:
- **TF-IDF**: Simpler but less effective
- **TF-IDF with sublinear scaling**: Manual BM25-like saturation
- **Hybrid approaches**: Learned sparse retrieval (Splade)

### 6. **Hybrid Search Strategy**
**WHAT**: Combining BM25 and FAISS results with weighted scoring
**WHY**:
- **Best of both worlds**: Semantic understanding + exact keyword matching
- **Robustness**: Handles different query types (factual vs conceptual)
- **Normalization tricks**: BM25 scores normalized by max score for combination
- **Alpha parameter**: Controls dense vs sparse weight (0.65 for semantic focus)

### 7. **OpenAI GPT-4o-mini**
**WHAT**: Lightweight GPT model (8B parameters) for generation
**WHY**:
- **Cost-effective**: Cheaper than larger models
- **Good quality**: Good instruction following for legal analysis
- **Structured output**: Useful for citation formatting and legal writing
- **Temperature 0.2**: Low randomness for consistent legal answers

**Alternatives**:
- **Open-source**: Mistral-7B, Llama-2-13B (privacy, cost)
- **Larger models**: GPT-4, Claude-2 (better reasoning, more expensive)
- **Legal-specific**: Specialized legal LLMs when available

## Architecture Decisions Based on Scale & Resources

### Small Scale (What This Project Uses)
- **Compute**: CPU-only, single machine
- **Documents**: ~25 PDFs, 10k+ chunks
- **Users**: Single user or small team
- **Cost**: Free/cheap hosting

**Why suitable**:
- MiniLM: Fast inference, good quality-cost ratio
- FAISS: Simple index, no server required
- Streamlit: Easy deployment, no infrastructure needed

### Medium Scale (10k-100k documents)
**Changes needed**:
- **Indexing**: Segment larger collections, use IVF indices in FAISS
- **Storage**: PostgreSQL with pgvector or cloud vector DB
- **Compute**: GPU acceleration for embedding generation
- **Serving**: FastAPI/Docker instead of Streamlit

### Large Scale (100k+ documents, concurrent users)
**Scaling considerations**:
- **Vector DB**: Pinecone, Weaviate, or managed FAISS
- **Preprocessing**: Distributed chunking (Spark/Dask)
- **Serving**: Kubernetes deployment with load balancing
- **Caching**: Redis for query result caching
- **Monitoring**: Retrieval quality metrics, latency tracking

## Technical Q&As for RAG Mastery

### **Q: Why normalize embeddings for cosine similarity?**
A: Normalization enables dot product = cosine similarity, which is faster and more numerically stable than explicit cosine calculation.

### **Q: Why combine BM25 with dense retrieval?**
A: Dense retrieval excels at semantic matching but can miss exact keyword matches. BM25 catches these gaps while dense retrieval handles fuzzy/conceptual queries.

### **Q: How to choose chunk size?**
A: Trade-off between context preservation vs retrieval precision. Legal texts: 200-800 tokens. Too small: loses context; too large: dilutes relevance.

### **Q: Why use inner product vs L2 distance in FAISS?**
A: With normalized embeddings, inner product = cosine similarity. Cosine similarity is better for text similarity than Euclidean distance.

### **Q: What's the alpha parameter controlling?**
A: α controls semantic vs lexical weight. α=0.7 means 70% semantic, 30% lexical. Tune based on query type.

### **Q: Why not use embedding similarity alone?**
A: Embeddings can hallucinate superficial similarities. BM25 provides grounded, exact-match verification.

### **Q: How to handle jurisdiction/domain filtering?**
A: Pre-filter candidates by metadata before vector search. Reduces noise and improves relevance.

### **Q: Why use prompt engineering vs fine-tuning?**
A: Fine-tuning requires expensive training data. Prompt engineering is faster, adaptable, and works with black-box APIs.

### **Q: What's the issue with duplicate chunks?**
A: MD5 deduplication prevents wasted storage and biased retrieval (same passage over-represented).

### **Q: Why sentence-level chunking for legal text?**
A: Legal documents have precise sentence boundaries. Sentence units preserve legal concepts better than arbitrary word windows.

## Common RAG Pitfalls to Avoid

1. **No evaluation**: Always measure retrieval quality (precision@K, recall)
2. **Poor chunking**: Test different strategies on your domain
3. **Single retrieval method**: Hybrid approaches almost always better
4. **No user feedback**: Add thumbs up/down for continuous improvement
5. **Static knowledge base**: Design for knowledge base updates
6. **No latency consideration**: Optimize for response time expectations

## Future RAG Trends to Watch

- **Instruction-tuned retrievers**: Models that learn retrieval from interactions
- **Long-context generation**: 32k+ context windows reducing retrieval needs
- **Multi-vector retrieval**: Multiple embeddings per chunk for better coverage
- **Advanced reranking**: Cross-encoders for post-retrieval relevance scoring
- **Personalized retrieval**: User preference-based reranking

Remember: The best RAG implementations combine domain expertise with engineering best practices.
