# TheLawSays

Modern AI legal assistant for Nigerian statutes with cited answers powered by a hybrid Retrieval-Augmented Generation (RAG) pipeline.

- **Backend** – Cloudflare Workers with Vectorize + Workers AI (`workers/`) providing serverless API endpoints.
- **Frontend** – Next.js App Router (`web/`) deployed on Cloudflare Pages with dark-first UI.
- **Core** – `thelawsays_core/` owns ingestion, retrieval, intent classification, and prompting logic.
- **Legacy demo** – `app.py` Streamlit playground (kept for workshops and demos).

---

## Product Snapshot

- **Pinned chrome**: Header and composer stay fixed while the conversation scrolls independently.
- **Slide-in sidebar**: Project info + “About St. Mark” content, identical across desktop and mobile.
- **Auto retrieval settings**: Frontend always sends the conversation only; backend defaults to `jurisdiction=auto` and `topK=5`.
- **RAG awareness**: Citations accordion appears only when retrieval is triggered and shows per-answer excerpts.
- **Loading UX**: Skeleton hero while bootstrapping and an inline “Thinking…” spinner for in-flight answers.

---

## Architecture Overview

### Development Architecture
```
PDF statutes --> build_index.py --> data/documents.json
                                  |                  data/indices/legal_index.faiss
                                  |                  data/indices/bm25_index.pkl
                                  v
                          thelawsays_core/
                                  |
 Next.js (web/) --> FastAPI (api/) --> OpenAI Chat Completions
                        ^
                        └── Intent classifier decides when to trigger retrieval
```

### Production Architecture (Cloudflare)
```
PDF statutes --> build_index.py --> Cloudflare Vectorize (embeddings)
                                  |                  Cloudflare D1 (metadata)
                                  v
                        Cloudflare Workers (JavaScript/TypeScript)
                                  |
 Next.js --> Cloudflare Pages --> Workers AI + Vectorize queries
                        ^
                        └── Intent classifier decides when to trigger retrieval
```

---

## Getting Started

### 1. Build the Knowledge Base

```bash
# inside repo root
pip install -r api/requirements.txt
python -m spacy download en_core_web_sm
python build_index.py
```

Place PDFs in `laws/federal` and `laws/lagos` before running `build_index.py`. The script emits `data/documents.json`, `data/indices/legal_index.faiss`, and `data/indices/bm25_index.pkl`.

#### Required artifacts for deployment

| File | Approx size | Purpose | Notes |
| --- | --- | --- | --- |
| `data/documents.json` | ~6.5 MB | Serialized chunk metadata/content | Must reside in `data/` directory or mounted volume |
| `data/indices/legal_index.faiss` | ~19 MB | Dense vector index for FAISS search | Requires CPU build of FAISS present on host |
| `data/indices/bm25_index.pkl` | ~5 MB | Sparse BM25 index for hybrid retrieval | Loaded alongside FAISS during startup |

These assets are git-ignored; upload them to your hosting volume (e.g., Railway persistent disk, S3 + download step) so FastAPI can load them when `load_knowledge_base()` runs.

### 2. Configure Environment Variables

```bash
cp config/.env.example .env
cp config/api.env.example api/.env
cp config/web.env.example web/.env.local
```

| Variable | Location | Notes |
| --- | --- | --- |
| `OPENAI_API_KEY` | `.env`, `api/.env`, `app.py` | Required for FastAPI + Streamlit |
| `OPENAI_MODEL` | `api/.env` | Defaults to `gpt-4o-mini`, set to any chat-capable model |
| `NEXT_PUBLIC_API_BASE_URL` | `web/.env.local` | Defaults to `http://localhost:8000` |
| `ALLOW_ORIGINS` | `api/.env` | Comma-separated CORS origins (include Railway/Vercel URLs) |
| `RETRIEVAL_TOP_K` | `api/.env` | Overrides default chunk count (5) |
| `RETRIEVAL_ALPHA` | `api/.env` | Hybrid FAISS/BM25 weighting (0-1, default 0.65) |
| `ENABLE_MODERATION` | `api/.env` | Toggle OpenAI moderation guard (`true`/`false`) |
| `ENVIRONMENT` | `api/.env` | `development`, `test`, or `production` |

### 3. Run Everything

```bash
python scripts/dev.py
```

This boots FastAPI (`localhost:8000`) and Next.js (`localhost:3000`). Prefer manual control?

```bash
# Terminal 1
uvicorn api.main:app --reload --port 8000

# Terminal 2
cd web
npm install          # first run only
npm run dev
```

### 4. Optional Streamlit Demo

```bash
streamlit run app.py
```

The Streamlit UI consumes `thelawsays_core`, so answers match the production stack.

---

## Cloudflare Deployment

### Prerequisites
1. [Cloudflare Account](https://cloudflare.com) with Workers subscription
2. [Wrangler CLI](https://developers.cloudflare.com/workers/wrangler/install-and-update/): `npm install -g wrangler`
3. Authenticate: `wrangler auth login`

### Setup Cloudflare Resources

```bash
# Create Vectorize index for embeddings
wrangler vectorize create thelawsays-vectorize --dimensions=384 --metric=cosine

# Create D1 database for metadata
wrangler d1 create thelawsays-db

# Create KV namespace for caching (optional)
wrangler kv:namespace create "LAWS_CACHE"
```

### Deploy to Cloudflare

```bash
# Deploy backend (Workers)
wrangler deploy

# Deploy frontend (Pages)
cd web
npx wrangler pages deploy dist --compatibility-date 2024-01-01
```

### Environment Variables

Set these in Cloudflare Dashboard > Workers > Your Worker > Settings > Variables:

| Variable | Value | Notes |
| --- | --- | --- |
| `CORS_ORIGINS` | `https://thelawsays.pages.dev,https://thelawsays.com` | Comma-separated allowed origins |
| `ENVIRONMENT` | `production` | Environment mode |
| `VECTORIZE_INDEX` | `thelawsays-vectorize` | Your Vectorize index name |
| `D1_DATABASE` | `thelawsays-db` | Your D1 database name |

### Migrate Knowledge Base

```bash
# Run locally to generate embeddings
python scripts/migrate-to-cloudflare.py

# This will:
# 1. Load existing FAISS indices
# 2. Generate embeddings for all chunks
# 3. Upload to Vectorize and D1
# 4. Update metadata in D1
```

---

## Testing

```bash
pytest                  # Core + FastAPI tests
cd web && npm run lint  # ESLint + TypeScript
cd web && npm run test  # Vitest component suite
```

---

## Project Layout

```
thelawsays_core/   Shared ingestion, retrieval, intent, and prompt helpers
api/               FastAPI app (routers, services, schemas, deps, tests)
web/               Next.js frontend (App Router, components, tests)
scripts/           Build and development scripts
data/              Generated artifacts (documents, indices, logs)
data/indices/      FAISS and BM25 index files
data/logs/         Build logs and temporary files
config/            Environment configuration templates
docs/              Project documentation and assets
notes/             Personal notes and study materials
legacy/            Streamlit demo (kept for workshops)
laws/              Source PDFs (ignored in git)
```

Generated artifacts in `data/` are git-ignored and should be regenerated with `python scripts/build_index.py`.

---

## Troubleshooting

- **Indices missing / stale** – rerun `python build_index.py`.
- **`en_core_web_sm` errors** – install via `python -m spacy download en_core_web_sm`.
- **OpenAI failures** – verify API key, org, and quota.
- **CORS issues** – update `ALLOW_ORIGINS` in `api/.env`.
- **Frontend hitting wrong API** – confirm `NEXT_PUBLIC_API_BASE_URL`.
- **Large PDF ingestion** – ensure enough RAM or run ingestion on a larger machine.

---

## Additional Docs

- [`docs/upgrade.md`](upgrade.md) – UI/feature roadmap.
- [`docs/RAG_pipeline.md`](RAG_pipeline.md) – Retrieval + chunking internals.
- [`api/README.md`](api/README.md) – Backend endpoints + setup notes.
- [`web/README.md`](web/README.md) – Frontend architecture + commands.

---

Built with Tika, pdf2image, Tesseract, sentence-transformers, BM25, FAISS, FastAPI, Next.js, and OpenAI.
