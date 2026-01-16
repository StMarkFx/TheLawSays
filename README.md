# TheLawSays

TheLawSays is a modern AI legal assistant for Nigerian statutes. It delivers citation-backed answers using a Cloudflare-native RAG stack and a clean, mobile-friendly UI.

---

## Highlights

- **Edge-first backend**: Cloudflare Workers + Workers AI + Vectorize.
- **Fast, cited answers**: Retrieval with top chunks and citations surfaced in the UI.
- **Clean UX**: Pinned header, inline composer, slide-in sidebar, and a light/dark theme toggle.
- **Mobile-ready**: Expo React Native app mirrors the web experience.

---

## Architecture

### Cloudflare (Production)
```
PDF statutes -> build_index.py -> data/documents.json
                                  |
                                  v
                        Vectorize (embeddings)
                                  |
                                  v
                     Cloudflare Workers (RAG)
                                  |
                                  v
                     Cloudflare Pages (Next.js)
```

### Local Development (Optional)
```
Next.js (web/) -> FastAPI (api/) -> OpenAI (local/dev)
```

---

## Project Layout

```
api/               FastAPI backend (local/dev only)
web/               Next.js frontend (Cloudflare Pages)
workers/           Cloudflare Workers backend
mobile/            Expo React Native app
scripts/           Build and migration scripts
data/              Generated artifacts (documents, indices)
config/            Environment templates
```

---

## Quick Start (Cloudflare)

### 1) Create Cloudflare resources

```bash
wrangler vectorize create thelawsays-vectorize --dimensions=384 --metric=cosine
wrangler d1 create thelawsays-db
wrangler kv namespace create "LAWS_KV"
```

### 2) Generate and upload embeddings

```bash
python scripts/migrate-to-cloudflare.py
wrangler vectorize upsert thelawsays-vectorize --file vectorize.ndjson --batch-size 500

# Optional: store metadata in D1 for secondary lookups
wrangler d1 execute thelawsays-db --remote --file d1-schema.sql
```

### 3) Deploy backend (Workers)

```bash
wrangler deploy
```

### 4) Deploy frontend (Pages)

```bash
cd web
npm install
npm run build
npx wrangler pages deploy dist --project-name thelawsays-frontend
```

---

## Environment Variables

### Web

Create `web/.env.local`:
```
NEXT_PUBLIC_API_BASE_URL=https://thelawsays-backend.thelawsays.workers.dev
```

### Mobile

Create `mobile/.env`:
```
EXPO_PUBLIC_API_BASE_URL=https://thelawsays-backend.thelawsays.workers.dev
```

---

## Run Locally (Optional)

```bash
# Terminal 1 (FastAPI)
uvicorn api.main:app --reload --port 8000

# Terminal 2 (Web)
cd web
npm install
npm run dev
```

---

## Mobile App

```bash
cd mobile
npm install
npm run start
```

---

## Testing

```bash
pytest                  # Core + FastAPI tests
cd web && npm run lint  # ESLint + TypeScript
cd web && npm run test  # Vitest component suite
```

---

## Notes

- The Cloudflare Workers backend is the production path.
- FastAPI is kept for local development and legacy use.
- Vectorize uses 384-dim embeddings (all-MiniLM-L6-v2).
- D1 lookups can be toggled with `USE_D1` in `wrangler.toml`.

---

Built with Cloudflare Workers, Vectorize, D1, Next.js, Expo, and FastAPI.
