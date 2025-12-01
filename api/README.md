# TheLawSays FastAPI Service

This service powers the production backend for TheLawSays. It exposes retrieval-augmented chat endpoints consumed by the Next.js frontend.

## Endpoints

| Method | Path          | Description                          |
| ------ | ------------- | ------------------------------------ |
| GET    | `/v1/health`  | Healthcheck for uptime monitoring.   |
| POST   | `/v1/chat`    | Submit a user message and receive a cited answer. |
| POST   | `/v1/feedback`| Record thumbs-up/down feedback (logged only). |

### `/v1/chat` payload

```json
{
  "message": "What does the law say about fraud?",
  "history": [
    {"role": "assistant", "content": "Hello! Ask about Nigerian law."}
  ],
  "jurisdiction": "Federal",
  "top_k": 5
}
```

Response:

```json
{
  "answer": "According to Section 419 ...",
  "chunks": [
    {
      "id": 12,
      "source": "CriminalAct.pdf",
      "jurisdiction": "Federal",
      "text": "Any person who by any false pretence...",
      "score": 0.74
    }
  ],
  "retrieval_used": true,
  "metadata": {
    "jurisdiction": "Federal",
    "intent_label": "legal_lookup"
  }
}
```

## Running Locally

```bash
pip install -r api/requirements.txt
cp api/.env.example api/.env
uvicorn api.main:app --reload --port 8000
```

Make sure the knowledge base files (`documents.json`, `legal_index.faiss`, `bm25_index.pkl`) exist in the repository root.

## Testing

```bash
pytest
```

## Environment Variables

| Key              | Default        | Notes                                         |
| ---------------- | -------------- | --------------------------------------------- |
| `OPENAI_API_KEY` | —              | Required for generating answers.              |
| `OPENAI_MODEL`   | `gpt-4o-mini`  | Override to use another OpenAI chat model.    |
| `RETRIEVAL_TOP_K`| `5`            | Number of candidate chunks analysed.          |
| `RETRIEVAL_ALPHA`| `0.65`         | Hybrid weighting between FAISS/BM25 results.  |
| `ALLOW_ORIGINS`  | `http://localhost:3000` | Allowed CORS origins for the frontend. |
| `ENVIRONMENT`    | `development`  | Optional label for logs/metrics.              |
