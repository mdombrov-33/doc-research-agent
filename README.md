# Document Research Agent

An adaptive agentic RAG service. Upload documents, ask questions, get streamed answers
grounded in your files — with a live web search mixed in when the question needs it.

Built on a **LangGraph** state machine with **hybrid retrieval** (dense + BM25), a
**cross-encoder reranker**, LLM **document grading** with a **web-search fallback**, SSE token
**streaming**, and LLM-based **guardrails**.

**Frontend:** Streamlit · **Backend:** FastAPI on GCP Cloud Run · **Vector DB:** Qdrant

> **Full reference:** [`docs/architecture.md`](docs/architecture.md) explains every part —
> ingestion, retrieval, the graph and its custom reducers, grading, fallbacks, streaming,
> guardrails, memory, evaluation, and deployment. Start there to understand how it works.

## Screenshots

![Document Research Agent UI](assets/1.png)

![Qdrant vector store dashboard](assets/2.png)

## How it works (at a glance)

```
POST /api/stream
      │
   Guardrails (input: moderation ‖ injection)
      │
   router ──► retrieve (+ websearch in parallel?) ──► grade_documents ⇄ websearch ──► generate
      │                                                  (fallback if <2 relevant)      │
   Guardrails (output: moderation, best-effort)                          streamed tokens ┘
      │
   SSE → client   +   telemetry recorded
```

- **Router** — one LLM call classifies the query (documents vs documents+web) and rewrites it
  for search.
- **Retrieve** — always runs. Hybrid dense + BM25 search in Qdrant (fused with RRF), then a
  cross-encoder reranks a wide candidate pool down to your `top_k`.
- **Grade** — an LLM scores each candidate yes/no; if too little is relevant, the graph falls
  back to web search and merges the results.
- **Generate** — answers from the graded context and streams tokens over SSE; conversation
  history is kept per session.

See [`docs/architecture.md`](docs/architecture.md) for the state machine, the parallel/fallback
web-search paths, and the custom state reducers.

## Quickstart

```bash
cp .env.example .env     # fill in OPENAI_API_KEY, OPENROUTER_API_KEY, Qdrant settings
make up                  # boot Qdrant + the API via docker compose
make ui                  # run the Streamlit UI (separate terminal)
```

Local dev without Docker:

```bash
make install             # uv sync
make dev                 # uvicorn with reload
```

Useful targets: `make test`, `make eval` (full offline eval), `make eval-retrieval` (the
CI retrieval gate), `make lint`, `make format`.

## Configuration

Every field in `src/config.py` can be set via an env var of the same name; `.env` only needs
the values that differ per environment. See [`.env.example`](.env.example) for the annotated
list. Keys:

- `OPENAI_API_KEY` — embeddings (`text-embedding-3-small`) **and** the guardrails moderation API
- `OPENROUTER_API_KEY` — all chat/LLM calls (routing, grading, generation)
- `QDRANT_MODE` — `local` (Docker) or `cloud` (uses `QDRANT_CLOUD_URL` + `QDRANT_API_KEY`)

## API

| Endpoint | Purpose |
|---|---|
| `POST /api/stream` | RAG query, streamed via SSE (`{question, session_id?, model?, top_k?}`) |
| `POST /api/upload` | ingest a document (`.pdf` / `.docx` / `.txt`) |
| `GET /api/monitoring/stats` | live query telemetry |
| `GET /health` | liveness |

## Evaluation

`evals/` holds an offline quality gate that runs the real pipeline against a labelled golden
set and scores retrieval, generation, and embeddings (details in
[`evals/README.md`](evals/README.md)). The deterministic retrieval tier runs in CI on every
push to main.

## Tech stack

**LangGraph** · **LangChain** · **Qdrant** (hybrid dense + BM25) · **FastEmbed** (BM25 sparse +
cross-encoder reranker) · **FastAPI** · **OpenAI** (embeddings + moderation) · **OpenRouter**
(LLMs) · **PyMuPDF** · **spaCy** · **Streamlit** · **Docker** · **Terraform**
