# Document Research Agent

An agentic RAG service. Upload documents, ask questions, get streamed answers grounded in your
files — with the agent reaching for a live web search on its own when your documents don't
cover the question.

Built on a **LangGraph** agent (a **ReAct** tool-calling loop), **hybrid retrieval** (dense +
BM25), a **cross-encoder reranker**, persistent **conversation memory**, SSE token
**streaming**, local **guardrails**, and per-IP **rate limiting**.

**Frontend:** Streamlit · **Backend:** FastAPI on GCP Cloud Run · **Vector DB:** Qdrant

> **Full reference:** [`docs/architecture.md`](docs/architecture.md) explains every part —
> ingestion, retrieval, the agent (its tools, state, and the tool-call cap), the web-search
> fallback, streaming, guardrails, memory, evaluation, and deployment. Start there to
> understand how it works.

## Screenshots

![Document Research Agent UI](assets/1.png)

![Qdrant vector store dashboard](assets/2.png)

## How it works (at a glance)

```
POST /api/stream
      │
   Guardrails (input: llm-guard Toxicity + PromptInjection, local)
      │
   ┌─── LangGraph agent (ReAct loop) ───────────────────┐
   │   agent ──► tools ──► post_tools ──┐                │
   │     ▲                              │   loop until   │
   │     └──────────────────────────────┘   cap or done  │
   │     └──► END (answer)                               │
   └─────────────────────────────────────────────────────┘
      │
   streamed answer tokens → SSE → client   +   telemetry recorded
```

- **Agent** — a tool-calling LLM decides each step: search the documents, search the web, or
  write the answer. Its trajectory isn't fixed — it's chosen per question.
- **Tools** — `retrieve_documents` runs hybrid dense + BM25 search in Qdrant (fused with RRF),
  then a cross-encoder reranks a wide candidate pool down to your `top_k`; `web_search` hits
  the live web.
- **Post-tools** — increments a per-turn tool-call counter; at the hard cap (4 calls) it
  injects a stop message that forces the agent to write its final answer immediately.
- **Answer** — once the agent has enough context (or hits the cap) it writes the answer and
  streams it over SSE; conversation history is persisted per session.

See [`docs/architecture.md`](docs/architecture.md) for the agent topology, the tools and state,
the web-search fallback walkthrough, and memory.

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

- `OPENAI_API_KEY` — embeddings (`text-embedding-3-small`) only
- `OPENROUTER_API_KEY` — all chat/LLM calls (the agent's reasoning + answer)
- `QDRANT_MODE` — `local` (Docker) or `cloud` (uses `QDRANT_CLOUD_URL` + `QDRANT_API_KEY`)

Guardrails run locally (llm-guard) and need no API key. Local conversation memory and live
telemetry persist to SQLite under `DATA_DIR` (default `./data`); production conversation memory
uses the configured Postgres checkpointer.

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

**LangGraph** (ReAct agent + SQLite/Postgres checkpointer) · **LangChain** · **Qdrant** (hybrid dense +
BM25) · **FastEmbed** (BM25 sparse + cross-encoder reranker) · **llm-guard** (local input
guardrails) · **FastAPI** · **slowapi** (rate limiting) · **OpenAI** (embeddings) ·
**OpenRouter** (LLMs) · **PyMuPDF** · **spaCy** · **Streamlit** · **Docker** · **Terraform**
