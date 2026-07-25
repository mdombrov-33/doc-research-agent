# Document Research Agent

## Product

Document-research application. A user uploads PDF, DOCX, or text documents, asks a question, and
receives a streamed answer grounded in retrieved document chunks. The agent may use web search
when the document corpus is insufficient.

## Stack

- Python 3.13, FastAPI, Streamlit, LangGraph, LangChain, Qdrant, OpenRouter, OpenAI embeddings.
- Docker for local/runtime packaging; GCP Terraform exists for deployment.
- Pytest, Ruff, and mypy for verification.

## Repository layout

- `ui.py` — Streamlit client.
- `src/api/` — HTTP routes, upload endpoint, SSE streaming, rate limiting, middleware.
- `src/core/agent/` — LangGraph workflow, nodes, tools, prompts, and state.
- `src/core/ingestion/` — document extraction, chunking, enrichment, and indexing.
- `src/core/retrieval/` — hybrid retrieval and reranking.
- `src/core/monitoring/` — query metrics with SQLite local storage or shared Postgres storage.
- `evals/` — offline retrieval and generation evaluation harness.
- `tests/` — unit and integration tests.
- `terraform/gcp/` — GCP infrastructure.

## Runtime flow

```text
Streamlit -> FastAPI -> LangGraph -> Qdrant retrieval
                              -> optional web search
                              -> OpenRouter chat model

Ingestion: upload -> extract -> chunk -> enrich -> Qdrant
```

## Glossary

- **Answer cache** — stored final answers served for repeated context-free questions instead
  of re-running the agent. Only document-grounded answers are cacheable.
- **Corpus version** — a counter identifying the state of the document corpus; it advances
  when a new document is added. Cached answers belong to the version they were produced under.
- **Conversational outcome** — a query resolved with a fixed capabilities-style reply
  (greeting, thanks, "what can you do") without consulting documents or the web. Distinct
  from an abstention.
- **Rerank floor** — a calibrated relevance threshold below which retrieved chunks are
  excluded from evidence.
- **Namespace** — partition key for cached/derived data. The app is single-tenant today, so
  there is exactly one namespace ("default"); the field exists so tenants can be added later.

## Commands

- `make test`
- `make lint`
- `make format`
- `make eval-retrieval`
- `make eval` — requires configured external services and API keys.
