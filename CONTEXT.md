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
- `src/core/monitoring/` — query metrics and local SQLite storage.
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

## Commands

- `make test`
- `make lint`
- `make format`
- `make eval-retrieval`
- `make eval` — requires configured external services and API keys.
