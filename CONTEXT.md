# Document Research Agent

## Product

Document-research application. A user uploads PDF, DOCX, or text documents, asks a question, and
receives a streamed answer grounded in retrieved document chunks. The agent may use web search
when the document corpus is insufficient.

## Language

**Document-grounded answer**:
An answer whose claims are supported by evidence from the user's document corpus.
_Avoid_: Document answer, RAG answer

**Web-grounded answer**:
An answer whose claims are supported by web evidence used because the document corpus was
insufficient.
_Avoid_: Web fallback

**Abstention**:
A response that declines to answer because neither document nor web evidence is sufficient.
_Avoid_: Failure, empty answer

**Evaluation corpus**:
A fixed collection of representative documents used to measure research quality.
_Avoid_: Golden corpus, test documents

**Evaluation case**:
A curated user scenario with an expected outcome and, when applicable, expected evidence and
answer.
_Avoid_: Golden question, test query

**Fact ledger**:
The authoritative set of synthetic facts from which the evaluation corpus and its expected
answers are derived.
_Avoid_: Ground-truth document, synthetic prompt

**Required evidence**:
Evidence containing facts without which an evaluation case cannot be answered correctly.
_Avoid_: Relevant document

**Supporting evidence**:
Evidence that corroborates or enriches an answer but is not necessary to answer correctly.
_Avoid_: Optional required document

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

## Commands

- `make test`
- `make lint`
- `make format`
- `make eval-retrieval`
- `make eval` — requires configured external services and API keys.
