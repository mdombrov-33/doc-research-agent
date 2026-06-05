# Document Research Agent

RAG system with LangGraph state machine, hybrid search, SSE streaming, and LLM-based guardrails.

**Frontend:** Streamlit | **Backend:** GCP Cloud Run | **Vector DB:** Qdrant Cloud

## Screenshots

![Document Research Agent UI](assets/1.png)

![Qdrant vector store dashboard](assets/2.png)

## Architecture

### System Overview

```
┌─────────────┐
│   FastAPI   │  REST API (upload, stream endpoints)
└──────┬──────┘
       │
       ▼
┌──────────────────┐
│   Guardrails     │  Input check (moderation API + injection classifier)
└──────┬───────────┘
       │ safe
       ▼
┌──────────────────┐
│  LangGraph Agent │  State machine → streams tokens via SSE
└──────┬───────────┘
       │ full response
       ▼
┌──────────────────┐
│   Guardrails     │  Output check (moderation API)
└──────────────────┘
```

### LangGraph Agent State Machine

```
                    ┌─────────┐
                    │  START  │
                    └────┬────┘
                         │
                         ▼
                   ┌───────────┐
                   │  Router   │ (Classify + rewrite query in one LLM call)
                   └─────┬─────┘
                         │
              ┌──────────┴──────────┐
              │                     │
           always            web_search=true
              │                     │
              ▼                     ▼
        ┌──────────┐          ┌──────────┐
        │ Retrieve │          │WebSearch │
        │(Hybrid)  │          │          │
        └────┬─────┘          └────┬─────┘
             │                     │
             └──────────┬──────────┘
                        │  (parallel fan-in)
                        ▼
                 ┌─────────────┐
                 │ Grade Docs  │ ◄──────┐ (Batch LLM: relevant?)
                 └──────┬──────┘        │
                        │               │ <2 relevant
                        │               │ & web not yet tried
                        │               │
                        ├───────────────┘
                        │  (fallback to WebSearch)
                        ▼
                   ┌──────────┐
                   │ Generate │ (Stream answer via SSE)
                   └────┬─────┘
                        │
                        ▼
                      ┌─────┐
                      │ END │
                      └─────┘
```

**Node Descriptions:**

- **Router**: LLM classifies query type and rewrites it for semantic search. If web search is needed, both branches run in parallel.
- **Retrieve**: Always runs. Hybrid search — dense vector + BM25 sparse, fused with Reciprocal Rank Fusion (RRF) in Qdrant, top-k results.
- **WebSearch**: Runs in parallel with Retrieve when router flags `web_search=true`.
- **Grade Docs**: Batch LLM grading over the merged result set from both sources. If fewer than 2 docs pass and web search hasn't run yet, loops back to WebSearch as a fallback.
- **Generate**: Synthesize answer from graded documents, stream tokens via SSE.

## How It Works

### 1. Document Upload & Processing

Documents (PDF, DOCX, TXT) are chunked with overlap, then stored in Qdrant with both a dense embedding (`text-embedding-3-small`, 1536 dimensions) and a BM25 sparse vector per chunk, alongside metadata (filename, page numbers, chunk index).

### 2. Query Flow

**Security (input check):**

- OpenAI Moderation API catches harmful/violent content
- `gpt-5.4-mini` classifier catches prompt injection, jailbreaks, system probing
- Blocked inputs return refusal immediately, before LangGraph runs

**Router Node:**

- Single LLM call: classifies as `vectorstore` or `websearch` AND rewrites query for semantic search
- Prompt instructs the model to route explicit phrases ("search web", "check online") to the web search path (best-effort, not a hard override)

**Retrieve Node (Hybrid Search):**

- Two independent retrievers over the full corpus, in a single Qdrant query:
  - Dense vector: cosine similarity over `text-embedding-3-small` embeddings
  - Sparse BM25: keyword ranking via a `langchain-sparse` vector with IDF computed server-side
- Fusion: Reciprocal Rank Fusion (RRF) applied by Qdrant; returns the top-k fused chunks

**Grade Documents Node:**

- Batch LLM grading over the merged result set from both retrieval sources
- Binary relevance scoring (yes/no) per document

**Generate Node:**

- Synthesizes answer from graded documents, streams tokens via SSE
- Includes `chat_history` for session-aware multi-turn responses

**Output check (post-streaming):**

- After streaming completes, full response is checked via OpenAI Moderation API
- If flagged, a correction event is sent to the client replacing the streamed content

### 3. Streaming (SSE)

`POST /api/stream` returns `text/event-stream`. Each event is a JSON object:

```
data: {"token": "partial text"}        # during generation
data: {"done": true, "sources_count": 5, "session_id": "..."}  # on completion
data: {"token": "...", "done": true, "correction": true}       # if output flagged
data: {"error": "...", "done": true}   # on error
```

### 4. Memory

Session-based conversation memory via LangGraph `MemorySaver` checkpointer. Pass a consistent `session_id` across requests to maintain context. Each session stores `chat_history` injected into the generation prompt.

### 5. State Management

LangGraph `AgentState` (TypedDict) tracks:

- `question`: Rewritten query (updated by router)
- `raw_documents`: Merged result set from parallel retrieval branches (reducer: append)
- `documents`: Filtered document list after grading
- `generation`: Current answer
- `web_search`: Router decision to run web search in parallel
- `web_search_done`: Guard flag preventing fallback loop
- `web_fallback_needed`: Grader signal that triggers web fallback
- `docs_retrieved_total`: Total docs retrieved across all sources (reducer: sum)
- `chat_history`: Multi-turn conversation history
- `model`: Per-query LLM model override
- `top_k`: Per-query retrieval depth override

### 6. Qdrant Modes

Controlled via `.env`:

```
QDRANT_MODE=local   # uses QDRANT_LOCAL_URL (Docker)
QDRANT_MODE=cloud   # uses QDRANT_CLOUD_URL + QDRANT_API_KEY
```

### 7. Evaluation & Monitoring

RAG metrics tracked per query and accessible via `/api/monitoring/stats`:

- **Retrieval Precision**: Ratio of relevant to total retrieved documents
- **Latency**: End-to-end query processing time
- **Web Search Rate**: Percentage of queries using external search
- **Avg Docs Retrieved**: Average number of chunks fetched per query
- **Avg Docs Relevant**: Average number of chunks passing the grader

All metrics are displayed in the UI.

## API Endpoints

**POST /api/stream**

- Query documents with full RAG pipeline, response streamed via SSE
- Request: `{question, session_id?}`
- Returns `text/event-stream` with token events

**POST /api/upload**

- Upload documents (PDF, DOCX, TXT)
- Response: `{document_id, filename, chunks_created, file_size}`

**GET /api/monitoring/stats**

- Aggregated evaluation metrics

## Tech Stack

**LangGraph**, **LangChain**, **Qdrant** (hybrid dense + BM25 sparse), **FastAPI**, **OpenAI**, **PyMuPDF**, **FastEmbed**, **spaCy**, **Streamlit**, **Docker**
