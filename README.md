# Document Research Agent

Production RAG system with LangGraph state machine, hybrid search, SSE streaming, and NeMo Guardrails security layer.

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
│ NeMo Guardrails  │  Input check (LLM-based: jailbreak, prompt injection)
└──────┬───────────┘
       │ safe
       ▼
┌──────────────────┐
│  LangGraph Agent │  State machine → streams tokens via SSE
└──────┬───────────┘
       │ full response
       ▼
┌──────────────────┐
│ NeMo Guardrails  │  Output check (LLM-based: harmful content, policy)
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
         web_search=true       web_search=false
              │                     │
              ▼                     ▼
        ┌──────────┐          ┌──────────┐
        │WebSearch │          │ Retrieve │ (Hybrid: Vector + BM25)
        └────┬─────┘          └────┬─────┘
             │                     │
             └──────────┬──────────┘
                        │
                        ▼
                 ┌─────────────┐
                 │ Grade Docs  │ (Batch LLM: relevant?)
                 └──────┬──────┘
                        │
              ┌─────────┴─────────┐
              │                   │
         no relevant          has relevant
            docs                 docs
              │                   │
              ▼                   ▼
        ┌──────────┐          ┌──────────┐
        │WebSearch │          │ Generate │ (Stream answer via SSE)
        │ (retry)  │          └────┬─────┘
        └──────────┘               │
                                   ▼
                                 ┌─────┐
                                 │ END │
                                 └─────┘
```

**Node Descriptions:**

- **Router**: LLM classifies query type (vectorstore vs websearch) and rewrites it for semantic search.
- **Retrieve**: Hybrid search (60% vector similarity + 40% BM25 keyword), top-5 results
- **WebSearch**: DuckDuckGo fallback when docs insufficient
- **Grade Docs**: Batch LLM grading (5 docs in parallel)
- **Generate**: Synthesize answer from graded documents, stream tokens via SSE

## How It Works

### 1. Document Upload & Processing

Documents (PDF, DOCX, TXT) are chunked with overlap, embedded using `text-embedding-3-small` (1536 dimensions), and stored in Qdrant with metadata (filename, page numbers, chunk index).

### 2. Query Flow

**Security (NeMo input check):**

- LLM-based input rail using `self_check_input` prompt
- Colang flows catch: prompt injection, jailbreaks, off-topic requests, system probing, code execution attempts
- Blocked inputs return refusal immediately, before LangGraph runs

**Router Node:**

- Single LLM call: classifies as `vectorstore` or `websearch` AND rewrites query for semantic search
- Explicit phrases ("search web", "check online") override to web search path

**Retrieve Node (Hybrid Search):**

- Vector search: Qdrant cosine similarity (k=5)
- BM25 search: Keyword-based ranking using spaCy tokenization
- Fusion ranking: Weighted combination (60% vector, 40% BM25), scores normalized

**Grade Documents Node:**

- Batch LLM grading of 5 retrieved documents in parallel
- Binary relevance scoring (yes/no) per document
- Triggers web search if zero relevant docs found

**Generate Node:**

- Synthesizes answer from graded documents, streams tokens via SSE
- Includes `chat_history` for session-aware multi-turn responses

**Output check (post-streaming):**

- After streaming completes, full response is checked using NeMo's `self_check_output` prompt template via direct LLM call
- NeMo's colang output patterns are not executed (incompatible with streaming architecture)
- If LLM returns "yes" to the policy check, correction event is sent to client

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
- `documents`: Retrieved/graded document list
- `generation`: Current answer
- `web_search`: Routing flag
- `retrieval_attempts`: Retry counter
- `generation_attempts`: Retry counter
- `chat_history`: Multi-turn conversation history

### 6. Qdrant Modes

Controlled via `.env`:

```
QDRANT_MODE=local   # uses QDRANT_LOCAL_URL (Docker)
QDRANT_MODE=cloud   # uses QDRANT_CLOUD_URL + QDRANT_API_KEY
```

### 7. Evaluation & Monitoring

RAG metrics tracked per query and accessible via `/api/evaluation/stats`:

- **Retrieval Precision**: Ratio of relevant to total retrieved documents
- **Latency**: End-to-end query processing time
- **Web Search Rate**: Percentage of queries using external search

## API Endpoints

**POST /api/stream**

- Query documents with full RAG pipeline, response streamed via SSE
- Request: `{question, session_id?}`
- Returns `text/event-stream` with token events

**POST /api/upload**

- Upload documents (PDF, DOCX, TXT)
- Response: `{document_id, filename, chunks_created, file_size}`

**GET /api/evaluation/stats**

- Aggregated evaluation metrics

## Tech Stack

**LangGraph**, **LangChain**, **NeMo Guardrails**, **Qdrant**, **FastAPI**, **OpenAI**, **PyMuPDF**, **BM25 + spaCy**, **Streamlit**, **Docker**
