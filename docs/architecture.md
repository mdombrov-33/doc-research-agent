# Architecture

## What it does

Adaptive RAG agent: takes a user question, retrieves relevant context from uploaded documents (and optionally the web), grades what it found, and streams back an answer. Multi-turn — conversation history is preserved per session.

---

## Request flow

```
POST /api/stream
      │
      ▼
Guardrails (input check)
      │
      ▼
    Router
  ┌───┴───────────────────┐
  │                       │ (if web_search=True)
  ▼                       ▼
Retrieve              Web Search
(always)             (parallel)
  │                       │
  └──────────┬────────────┘
             ▼
     Grade Documents
             │
             ▼
          Generate
             │
             ▼
Guardrails (output check)
             │
             ▼
       SSE stream → client
```

---

## Node by node

**Router**
- Single structured-output LLM call decides if the query needs recent/web information
- Rewrites the query into a search-optimized phrase
- Returns `web_search: True/False` — determines whether web search runs in parallel
- Resets `raw_documents` state so previous turn's docs don't bleed in

**Retrieve** (always runs)
- Single hybrid query to Qdrant runs two independent retrievers over the full corpus:
  - Dense: cosine similarity over `text-embedding-3-small` embeddings
  - Sparse: BM25 over a `langchain-sparse` vector, with IDF computed server-side
- Qdrant fuses the two ranked lists with Reciprocal Rank Fusion (RRF) and returns the top-k

**Web Search** (parallel with Retrieve when triggered)
- Calls DuckDuckGo with the rewritten query
- Result lands in the same `raw_documents` state bucket via reducer

**Grade Documents**
- Reads merged `raw_documents` (vector + web combined)
- Sends all docs to LLM in a single batch call: relevant / not relevant per doc
- Filtered results go into `documents`

**Generate**
- Joins graded docs as context, injects `chat_history` for multi-turn
- Streams response token by token via SSE
- Updates `chat_history` in state for next turn

---

## State

LangGraph `StateGraph` with `MemorySaver` checkpointer — state persists across turns per `thread_id`.

Key fields:
- `raw_documents` — parallel fan-in target, reset each query via custom reducer
- `documents` — grader output, passed to generate
- `docs_retrieved_total` — accumulated count per query for eval metrics
- `chat_history` — full turn history, injected into generate prompt

---

## Retrieval: why hybrid

Pure vector search misses exact keyword matches. Pure BM25 misses semantic similarity. Hybrid covers both — e.g. a query about "scaled dot-product attention" ranks high on BM25 for the exact term and high on vector for semantic neighbors like "attention mechanism efficiency".

Both retrievers run over the full corpus and their rankings are combined with Reciprocal Rank Fusion (RRF): each document scores `Σ 1/(k + rank_i)` across the lists it appears in. RRF needs no score normalization or weight tuning — it fuses on rank position alone, which is robust to the different score scales of cosine similarity vs BM25. Qdrant computes this server-side.

---

## Guardrails

Guardrails wrap the entire agent — input is checked before the graph runs, output is checked after generation. Input uses OpenAI Moderation API + a `gpt-5.4-mini` injection classifier. Output uses Moderation API only. Fails open on errors so a guardrails outage doesn't block the agent.

---

## Eval tracking

Every query logs: latency, docs retrieved, docs graded relevant, web search triggered, generation attempts, retrieval precision. Accessible at `GET /api/monitoring/stats`, all metrics visible in the UI.
