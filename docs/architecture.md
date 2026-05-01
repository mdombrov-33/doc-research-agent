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
- Detects if the query needs recent/web information (keyword match → LLM confirmation)
- Rewrites the query into a search-optimized phrase
- Returns `web_search: True/False` — determines whether web search runs in parallel
- Resets `raw_documents` state so previous turn's docs don't bleed in

**Retrieve** (always runs)
- Pulls top-k chunks from Qdrant via cosine similarity
- Re-ranks using fusion: `score = 0.6 × vector + 0.4 × BM25`
- BM25 index is built on the fly from retrieved chunks using spaCy tokenization
- Both score arrays are normalized to [0,1] before blending

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

Pure vector search misses exact keyword matches. Pure BM25 misses semantic similarity. Fusion covers both — e.g. a query about "scaled dot-product attention" scores high on BM25 for the exact term and high on vector for semantic neighbors like "attention mechanism efficiency".

Alpha=0.6 weights vector slightly higher since semantic match is usually more important for document Q&A.

---

## Guardrails

Guardrails wrap the entire agent — input is checked before the graph runs, output is checked after generation. Input uses OpenAI Moderation API + a `gpt-4o-mini` injection classifier. Output uses Moderation API only. Fails open on errors so a guardrails outage doesn't block the agent.

---

## Eval tracking

Every query logs: latency, docs retrieved, docs graded relevant, web search triggered, generation attempts, retrieval precision. Accessible at `GET /api/evaluation/stats`, all metrics visible in the UI.
