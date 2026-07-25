# Architecture

This is the canonical reference for how the Document Research Agent works. Open it to
re-understand any part of the system: ingestion, retrieval, the LangGraph agent and its
tools, the web-search fallback, streaming, guardrails, memory, evaluation, and deployment.

File paths are given throughout so you can jump from a concept to the code.

---

## 1. What it is

An **evidence-controlled RAG** service. You upload documents; you ask questions. For each
question the workflow:

1. screens the input for safety,
2. turns the current conversation into a standalone document-retrieval query,
3. retrieves candidate chunks with **hybrid search**, then re-orders them with a
   **cross-encoder reranker**,
4. structurally assesses whether the evidence is sufficient; it uses one web fallback only when
   document evidence is insufficient,
5. writes a cited answer only after evidence passes, otherwise abstains, and records telemetry.

The graph is deliberately **bounded**: document retrieval happens once, live web search can
happen at most once, and answer generation receives only evidence that the classifier accepted.
Conversations are multi-turn; history is persisted per session (§12).

**Two LLM providers, by role:**
- **OpenRouter** — every chat/LLM call (planner, assessor, answer, and eval judges). The UI
  selects the answer model; planner and assessor models are configured independently. See
  `src/core/llm.py`.
- **OpenAI (direct)** — embeddings only (`text-embedding-3-small`), for ingestion and
  retrieval. Each request has a configurable 30-second default deadline and bounded SDK retries
  for transient rate-limit/server failures. See `src/core/vectorstore.py`.

Safety screening uses **neither** provider: it runs **local** HuggingFace models via
`llm-guard` (§11).

---

## 2. Request lifecycle (the big picture)

```
POST /api/stream  (QueryRequest: question, session_id?, model?)
      │
      ▼
Guardrails — INPUT check        llm-guard Toxicity scanner, local
      │  (flagged → refusal, the graph never runs)
      ▼
Conversational check            greeting / "what can you do?" → fixed reply, the graph never runs
      │  (content-bearing question)
      ▼
Answer cache — first-turn lookup   hit → replay cached answer + sources, the graph never runs
      │  (miss, or a follow-up turn)
      ▼
┌────────────────── LangGraph evidence workflow ───────────────────────┐
│                                                                        │
│ query agent ─► document retrieval ─► assess evidence ─► answer        │
│                                           │                              │
│                                           └── insufficient ─► web       │
│                                                                   │      │
│                                                       assess ─► answer  │
│                                                           └──► abstain   │
└────────────────────────────────────────────────────────────────────────┘
      │  answer-node tokens stream out as they are produced (SSE)
      ▼
MetricsTracker.record(...)      latency, first-token timing, sources retrieved, route outcome
      │
      ▼
SSE stream closes  ({"done": true, "sources": [...]})
```

Input guardrails are a **hard gate** (they run before the graph). There is **no output
guardrail** — once the agent starts answering, tokens stream straight to the client (§11). After
the guardrail, a **conversational check** short-circuits bare greetings and "what can you do?"
meta-questions with a fixed capabilities reply (outcome `conversational`), so they never run the
graph and land as a hard abstention. It matches conservatively — a fixed set of normalized
greetings/thanks and help phrases — so anything carrying real content words falls through
unchanged. The answer cache then sits between that check and the graph and only ever *shortcuts*
it — a miss (or any follow-up turn) runs the full workflow unchanged (§4a).

---

## 3. Process startup & dependency injection

Expensive, process-wide resources are built **once** at startup and stashed on
`app.state`; request handlers read them back through tiny provider functions. This keeps
per-request work cheap and makes everything trivially mockable in tests.

- `src/main.py` — the `lifespan` context manager runs on boot:
  - `ensure_collection_exists()` — create the Qdrant collection + indexes if missing.
  - `ensure_answer_cache_collection()` — create the `answer_cache` collection, its payload
    indexes, and the reserved corpus-version point if missing (§4a).
  - `app.state.vector_store` — the hybrid `QdrantVectorStore`.
  - `app.state.nlp` — the loaded spaCy model (`en_core_web_sm`).
  - `app.state.agent` — the compiled LangGraph app, using the configured checkpointer (§12).
  - `guardrails.warmup()` — primes the llm-guard scanners so the first request doesn't pay
    their cold start (§11).
  - `rerank.warmup()` — primes the cross-encoder for the same reason (§9).
  - `app.state.metrics_tracker` — the online telemetry tracker using the configured store (§14).
- `src/api/dependencies.py` — `get_vector_store`, `get_nlp`, `get_agent`,
  `get_metrics_tracker` just return the corresponding `app.state` object via FastAPI `Depends`.
- `src/api/routes.py` — the routes wire those dependencies into the handlers.

Settings and singletons are cached: `get_settings()` is `@lru_cache`'d (`src/config.py`), as
are the purpose-specific Qdrant clients/stores, the spaCy model, the cross-encoder loader, and
the llm-guard scanners. `get_retrieval_vector_store()` uses the short query deadline; the
lifespan-owned ingestion store and collection setup use the longer indexing deadline.

`GET /health` is liveness only: it is immediate and does not contact a dependency. `GET /ready`
uses `is_qdrant_ready()` to make a read-only `collection_exists` call with the normal query
deadline off the event loop. It returns `{status: "ready"}` only when the required collection is
reachable; a missing collection or Qdrant failure becomes the stable 503 `{status: "unavailable"}`.
It intentionally does not call an LLM or embedding provider, so readiness itself adds no model
cost.

---

## 4. Ingestion pipeline (`POST /api/upload`)

`src/api/handlers/upload.py` → `src/core/ingestion/` (`pipeline.py` orchestrates the steps).

```
upload file (pdf / docx / txt)
   │  streamed to a temp path under UPLOAD_DIR in bounded chunks; rejects files over
   │  MAX_UPLOAD_BYTES (25 MiB by default) and deletes the temp file in a finally-block;
   │  SHA-256 of the bytes is computed during the same stream
   ▼
dedupe check            dedupe.py  (find_duplicate → filtered Qdrant scroll on file_sha256)
   │   already-indexed bytes → return the existing document_id with duplicate:true,
   │   skipping extract/chunk/enrich/embed and leaving corpus_version untouched
   ▼
extract text            extract.py  (extract_from_file)
   │   .pdf → PyMuPDF (200-page cap)   .docx → python-docx paragraphs   .txt → aiofiles
   │   reject extracted text over MAX_EXTRACTED_CHARACTERS (1,000,000 by default)
   ▼
chunk                   chunk.py  (chunk_text → RecursiveCharacterTextSplitter)
   │   chunk_size=1200 chars, chunk_overlap=240, split on ¶ → line → sentence → … 
   │   chunks shorter than 100 chars are dropped; reject documents over
   │   MAX_CHUNKS_PER_DOCUMENT (1,000 by default) before enrichment and indexing
   ▼
enrich each chunk       enrich.py  (enrich_chunks → spaCy NER + keywords)
   │   entities     = ent.text     (first 10)
   │   entity_types = ent.label_   (first 10)
   │   keywords     = NOUN/PROPN, non-stopword, len>2, deduped (first 15)
   ▼
store in Qdrant         index.py  (index_chunks → vector_store.add_documents, offloaded via asyncio.to_thread)
       page_content = "Document: {filename}\n\n{chunk}"  (filename header prepended)
       computes BOTH vectors per chunk:
         • dense  = OpenAI text-embedding-3-small (1536-d)
         • sparse = BM25 (FastEmbed "Qdrant/bm25")
       payload = {document_id, filename, file_sha256, chunk_id, chunk_index, chunk_length,
                  entities, entity_types, keywords, file_extension}
```

Why the filename header? A bare chunk like "It costs $200/month" is unfindable once separated
from its document. Prepending `Document: {filename}` situates it, and because the header is part
of `page_content` it flows into the dense embedding, the BM25 index, and the evidence the answer
model sees (the last is intentional — it helps the model attribute evidence). The header is added
at indexing, *after* the ≥100-char chunk filter, so it never inflates chunk-length decisions.
Only filename is available (the extractor produces flat text with no section headings), and
already-indexed documents pick up the header only on re-upload.

Why enrich with entities? So retrieval can add entity-matched chunks to its candidate pool
(see §5). The same spaCy NER runs at ingestion and at query time, so a query's entities can
match the stored ones without excluding ordinary hybrid results.

`index_chunks` calls `langchain_qdrant`'s sync `add_documents`, which blocks on a Qdrant HTTP
request. It is offloaded to a thread pool via `asyncio.to_thread` so the event loop stays free
during the upload.

**File-level dedupe.** Before extraction, the SHA-256 of the uploaded bytes is looked up
against the `file_sha256` stored in existing chunk payloads. Identical bytes short-circuit the
whole pipeline: the handler returns the existing `document_id` with `duplicate: true` and does
not re-embed. Dedupe is file-level only (the corpus is append-only — there is no update
workflow that would need chunk-level caching). A deduped upload leaves `corpus_version`
untouched, so a no-op re-upload never flushes the answer cache.

A successful *new* upload then bumps the answer cache's `corpus_version` (§4a): a new document can
invalidate any previously cached answer, so bumping the version retires the whole cache in one
integer comparison. The bump is best-effort — it never fails an already-indexed upload.

---

## 4a. Answer cache (`src/core/answer_cache.py`)

A repeated question shouldn't pay the full graph again. Before the graph runs, the stream handler
consults a two-layer cache backed by a dedicated Qdrant collection (`answer_cache`), where each
entry is one point: the question-embedding vector plus a payload holding the answer, its sources
(JSON), and the keys below.

- **Two layers, one entry.** **L1 exact** is a filtered scroll on a SHA-256 `question_hash` of
  the normalized question — no embedding call. On an L1 miss, **L2 semantic** embeds the question
  once and vector-searches the same filter, accepting a hit at cosine ≥ 0.95.
- **Cache key = `question_hash` + `model` + `corpus_version` + `namespace`.** Answers from
  different selectable models must not cross. `namespace` is hardcoded to `"default"` (the app
  is single-tenant, forward-compatible). Startup drops the legacy cache collection once if its
  removed retrieval-width field is still present.
- **Correctness by version, not heuristics.** A reserved point (nil-UUID id) holds the current
  `corpus_version`; every upload bumps it (§4) and deletes now-stale entries. Lookups only match
  the current version, so a new document can never surface a stale answer. A 24 h `created_at`
  floor is a growth/staleness backstop.
- **Only first-turn `document_answer`s are cached.** The handler checks the session checkpoint
  (`aget_state`) and consults/populates the cache only when there are no prior messages — a
  follow-up like "expand on that" depends on history the cache can't reproduce. Web answers and
  abstentions always re-run the graph (tomorrow's web can change them).
- **Serving a hit.** One SSE `token` event with the full answer, then the normal `done` event
  with the cached sources; the turn is recorded with `cache_hit=true` and cost 0 (§14). The
  Human+AI messages are written into the checkpoint (`aupdate_state`) so later follow-ups still
  see the turn in history.
- **Never fatal.** A lookup, store, state-probe, or checkpoint-write failure degrades to running
  the graph (or skipping the cache write) — the cache can only speed the request up, never break
  it. Set `ANSWER_CACHE_ENABLED=false` to bypass it entirely.

---

## 5. The vector store & hybrid search

`src/core/vectorstore.py` (collection setup + the store) and `src/core/retrieval/search.py`
(the bounded retrieval-plan module).

The Qdrant collection holds **two vectors per chunk**:

| Vector | What | Config |
|---|---|---|
| Dense | semantic meaning | `text-embedding-3-small`, 1536-d, **cosine** distance |
| Sparse | exact terms (BM25) | named `langchain-sparse`, IDF computed **server-side** (`Modifier.IDF`) |

Plus a **payload index** on `metadata.entities` (KEYWORD) so entity supplements are fast.

**TF/IDF split — who owns what.** `FastEmbedSparse` (BM25 tokenizer) runs both at ingestion
and at query time. It outputs a sparse vector of `{token_id: tf_score}` — term frequency only,
no IDF. At ingestion, that TF vector is stored in Qdrant and Qdrant updates its collection-wide
document-frequency counts. At query time, `Modifier.IDF` intercepts the dot product and
multiplies each stored dimension by the global `log(N / df)` before scoring. IDF is never
computed by the client; it is an incrementally maintained server-side value that reflects the
entire collection. Adding a new book updates those counts automatically — no reindexing needed.

**Why hybrid?** Dense (bi-encoder) search matches meaning even when words differ, but is weak
on exact tokens — names, codes, rare jargon. BM25 is the opposite. Running both and merging
recovers a class of misses neither catches alone.

**How they merge — Reciprocal Rank Fusion (RRF).** Qdrant fuses the two ranked lists by rank
position, not raw score: a chunk scores `Σ 1/(k + rank_i)` across the lists it appears in.
RRF needs no score normalization or weight tuning, which is what makes mixing cosine and BM25
scales robust. `RetrievalMode.HYBRID` turns this on; it's a single Qdrant query.

**`retrieve_evidence(search_queries, rerank_query)`** is the serving and eval interface:

1. Accept exactly one focused query, or two for a genuinely multipart/follow-up plan. Runtime
   validation rejects empty plans and plans wider than two queries.
2. Run the one or two hybrid Qdrant branches concurrently. Every branch extracts spaCy entities
   and supplements its ordinary hybrid results with entity-filtered matches; supplements can add
   candidates but never exclude the unfiltered results.
3. Each Qdrant request uses `QDRANT_QUERY_TIMEOUT_SECONDS` (10 seconds by default). A transient
   timeout, network error, Qdrant rate limit, or retriable HTTP status gets one short jittered
   retry; permanent errors are not retried.
4. Map results to document dicts and fuse branch ranks deterministically with RRF. Deduplicate by
   stable chunk identity and retain at most `RETRIEVAL_CANDIDATE_BUDGET` unique chunks (40).
5. **Rerank once** against the current user question, apply the calibrated score floor, and keep
   at most `RETRIEVAL_EVIDENCE_BUDGET` chunks (8). There is no raw-order serving mode.

> Bi-encoder vs cross-encoder: hybrid search is still **bi-encoder** — query and chunk are
> embedded separately and compared. That is fast but approximate. The reranker (§9) is a
> **cross-encoder** that reads query and chunk *together* for a sharper relevance judgment.
> We use the fast one to shortlist and the sharp one to re-order the shortlist.

`retrieve_evidence` is called from the `retrieve_documents` **tool** (§6) during serving and
directly from the offline eval (§15), so both exercise the same bounded retrieval interface.

---

## 6. The LangGraph agent — state, tools, topology

`src/core/agent/graph.py` builds the graph; `src/core/agent/state.py` defines the shared
state; `src/core/agent/nodes.py` holds the agent node; `src/core/agent/tools.py` holds the
tools.

### 6.1 The shape: controlled evidence workflow

A tool-calling LLM (`agent`) forms a history-aware standalone query, and can call only
`retrieve_documents`. The rest of the path is graph-controlled: an evidence classifier returns
a structured verdict, then the graph either synthesizes an answer, runs one web fallback, or
abstains.

```
                     START
                       │
                       ▼
                   agent (query only)
                       │
                       ▼
            tools (retrieve_documents)
                       │
                       ▼
              assess_evidence
                 │          │
            sufficient   insufficient
                 │          ▼
                 │     web_fallback
                 │          │
                 │          ▼
                 │   assess_evidence
                 ▼       │       │
              answer  sufficient  insufficient
                 │       │          │
                 └───────┘       abstain
                    │               │
                    └──────► END ◄─┘
```

Edges (`graph.py`):
- entry: `agent`
- `agent` → `tools` only when it requested document retrieval; a non-tool response abstains.
- `tools` → `assess_evidence`.
- sufficient evidence → `answer`; insufficient document evidence → `web_fallback` →
  `assess_evidence`; still-insufficient evidence → `abstain`. The router shares
  `used_web_fallback` with the nodes, so the one-fallback budget is scoped to the current turn:
  a persisted thread's earlier web search must not disable the fallback for later questions.

Each custom node is wrapped by `timed(...)` (`src/utils/node_timer.py`), which logs a
`node_complete` event with `duration_ms`. (`tools` is the prebuilt `ToolNode`, not wrapped.)

The graph is compiled with a checkpointer (`AsyncSqliteSaver` by default,
`AsyncPostgresSaver` when configured, and a `MemorySaver` in tests) — that's what makes it
multi-turn (§12).

### 6.2 State

`AgentState` (`state.py`) has two channels:

```python
class AgentState(TypedDict, total=False):
    messages: Required[Annotated[list[AnyMessage], add_messages]]
    evidence_sufficient: bool
    supporting_source_ids: list[str]
    outcome: FinalOutcome
    stop_reason: FinalStopReason
```

- **`messages`** — the conversation and workflow scratchpad in one list. The
  `add_messages` reducer appends each turn (human question, agent tool-call, tool result,
  agent answer…). The checkpointer persists this per `thread_id`, so it doubles as the
  multi-turn history — there is no separate `chat_history` field. `Required[...]` marks it as
  always-present inside the otherwise-optional `TypedDict`.

- **`evidence_sufficient`** — the latest validated assessment verdict. It is not user input;
  `assess_evidence` rewrites it after document retrieval and, when needed, after web fallback.

- **`supporting_source_ids`** — the selected, validated source IDs for a sufficient verdict.
  They justify the verdict and are what validation checks against the real artifacts; they do
  not narrow what `answer_node` reads.

- **`outcome`** — the final route result: `document_answer`, `web_answer`, or `abstained`.
  It is written only by the terminal `answer` and `abstain` nodes, then reported in the final
  SSE event and aggregated by monitoring. A fourth value, `conversational`, is set by the
  pre-graph conversational check (§2), never by a graph node.

- **`stop_reason`** — the precise terminal cause: `document_evidence_sufficient`,
  `web_evidence_sufficient`, `insufficient_evidence_after_web`, or
  `retrieval_not_requested`. It is per-request diagnostic metadata, not an aggregate metric.
  The graph topology is the execution budget: it permits one document retrieval and, only after
  insufficient evidence, one web fallback—there is no retry loop to count or configure.

> **Per-request configuration is not in state.** The selected answer `model` rides in the
> `RunnableConfig` alongside `thread_id`, not in `AgentState`. Retrieval budgets are internal
> application policy. Keeping invocation-scoped configuration out of evolving, persisted state
> keeps the state minimal (§12).

### 6.3 The tools

`src/core/agent/tools.py` defines two tools, both declared
`@tool(response_format="content_and_artifact")`. That response format is the key trick: each
tool returns a content string plus a structured artifact. The **string** becomes the
`ToolMessage` content the model reads. Document retrieval wraps its bounded document list and
safe count metrics in the artifact; web search uses a document list. `format_docs` includes each
valid document's stable source ID in the model
context. The stream handler later validates only source IDs referenced by the final answer
against these artifacts (§10).

| Tool | What it does | Notes |
|---|---|---|
| `retrieve_documents(search_queries, state)` | one bounded hybrid-search plan over uploaded docs (§5) | One query normally, at most two; `state` is injected by `ToolNode` so reranking uses the actual current user question and the model never supplies it. |
| `web_search(query)` | DuckDuckGo live web search | Returns up to five structured title/link/snippet results, each independently citable. DDGS uses `WEB_SEARCH_TIMEOUT_SECONDS` (30 s default); its explicit timeout/rate-limit errors get one short jittered retry. **Fails soft**: all other/exhausted errors log only their type and return `("Web search failed.", [])`, so an outage never breaks the request. |

`format_docs` renders every artifact inside one `<untrusted_evidence_json>` envelope as JSON:
stable source ID, source type, title, and content. The two evidence-consuming system prompts
explicitly state that the envelope is reference data, never instructions, so prompt-like text in
a document or web snippet is not part of their task. This is an instruction/data separation, not
a claim that prompt injection is impossible for an LLM. Only `retrieve_documents` is bound to the
query agent and wrapped by `ToolNode`; `web_fallback` calls the same `search_web` implementation
directly.

---

## 7. Node by node

### Query — `agent_node` (`nodes.py`)
The dedicated `PLANNER_MODEL` binds only `retrieve_documents`, prepends `AGENT_SYSTEM_PROMPT`,
and resolves a follow-up into a standalone one- or two-query plan using a bounded projection of
persisted `messages`. It cannot write the final answer or call the web. A missing retrieval call
is an abstention, not an unsupported answer. If a provider emits duplicate retrieval calls, the
node deterministically collapses them into one plan capped at two queries. The projection keeps
the most recent `CONVERSATION_HISTORY_TURNS` user turns (three by default, including the current
question) and completed assistant answers; it excludes prior tool calls, tool results, and
evidence artifacts. Planning is capped by `PLANNER_MAX_TOKENS`.

The request's selected model is not used here. It remains the answer model only; the Streamlit
selector reads `SUPPORTED_MODELS`, and the API rejects all other request model IDs with a 422.

### Evidence assessment, answer, and fallback — `nodes.py`
`evidence_assessment_node` sends the current question plus this turn's tool evidence to the
small `ASSESSOR_MODEL` as `EvidenceAssessment(sufficient, supporting_source_ids)`, bounded by
`ASSESSOR_MAX_TOKENS`. A verdict
is accepted only when it is sufficient, names at least one source ID, and every ID exists in the
actual artifacts. The evidence is explicitly labelled untrusted source data before the model sees
it; malformed evaluator output and evaluator failures fail closed. This is the relevance gate:
retrieval and its cross-encoder optimize the candidate order, while the assessor decides whether
the supplied corpus evidence supports an answer at all. We deliberately do not turn raw reranker
scores into a minimum relevance cutoff without a calibrated labelled set.

`answer_node` receives the question and the final bounded evidence set, then writes a cited answer
without tools. It is deliberately not restricted to the IDs the verdict named: the assessor names
enough sources to justify sufficiency, not every source worth quoting, so filtering on them
starved answers of relevant retrieved chunks. User-facing sources stay precise because they are
resolved from the IDs the answer actually cites, not from everything retrieved. Its system prompt separately repeats that those artifacts are
untrusted data rather than instructions. It records `document_answer` when document evidence
passed directly, or `web_answer` when the bounded web fallback ran before evidence passed.
Ordinary synthesis is capped by `ANSWER_MAX_TOKENS` (1,000 by default) and prompted to answer
more briefly when the question is narrow.
`web_fallback_node` searches once using the query agent's standalone retrieval query. If neither
evidence set passes, `abstain_node` emits a fixed honest response and records `abstained`.

---

## 8. Walkthroughs: the two retrieval paths

**Documents-only (the common case).**
1. `agent` → makes exactly one `retrieve_documents` call with one or two planned queries.
2. `tools` → branches run concurrently, merge to 40 candidates, and rerank once to at most
   eight chunks returned as a `ToolMessage` + artifact.
3. `assess_evidence` validates that the artifacts answer the question.
4. `answer` writes a cited response and the graph ends.

**Web-search fallback.**
1. `agent` → calls `retrieve_documents`.
2. `assess_evidence` marks the document artifacts insufficient.
3. `web_fallback` searches with the standalone retrieval query and appends web artifacts.
4. `assess_evidence` checks the combined evidence.
5. `answer` writes a cited response if sufficient; otherwise `abstain` ends the graph.

---

## 9. Reranking (cross-encoder)

`src/core/retrieval/rerank.py`. Retrieval is tuned for recall (cast a wide net); the reranker
is tuned for precision (sharpen the order).

- **Model**: `Xenova/ms-marco-MiniLM-L-6-v2` via fastembed's `TextCrossEncoder` (ONNX, CPU,
  no torch in the hot path). Loaded once and cached (`@lru_cache`).
- **Flow**: one/two hybrid branches → deterministic RRF merge and stable dedupe (≤40) →
  `rerank(current_question, candidates, limit=8)` scores each pair together → sort → apply the
  floor → keep at most eight.
- **Score floor** (`RERANK_SCORE_FLOOR`, raw logit): chunks scoring below it are dropped from the
  evidence pool *before* assessment or the answer model see them, fixing the root cause of
  irrelevant chunks riding into the sources. `None` (code default) keeps every chunk; the
  calibrated value ships in `.env.example`. Calibrated by `make eval-rerank-sweep`, which sweeps
  the threshold over the golden set and picks the highest floor that drops labelled-irrelevant
  chunks at ~zero recall cost. If every chunk falls below the floor the pool is empty, so
  assessment sees no evidence and the graph runs its web fallback or abstains — the intended
  evidence-controlled behaviour, not an error.
- **Logging**: `documents_reranked` reports `candidates`, `returned`, and **`promoted`** — how
  many returned docs ranked below the final evidence cutoff in the merged order, i.e. hits reranking
  rescued. `promoted > 0` proves it changed the outcome.
- **Warmup**: `rerank.warmup()` runs at startup (§3) so the first query doesn't pay the model
  load. Reranking is mandatory on document-serving paths.
- **Deployment**: the model is **baked into the Docker image** at build time
  (`FASTEMBED_CACHE_PATH=/app/.model_cache`) so the first request doesn't pay a ~9s download.
  See §19.

---

## 10. Streaming (SSE)

`src/api/handlers/stream.py`. `POST /api/stream` returns `text/event-stream`.

The handler drives the graph with `agent.astream_events(..., version="v2")` and filters two
event kinds:
- `on_chat_model_stream` **where `metadata.langgraph_node == "answer"`** → forwards the token.
  The query node only emits a retrieval call. The answer node runs only after evidence passes,
  so no planning narration can reach the user.
- `on_chain_end` with `name == "LangGraph"` → the final graph state. The handler scans
  `messages` for `ToolMessage`s that appeared after the last `HumanMessage` in this turn,
  collects their `.artifact` lists, and normalizes them through `SourceCitation`. It then parses
  square-bracket source IDs from the final `AIMessage` and emits only matching artifacts, in
  answer-reference order. It redacts those internal IDs from the streamed answer tokens, so
  users see normal prose rather than implementation identifiers. Unknown IDs, untraceable
  evidence, duplicate IDs, and retrieved but unreferenced evidence are never shown as
  citations. It also flags `web_search_triggered` if any artifact came from the `web_search`
  tool.

`SourceCitation` is the public evidence contract:

```json
{
  "source_id": "document:<chunk_id> | web:<url>",
  "source_type": "document | web",
  "title": "report.pdf or page title",
  "document_id": "required for documents",
  "chunk_id": "required for documents",
  "page": "optional document page",
  "url": "required for web sources",
  "excerpt": "short text actually supplied as evidence"
}
```

The serving prompt requires factual claims to cite these exact IDs, e.g.
`[document:<chunk_id>]` or `[web:<url>]`. The backend is the authority: a marker is displayed
only when it exactly identifies evidence from the same turn.

Event shapes the client sees:
```
data: {"token": "partial text"}                                              # during the answer
data: {"done": true, "sources_count": N, "sources": [...], "session_id": "...", "outcome": "document_answer"}  # success
data: {"error": "...", "done": true}                                         # on error
```
Headers disable proxy buffering (`X-Accel-Buffering: no`, `Cache-Control: no-cache`). After
the stream closes the handler records metrics (§14).

**Whole-query deadline.** `QUERY_TIMEOUT_SECONDS` (1,000 seconds by default) starts at the
stream-handler boundary, before the
input guardrail and is shared by every subsequent graph-event await. It prevents otherwise
bounded dependency calls from accumulating into an unbounded request. On expiration, the handler
cancels the in-flight work, closes the event iterator, logs only the configured timeout, sends
the stable timeout SSE error, and records a separately labelled timeout path.

**Client disconnects.** The handler checks `Request.is_disconnected()` before the input
guardrail and before requesting every next LangGraph event. Once disconnected, it logs only the
stage, closes the graph event iterator, and sends no more tokens, final event, or metrics.
Starlette also cancels the streaming generator when the ASGI server reports a disconnect; that
`CancelledError` intentionally propagates into a currently awaited graph/provider operation.
Cancellation cannot retract a provider request already sent, but it prevents later graph work
from starting.

---

## 11. Guardrails

`src/core/guardrails.py`. **Input only** — there is no output guardrail.

- **Input (hard gate, before the graph).** `check_input` runs the question through
  **llm-guard**'s `scan_prompt` with the local HuggingFace `Toxicity` scanner. The scan runs in
  a thread executor so it doesn't block the event loop. If it flags the text, the request is
  refused immediately (a fixed refusal string) and the graph never runs.
- **No external provider.** Unlike the rest of the app, guardrails touch neither OpenAI nor
  OpenRouter — the models run in-process. They're loaded once (`@lru_cache` on
  `_get_scanners`) and primed at startup by `guardrails.warmup()` (§3); the models are **baked
  into the Docker image** so the first request skips a ~64s download (§19).
- **No output check.** Because the agent streams its answer token by token, there is no point
  at which a full response exists to screen before the user sees it. Output safety would
  require giving up streaming.
- **Retrieved-content isolation.** Documents and web snippets are not trusted merely because the
  system retrieved them. `format_docs` serializes them in an
  `<untrusted_evidence_json>` envelope, and the assessment and answer prompts require the models
  to treat everything inside as factual reference material only. The query model never sees tool
  payloads, including prior-turn evidence (§12). This reduces prompt-injection exposure; it is
  not a guarantee that a probabilistic model will never follow malicious source text.

> Note: the input scanners are not wrapped in a try/except, so a scanner *internal* error
> propagates rather than failing open. If you need fail-open behaviour (degrade safety but
> never block on a guardrails outage), that's a deliberate change to make in `check_input`.

---

## 12. Memory & multi-turn

The graph is compiled with a **checkpointer** (`graph.py`). State is keyed by `thread_id`,
which the stream handler sets to the request's `session_id`
(`config={"configurable": {"thread_id": session_id, ...}}`). Across turns with the same
`session_id`, the `messages` list accumulates (via the `add_messages` reducer) and the agent
receives a bounded conversational projection on every turn.

**`messages` *is* the memory.** There is no separate `chat_history`. Each turn the stream
handler appends only the new `HumanMessage`; everything else — prior questions, the agent's
tool calls, tool results, prior answers — is reloaded from the checkpoint. The full list remains
the workflow record, but `agent_node` selects only the most recent
`CONVERSATION_HISTORY_TURNS` user turns (three by default, including the current question) plus
completed assistant answers. It never gives old tool calls, tool results, or evidence artifacts
to the query model. That split gives bounded prompt cost and avoids treating old untrusted
evidence as conversational instructions, while still feeding both:
- **Conversational memory** — the agent answers *"expand on that"* coherently because the
  prior turns are right there in `messages`.
- **Conversational retrieval** — the agent itself, seeing the history, is instructed by its
  system prompt to resolve a context-dependent follow-up into a standalone `retrieve_documents`
  query before searching.

**Persistent checkpointer.** `CHECKPOINT_BACKEND=sqlite` is the local default and uses
`AsyncSqliteSaver` under `DATA_DIR` (§13). Setting `CHECKPOINT_BACKEND=postgres` with a
`DATABASE_URL` uses `AsyncPostgresSaver` with tables initialized at startup. The serving
path is async (`astream_events`), so both are async checkpointers. The FastAPI lifespan owns
the connection and closes it at shutdown. Tests inject a sync `MemorySaver` via
`build_graph(checkpointer=...)`, which works with the sync `.invoke` API they drive.

> **Durability caveat.** SQLite makes history survive a process restart *when the database
> file persists* — true under docker-compose (the `./data` volume, §19) but **not** on Cloud
> Run, whose local disk is ephemeral and not shared across instances. The current Cloud Run
> deployment is limited to one instance; its state still resets when that instance is replaced.
> Use the Postgres configuration above when durable or shared state is required.

---

## 13. Models, providers & configuration

Model roles:

| Role | Setting / value | Provider | Temp | Where |
|---|---|---|---|---|
| Retrieval planning | `PLANNER_MODEL` | OpenRouter | 0 | `agent_node` |
| Evidence assessment | `ASSESSOR_MODEL` | OpenRouter | 0 | `evidence_assessment_node` |
| Answer synthesis | `LLM_MODEL` (overridable from `SUPPORTED_MODELS`) | OpenRouter | 0 | `answer_node` |
| Embeddings | `EMBEDDING_MODEL` (`text-embedding-3-small`) | OpenAI | — | ingestion + retrieval |
| Reranker | `RERANK_MODEL` (MiniLM cross-encoder) | local ONNX | — | `rerank` |
| Guardrails | Toxicity | local (HF, via llm-guard) | — | `guardrails` |
| Judge | `JUDGE_MODEL` (eval only) | OpenRouter | 0 | `evals/judges.py` |

**Config vs constants** (the project's rule):
- `src/config.py` (`Settings`, pydantic-settings) — anything the running app reads that can
  vary by environment. **Every field is overridable by an env var of the same name**; `.env`
  only needs the per-environment values (secrets, deployment URLs, toggles). Unset fields use
  their code default.
- Genuine code-fixed invariants live next to their use (e.g. `SPARSE_VECTOR_NAME` in
  `vectorstore.py`, the abstention text in `nodes.py`, the OpenRouter base URL in `llm.py`).
- Eval-only values (thresholds, `K`, `JUDGE_MODEL`, judge prompts) live in `evals/`, never in
  the app package.

**`DATA_DIR`.** A single `DATA_DIR` (default `./data`) is the home for local SQLite state;
`metrics_db_path` is used only when `METRICS_BACKEND=sqlite` and `checkpoints_db_path` only
when `CHECKPOINT_BACKEND=sqlite`. Mounting one directory persists both locally.

Other key settings: `RETRIEVAL_*` and `RERANK_*` (§9), `QDRANT_MODE`
(`local`|`cloud`, picks the URL/credentials),
`LLM_MAX_RETRIES` (§17), `RATE_LIMIT*` (§17). See `.env.example` for the full annotated list.

---

## 14. Monitoring — online telemetry

`src/core/monitoring/tracker.py` + `db.py`. This is **live production telemetry**, distinct
from offline evaluation (§15).

After each query the stream handler calls `MetricsTracker.record(QueryMetrics(...))`.
`METRICS_BACKEND=sqlite` is the default and writes to `metrics_db_path` under `DATA_DIR`.
`METRICS_BACKEND=postgres` with `DATABASE_URL` records atomic database increments and
`/api/monitoring/stats` reads the shared aggregate across instances.

Two tables back this: the aggregate `monitoring_stats` row (running totals) and per-query
`query_events` rows (one per query: answer/planner/assessor models, path, query shape, candidate
and evidence counts, reranker-call count, token counts, provider cost, completion latency,
first-token latency, outcome, and `cache_hit` — no question, answer, or evidence content). The aggregate rates
come from `monitoring_stats`; the token/cost averages, cache-hit rate, and per-model breakdown
come from SQL over `query_events`.

`GET /api/monitoring/stats` returns:

| Stat | Meaning |
|---|---|
| `total_queries` | queries served |
| `web_search_rate` | fraction of queries where the agent used `web_search` |
| `avg_sources_retrieved` | mean sources returned per query (all tool calls combined) |
| `avg_latency_ms` | mean end-to-end latency |
| `avg_time_to_first_token_ms` | mean API-handler-entry-to-first-visible-token time; only completed streams that emitted visible text contribute |
| `document_answer_rate` | fraction answered from sufficient document evidence |
| `web_answer_rate` | fraction answered after the one web-fallback route ran |
| `abstention_rate` | fraction where no evidence passed assessment (excludes conversational turns) |
| `conversational_rate` | fraction short-circuited as a greeting/meta-question before the graph |
| `refusal_rate` | fraction rejected by the fail-closed input guardrail |
| `error_rate` | fraction ending with the fixed safe stream error |
| `timeout_rate` | fraction ending at the whole-query deadline |
| `avg_input_tokens` | mean prompt tokens per query, summed across the turn's LLM calls |
| `avg_output_tokens` | mean completion tokens per query |
| `avg_cost_per_query` | mean OpenRouter-reported cost, or `null` when cost is unavailable (see below) |
| `cache_hit_rate` | fraction of queries served from the answer cache (§4a), cost 0 |
| `models` | per-model breakdown: query count and summed input/output tokens and cost |

These are aggregate counters and per-query numbers only: the tracker does not persist questions,
answer text, or evidence excerpts. `web_answer` describes the graph route, so its cited answer
may still include document evidence retained from the first retrieval.

**Tokens and cost.** Token counts come from each LLM call's `usage_metadata`, summed over the
turn (planner tool-call, assessor, answer). Cost is requested from OpenRouter via usage
accounting (`extra_body={"usage": {"include": true}}` in `src/core/llm.py`), but LangChain drops
the provider cost field on streamed calls, and the stream handler consumes the graph through
`astream_events`, which streams every model call. So `reported_cost` is currently always `null`
and `avg_cost_per_query` is `null` — tokens are captured, cost is not. A static price table was
deliberately rejected; cost can be derived from token counts externally if needed.

OpenTelemetry spans and structured request logs complement those aggregates for debugging. The
request middleware binds `x-request-id` for the lifetime of each HTTP request; async graph,
retrieval, and provider logs inherit it. They record operational metadata only: route/outcome,
counts, booleans, configured limits, and exception class. Raw questions, rewritten retrieval
queries, extracted entities, answers, document chunks, web snippets, and exception messages are
not attached to normal telemetry. Upload events similarly use a document's extension and
size bucket instead of its filename. A completed stream emits one `query_completed` event with
the outcome/path, role models, query shape, candidate/evidence/reranker counts, stage durations,
source counts, web-search flag, full latency, and optional first-token latency; its inherited
request ID correlates it with the request log without adding content.

Time to first token starts at the stream-handler boundary, before guardrails, conversation/cache
preflight, and the graph, and stops only when the handler sends its first visible answer token.
Internal citation markers that are buffered or removed do not count. Completion uses that same
request clock through the terminal SSE event. Refusals, cache hits, conversational replies,
document answers, web answers, abstentions, errors, and timeouts have separate path labels.

---

## 15. Evaluation — offline quality gate

`evals/` (repo root, **not** part of the shipped app). Full details in `evals/README.md`;
the summary:

It scores the **real retrieval layer** (`retrieve_evidence`) — and, under `--full`, a generation
step driven by the eval's own `GENERATION_*` prompts — against a fixed labelled **golden set**
(`evals/golden.jsonl`) over a fixed **corpus** (`evals/corpus/`, ingested into an isolated
`documents_eval` Qdrant collection). It deliberately scores the retrieval layer **directly**,
not through the agent's tool loop, so the numbers measure retrieval quality rather than the
LLM's tool-choice behaviour. Three independent levels:

| Level | Question | Metrics | File |
|---|---|---|---|
| 1. Retrieval | did we fetch the right docs? | recall@k, precision@k, MRR, MAP, nDCG | `ranking.py` |
| 2. Generation | grounded + on-topic answer? | faithfulness, answer-relevance (LLM judge) | `judges.py` |
| 3. Embeddings | does the embedder still separate? | cosine separation guard | `embeddings_check.py` |

The manual graph eval also loads `evals/graph_cases.jsonl`: explicit out-of-corpus and partially
answerable cases must abstain with web disabled, while multipart and history-aware follow-up
cases must retain document coverage and valid citations.

Each aggregate is compared to a threshold; below it, the run exits non-zero. **Two tiers:**
- **Default** (`make eval-retrieval`, run in CI on push to main): retrieval + embeddings only.
  Deterministic, cheap (no generation, no judges) — needs only `OPENAI_API_KEY` for embeddings.
- **`--full`** (`make eval`, local): adds generation + LLM judges. Expensive and noisy, so it
  never gates CI.

This is the offline counterpart to §14: monitoring measures live traffic; evals measure
quality against known-correct answers.

The graph itself has a separate deterministic route contract in
`tests/integration/test_agent_graph.py`, exposed as `make eval-graph-contract`. It runs the
compiled graph with corpus artifacts from `evals/corpus/` and scripted model verdicts to cover
document answer, web fallback, and abstention. It verifies graph control, evidence-source
validation, selected-evidence isolation, and outcome reporting—not semantic model quality. It
needs no service, API key, or live model, so CI runs it as a named check on every PR.

`make eval-graph` adds the complementary **manual live-graph check**. It re-ingests that same
corpus into the isolated `documents_eval` collection, then invokes the compiled graph with the
live query, assessment, and answer models for every golden question. A case passes only when it
ends as `document_answer` and its final citations cover every labelled filename. Web fallback is
disabled: a known-corpus question that needs it is precisely the regression the command should
report. It is intentionally not a CI gate because it uses live models. Pass `--limit N` to the
module for a short live smoke run; `make eval-graph` runs the full golden set.

---

## 16. API reference

| Endpoint | Purpose | Request | Response |
|---|---|---|---|
| `POST /api/stream` | RAG query, streamed | `{question, session_id?, model?}` | `text/event-stream` (§10) |
| `POST /api/upload` | ingest a document | multipart file (`.pdf`/`.docx`/`.txt`, 25 MiB default cap) | `{document_id, filename, chunks_created, file_size, duplicate}` (`duplicate:true` + `chunks_created:0` when the same bytes were already indexed) |
| `GET /api/monitoring/stats` | live telemetry | — | aggregates (§14) |
| `GET /health` | liveness | — | `{status, environment, llm_model}` without a dependency probe |
| `GET /ready` | readiness | — | `200 {status: "ready"}` when Qdrant collection is readable; otherwise `503 {status: "unavailable"}` |

Schemas: `src/api/schemas.py`. Every request gets an `x-request-id` (header + bound into
every log line) via `RequestLoggingMiddleware` (`src/api/middleware.py`); query-serving logs
and spans use safe operational fields rather than request or evidence content. `/stream` and
`/upload` are rate-limited per IP and may return **429** (§17).

---

## 17. Error handling & resilience

| Failure | Handling | Where |
|---|---|---|
| Chat-model timeout or transient API error | `ChatOpenAI(timeout=LLM_TIMEOUT_SECONDS, max_retries=LLM_MAX_RETRIES)`; 60 s default | `llm.py` |
| OpenAI embedding timeout or transient API error | `OpenAIEmbeddings(timeout=EMBEDDING_TIMEOUT_SECONDS, max_retries=EMBEDDING_MAX_RETRIES)`; 30 s and two retries by default. Retrieval follows the fixed-error/web-fallback path; uploads return the existing stable processing error. | `vectorstore.py`, `graph.py`, `handlers/upload.py` |
| Qdrant retrieval timeout / transient failure | `get_retrieval_vector_store()` uses `QDRANT_QUERY_TIMEOUT_SECONDS`; 10 s default; each candidate branch retries once with jitter, then emits a fixed tool error and follows the normal one-shot web fallback | `vectorstore.py`, `search.py`, `graph.py` |
| Qdrant collection/indexing timeout | `get_ingestion_qdrant_client()` and `get_ingestion_vector_store()` use `QDRANT_INGESTION_TIMEOUT_SECONDS`; 30 s default | `vectorstore.py` |
| Qdrant unavailable for readiness | one read-only collection check with the 10 s query deadline; return stable 503, never provider text | `vectorstore.py`, `main.py` |
| Web-search timeout/rate limit | `DDGS(timeout=WEB_SEARCH_TIMEOUT_SECONDS)`; one short jittered retry, then fail soft → empty docs and the graph abstains | `agent/tools.py` |
| Imperfect entity tags | add entity matches to, never restrict, the unfiltered hybrid pool | `search.py` |
| Empty / unsupported upload | safe client message → 400/500; safe failure class and extension/size bucket logged; temp file always cleaned up | `handlers/upload.py` |
| Request floods / runaway cost | per-IP rate limit (slowapi) → 429 + `Retry-After` | `api/rate_limit.py` |

### Rate limiting

The two endpoints that cost money or do real work — `/stream` (LLM + embeddings per call) and
`/upload` (parsing + embedding) — are rate-limited with **slowapi** (`api/rate_limit.py`). The
shared `Limiter` decorates those two routes (`@limiter.limit(rate_limit)`), so
`/monitoring/stats`, `/health`, and `/ready` stay unmetered.

- **What it limits:** requests per client IP. The limit is the `RATE_LIMIT` string
  (e.g. `30/minute`); exceeding it returns **HTTP 429** with a `Retry-After` header, via the
  `_rate_limit_exceeded_handler` registered in `main.py`. A custom `key_func` reads the client
  from the first `X-Forwarded-For` entry (the real caller behind Cloud Run / a proxy), falling
  back to the socket address.
- **Where it lives:** in the *application* layer, because only the app knows what to meter
  (this endpoint vs that one). In a fuller deployment this is one of several layers — a
  CDN/WAF and the load balancer would shed coarse IP floods *before* they reach the app, and
  OpenRouter/OpenAI already rate-limit us upstream (their 429s are absorbed by `max_retries`).
- **Single-instance vs scale:** the limiter's `RATE_LIMIT_STORAGE_URI` defaults to `memory://`
  — correct on one instance, like the SQLite local modes for monitoring (§14) and conversation
  memory (§12).
  Cloud Run autoscales, so in-memory counters aren't shared and a client spread across N
  instances effectively gets N× the limit. The fix is **one config change**: point the URI at
  `redis://…` and slowapi shares counters across instances. Keyed on IP rather than user
  because the app has no auth.

---

## 18. Project layout

```
src/
  api/            delivery layer: routes, handlers (stream/upload), middleware, DI, schemas, rate_limit
  core/
    agent/        the LangGraph evidence workflow
      graph.py    builds + compiles query → evidence assessment → answer/fallback, owns checkpointing
      state.py    AgentState (messages + evidence verdict and source IDs)
      nodes.py    query, assessment, answer, fallback, and abstention nodes
      tools.py    retrieve_documents + reusable web search (content_and_artifact)
      prompts.py  agent system prompt + eval-only generation prompts
    retrieval/    hybrid search (search.py) + cross-encoder reranker (rerank.py)
    ingestion/    extract, chunk, enrich (spaCy), index, pipeline
    monitoring/   online telemetry (tracker + SQLite/Postgres store)
    guardrails.py llm-guard input scanners
    vectorstore.py  Qdrant setup + the hybrid store
    nlp.py, llm.py, exceptions.py
  utils/          logger, node timer, retry
  config.py       Settings (env-overridable)
  main.py         FastAPI app + lifespan wiring
evals/            offline evaluation (golden set, corpus, metrics) — NOT shipped
tests/            unit + integration
ui.py             Streamlit frontend
```

Principle: the **app package stays free of eval-only concerns**; `evals/` and `tests/` are dev
tooling (excluded from the Docker image via `.dockerignore`).

---

## 19. Deployment notes

- **Docker**: multi-stage build (`Dockerfile`). The runtime stage bakes **two** sets of models
  into the image so cold starts don't download them:
  - the reranker cross-encoder into `FASTEMBED_CACHE_PATH=/app/.model_cache` (skips ~9s);
  - llm-guard's Toxicity model into `HF_HOME=/app/.hf_cache`.
  Both bakes run *before* `COPY . .` and inline the model names (not imported from `src`) so the
  layers — and the downloads — stay cached across code changes. Keep the names in sync with
  `config.RERANK_MODEL` and `guardrails.py`.
- **Why not `/tmp`**: fastembed's default cache is `/tmp/fastembed_cache`, but Cloud Run mounts
  a fresh in-memory tmpfs over `/tmp` per instance — a model baked there would vanish at
  runtime. The persistent `/app/.model_cache` and `/app/.hf_cache` paths are the fix.
- **Runtime state**: `DATA_DIR` (`/app/data`) holds the local `metrics.db` and
  `checkpoints.db`. docker-compose mounts `./data:/app/data` so both persist locally; on
  Cloud Run the disk is ephemeral, so state resets when an instance is replaced.
- **Qdrant compatibility**: Compose pins the Qdrant server image and `pyproject.toml` pins the
  Python SDK to the same minor release. Keep their minor versions equal when upgrading; `latest`
  can move the server beyond the client's supported minor-version window.
- **Target**: GCP Cloud Run (Terraform in `terraform/gcp`). spaCy + onnxruntime + the
  reranker + llm-guard's models must fit in the instance memory; watch for OOM if you scale the
  corpus or models.
- **CI**: `.github/workflows/ci.yml` (ruff, format, mypy, pytest, docker build) and
  `eval.yml` (retrieval eval gate on push to main; needs `OPENAI_API_KEY`).
- **Config**: see `.env.example`. Local secrets (`OPENAI_API_KEY`, `OPENROUTER_API_KEY`, Qdrant
  cloud credentials) go in the environment. Cloud Run reads those credentials from Secret
  Manager; its Terraform configuration never accepts their plaintext values.

```
make up             # boot Qdrant + API via docker compose
make dev-compose    # API + Streamlit + Qdrant + Jaeger, with container hot reload
make dev            # run the API locally (uvicorn --reload)
make ui             # run the Streamlit UI
make test           # pytest
make eval           # full offline eval (local)
make eval-retrieval # retrieval+embeddings eval (what CI runs)
make eval-graph-contract # deterministic graph workflow gate (what CI runs)
```
