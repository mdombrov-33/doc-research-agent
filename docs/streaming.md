# Streaming

How a question goes from the user typing it to tokens appearing on screen one by one.

---

## The short version

We use Server-Sent Events (SSE). The backend streams JSON lines over an open HTTP connection,
and the frontend reads them as they arrive. Each token is its own message. At the very end,
one final message carries metadata (sources, session id, etc.).

---

## Why SSE and not just `yield chunks`?

Plain chunked streaming works fine if all you're sending is text. But over the same connection
we need to send different kinds of things:

- Tokens (the user-visible answer text, one by one)
- Typed source citations at the end (document or web identity, plus an internal evidence excerpt)
- Errors

With raw chunked streaming the frontend gets a byte stream with no structure — it can't tell
whether what just arrived is a token, a source list, or an error. We'd end up inventing our
own delimiter format anyway.

SSE gives us that structure for free. Every message is `data: {...}\n\n`, so the client parses
JSON and checks what's inside: is this a token? Is this the done event?

WebSockets would also work but they're bidirectional — the client can send messages back
mid-stream. We don't need that. SSE is unidirectional (server → client only), which is exactly
the case here, and it's plain HTTP so there's nothing extra to set up.

---

## The flow end to end

```
User types question
      ↓
POST /api/stream
      ↓
Guardrails input check → flagged? send refusal and close stream
      ↓ (if safe)
LangGraph workflow runs (query → retrieval → evidence assessment → answer)
      ↓
on_chat_model_stream events from the "answer" node → each token → SSE line → client renders it
      ↓
on_chain_end event → grab sources metadata from the final state
      ↓
Final SSE event: done (with sources)
      ↓
Metrics recorded
```

---

## Backend — how tokens get sent

Everything lives in `src/api/handlers/stream.py`. FastAPI's `StreamingResponse` wraps an async
generator, which is the key piece:

```python
async for event in agent.astream_events(inputs, config=config, version="v2"):
    if (
        event["event"] == "on_chat_model_stream"
        and event.get("metadata", {}).get("langgraph_node") == "answer"
    ):
        token = event["data"]["chunk"].content
        if token:
            accumulated.append(token)
            yield f"data: {json.dumps({'token': token})}\n\n"
```

Each `yield` sends one SSE line to the client immediately. The format is always
`data: {...}\n\n` — that's the SSE spec.

**What if the client leaves?** Before guardrails and before each next LangGraph event, the
handler checks FastAPI's `Request.is_disconnected()`. A disconnected browser receives no more
tokens or final event; the graph event iterator is closed and the request is not recorded as a
completed query. Starlette also propagates server disconnect cancellation into an in-flight async
graph await. That cannot undo a provider request that has already reached the provider, but it
prevents subsequent graph work from starting.

**What if the request takes too long?** `QUERY_TIMEOUT_SECONDS` is one wall-clock deadline for
the input guardrail and every later graph-event await (120 seconds by default). It prevents the
separate LLM, retrieval, and web-search timeouts from adding up without a cap. When it expires,
the current await is cancelled, the graph iterator closes, and the client gets the stable timeout
error event. It is not counted as a completed query.

**Why `astream_events` and not `astream`?**

`astream` gives us full state snapshots after each node completes — too coarse for token-level
streaming. `astream_events` gives us low-level events including `on_chat_model_stream`, which
fires for every single token the LLM produces.

**Why filter on the `answer` node?**

The query node only forms a standalone document-retrieval query; it never writes user-visible
text. The dedicated **answer** node runs only after evidence assessment passes, so filtering on
`langgraph_node == "answer"` guarantees that process narration cannot leak into the stream.

**What events we actually care about:**

| Event                                       | When it fires  | What we do with it                          |
| ------------------------------------------- | -------------- | ------------------------------------------- |
| `on_chat_model_stream` from `answer` node   | Every token    | Yield it to client, append to `accumulated` |
| `on_chain_end` from `LangGraph`             | Graph finishes | Pull sources metadata out of the final state |

Everything else (tool events, graph lifecycle events, etc.) is ignored.

---

## The SSE message format

Three kinds of messages the frontend can receive:

**Token (mid-stream):**

```json
{"token": "The"}
{"token": " answer"}
{"token": " is"}
```

**Done (happy path):**

```json
{"done": true, "sources_count": 3, "sources": [...], "session_id": "..."}
```

Each `sources` item is a validated `SourceCitation`. Documents provide `document_id` and
`chunk_id` (and optionally `page`); web sources provide a real `title` and HTTP(S) `url`.
Both include a short `excerpt` copied from the evidence for programmatic use. The model's raw
answer must reference a source ID in square brackets for it to appear here; the stream removes
those implementation markers before sending answer tokens to the UI. The server drops invalid,
unknown, duplicate, and retrieved-but-unreferenced citations, preserving the raw answer's
first-reference order.

**Error:**

```json
{ "error": "Unable to complete the request. Please try again.", "done": true }
```

The client receives this fixed message for stream failures; exception details stay in server logs.

---

## Frontend — how tokens get rendered

In `ui.py`, the `stream_query()` function is a generator that consumes the SSE stream:

```python
def stream_query(question: str, model: str | None, top_k: int):
    with requests.post(..., stream=True) as response:
        for line in response.iter_lines():
            if not line or not line.startswith(b"data: "):
                continue
            data = json.loads(line[6:])
            if "error" in data:
                yield f"\n\n*Error: {data['error']}*"
                return
            if "token" in data:
                yield data["token"]
            if data.get("done"):
                st.session_state.stream_meta = data
                return
```

Streamlit's `st.write_stream()` accepts a generator and renders each yielded value as it
arrives. The final metadata (sources, etc.) is stashed in session state and rendered
separately after the stream closes.

---

## Metadata and telemetry

When the graph finishes, the `on_chain_end` event carries the final state, from which the
handler reads everything the monitoring tracker needs:

- `sources_count` — validated evidence citations referenced by the final answer
- `sources` — typed citations: `source_id`, `source_type`, title, an internal evidence excerpt,
  and either document identifiers or a web URL. The Streamlit UI groups document citations by
  file and page and does not repeat excerpts.
- `web_search_used` — whether the agent used the web_search tool this query
- `sources_retrieved_total` — total raw artifacts returned by retrieval and web search; it is
  intentionally distinct from `sources_count`
- `outcome` — `document_answer` when document evidence passed directly, `web_answer` when the
  one web fallback ran before an answer, or `abstained` when no evidence passed
- `stop_reason` — the terminal cause: document or web evidence passed, evidence stayed
  insufficient after web fallback, or the query model never requested retrieval
- `latency_ms` — full request duration
- `time_to_first_token_ms` — graph-start-to-first-visible-answer-token time, present only when
  the completed stream emitted visible answer text. Terminal nodes that build their message
  without calling a model (`abstain`) produce no `on_chat_model_stream` events, so the handler
  sends their text once the graph ends; for those the value is the full request duration.

This gets recorded into the `MetricsTracker` right before the final SSE event is sent (see
`docs/architecture.md` §14). The tracker persists aggregate counters only; the accompanying
request logs and OpenTelemetry spans use request IDs plus safe operational fields, never the
question, answer, or retrieved evidence text. That same point emits one `query_completed` log
event with outcome, stop reason, aggregate source counts, web-search flag, latency, and optional
first-token latency; the existing request ID correlates it with the HTTP completion log.
