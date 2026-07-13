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

- Tokens (the answer text, one by one)
- Typed source citations at the end (actual evidence excerpts plus document or web identity)
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
LangGraph agent starts running (agent ⇄ tools → post_tools loop)
      ↓
on_chat_model_stream events from the "agent" node → each token → SSE line → client renders it
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
        and event.get("metadata", {}).get("langgraph_node") == "agent"
    ):
        token = event["data"]["chunk"].content
        if token:
            accumulated.append(token)
            yield f"data: {json.dumps({'token': token})}\n\n"
```

Each `yield` sends one SSE line to the client immediately. The format is always
`data: {...}\n\n` — that's the SSE spec.

**Why `astream_events` and not `astream`?**

`astream` gives us full state snapshots after each node completes — too coarse for token-level
streaming. `astream_events` gives us low-level events including `on_chat_model_stream`, which
fires for every single token the LLM produces.

**Why filter on the `agent` node?**

The **agent** node both decides tools and writes the answer. We filter on
`langgraph_node == "agent"` so tool events never reach the client. In the normal case,
tool-deciding agent turns carry no content, so the only tokens that stream are the final
answer's.

> **Known edge — preamble tokens.** The agent node *can* emit content alongside a tool call —
> e.g. *"Let me search the web for you!"* immediately before a `web_search` call. That text is
> on the `agent` node, so it streams to the user ahead of the real answer. It's harmless (the
> final answer is still complete and correct), just a bit of process-narration leaking in. Tune
> the system prompt if you want to suppress it.

**What events we actually care about:**

| Event                                       | When it fires  | What we do with it                          |
| ------------------------------------------- | -------------- | ------------------------------------------- |
| `on_chat_model_stream` from `agent` node    | Every token    | Yield it to client, append to `accumulated` |
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
Both include a short `excerpt` copied from the evidence used by the model. The server removes
invalid and duplicate citations before sending the event, preserving the first evidence order.

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

- `sources_count` — sources returned by the agent's tool calls for this question
- `sources` — typed citations: `source_id`, `source_type`, title, evidence excerpt, and either
  document identifiers or a web URL
- `web_search_used` — whether the agent used the web_search tool this query
- `sources_retrieved_total` — total sources returned by retrieval and web search
- `latency_ms` — full request duration

This gets recorded into the `MetricsTracker` right before the final SSE event is sent (see
`docs/architecture.md` §14).
