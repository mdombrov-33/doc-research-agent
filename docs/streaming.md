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
- Sources metadata at the end (filenames, chunk indices, where it came from)
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
LangGraph agent starts running (agent ⇄ tools → grade loop)
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

More than one node calls an LLM: the **agent** node (which both decides tools *and* writes the
answer) and the **grade** node (which scores documents). Only the agent node's tokens should
reach the user, so we filter on `langgraph_node == "agent"`. In the normal case, tool-deciding
agent turns carry no content, so the only tokens that stream are the final answer's.

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

Everything else (grader token events, tool events, etc.) is ignored.

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

**Error:**

```json
{ "error": "something went wrong", "done": true }
```

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

- `sources_count` — how many docs passed grading (`len(documents)`)
- `sources` — per-doc filename, chunk index, chunk length, source (vectorstore vs web)
- `web_search_used` — whether the agent used the web_search tool this query
- `docs_retrieved_total` — total candidates before grading
- `latency_ms` — full request duration

This gets recorded into the `MetricsTracker` right before the final SSE event is sent (see
`docs/architecture.md` §14).
