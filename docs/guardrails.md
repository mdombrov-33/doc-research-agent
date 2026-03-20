# Guardrails

NeMo Guardrails sits in front of the agent and decides what gets through and what doesn't. Here's what it actually does, and why it's set up the way it is.

---

## The short version

Before a question reaches the agent: NeMo checks it. After the agent finishes streaming: the output gets checked too. If either check fails, a refusal message goes back instead.

---

## Why not just run NeMo normally?

NeMo wasn't built for streaming. Its `generate_async()` call runs the whole pipeline and returns a complete response. But we stream tokens one by one as they're generated — we can't pause mid-stream and hand it to NeMo.

So the guardrails are split into two separate phases that work around this.

---

## Phase 1 — Input check (before anything runs)

This happens in `guardrails_wrapper.py → check_input()` before a single token is streamed.

NeMo's full pipeline runs here. Two things can block a message:

**1. Pattern matching (input.co)**

The Colang files define literal phrases to watch for:

```colang
define user express prompt injection
  "ignore previous instructions"
  "you are now"
  ...

define bot refuse prompt injection
  "I cannot process that request. Please ask a question about your documents."

define flow block prompt injection
  user express prompt injection
  bot refuse prompt injection
  stop
```

If the message contains one of those strings, NeMo matches the flow, returns the exact bot message defined in the `.co` files, and stops.

**2. LLM self-check (self_check_input)**

For everything that doesn't match a literal pattern, an LLM looks at the message and decides yes/no: should this be blocked? This is configured in `config.yml` under `rails.input.flows`. When it says yes, NeMo generates its own generic refusal — not the bot messages we defined in the `.co` files. This is why "I'm sorry, I can't do that" shows up instead of our phrases — the LLM check fires, not the pattern match.

**The probe trick**

There's one awkward thing here. NeMo's pipeline expects to call a `rag_query` action at the end (defined in `general.co`). But we don't want NeMo running the actual RAG — LangGraph does that separately. So `check_input()` registers a fake `rag_query` action that just sets a flag and returns `"__SAFE__"`. If NeMo reaches that action, the input passed — flag is set, we return `None` (safe). If NeMo blocked before reaching it, the flag is never set, and we return the refusal message.

```
User message → NeMo pipeline runs
                    ├── pattern match? → exact bot message from .co → stop
                    ├── LLM says block? → NeMo's generic message → stop
                    └── reached rag_query probe? → safe, return None
```

---

## Phase 2 — Output check (after streaming finishes)

This happens in `guardrails_wrapper.py → check_output()` after all tokens have been accumulated.

This one does NOT run NeMo's pipeline. Instead it manually:

1. Takes the `self_check_output` prompt template from `config.yml`
2. Fills in the full bot response where `{{ bot_response }}` is
3. Sends that prompt directly to the LLM and asks "should this be blocked? yes or no?"
4. If yes → returns a hardcoded correction message

The output Colang flows in `output.co` do not run. They're loaded by NeMo but never wired up — `rails.output.flows` in `config.yml` is empty. This was a deliberate tradeoff: those flows only do literal string matching anyway, which is weaker than the LLM check.

**One thing to keep in mind:** by the time the output check runs, the streamed tokens are already on their way to the client. If the check flags something, a correction event is sent at the end with `correction: true`. The frontend is responsible for replacing what was already shown.

---

## What's actually working vs not

| | Working? |
|---|---|
| Input pattern matching (.co phrases) | Yes |
| Input LLM self-check | Yes |
| Output LLM self-check | Yes |
| Output Colang flows (output.co) | No — never wired up |
| RAG through NeMo | No — probe trick bypasses it |

---

## Where the files live

```
src/guardrails/
├── config.yml              # NeMo config: model, prompts, which flows to run
├── guardrails_wrapper.py   # The actual check_input / check_output logic
└── rails/
    ├── input.co            # Pattern-matching flows for blocking bad inputs
    ├── output.co           # Output filters (defined but not wired up)
    ├── general.co          # Default flow — defines the rag_query action
    └── dialog.co           # Intent definitions (informational, not used for blocking)
```

The streaming integration is in `src/api/handlers/stream.py` — that's where `check_input()` is called before streaming starts and `check_output()` is called after it finishes.