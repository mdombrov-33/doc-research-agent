# Guardrails

One check, before the agent runs. There is **no output guardrail** (see below).

`src/core/guardrails.py`.

## Input check

`check_input` runs before LangGraph. It screens the question with **llm-guard**'s
`scan_prompt` using one **local** HuggingFace scanner:

1. **`Toxicity`** — harmful, abusive, or explicit content.

If the scanner flags the text, a fixed refusal string is returned and LangGraph never
runs. The scan is CPU-bound, so it runs in a thread executor (`run_in_executor`) to avoid
blocking the event loop.

> **No external provider.** Unlike the rest of the app, guardrails call neither OpenAI nor
> OpenRouter — the models run in-process. They're loaded once (`@lru_cache` on
> `_get_scanners`) and primed at startup by `guardrails.warmup()`, so the first real request
> doesn't pay the cold start. The models are **baked into the Docker image** (`HF_HOME=
> /app/.hf_cache`) so the image build, not the first request, absorbs the ~64s download.

## No output check

The agent streams its answer token by token, so there is no point at which a complete response
exists to screen before the user has already seen it. Hard-blocking would mean buffering the
whole answer first and giving up streaming — which we don't. So output safety is simply not
enforced here.

## Failure behaviour

The scan is wrapped in a `try/except` inside `check_input`. If the scanner throws, the error
is logged and the request is **blocked** (fail-closed) — the exception can't propagate after
`200 OK` + `text/event-stream` headers are already sent, which would silently corrupt the SSE
stream.
