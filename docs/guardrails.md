# Guardrails

One check, before the agent runs. There is **no output guardrail** (see below).

`src/core/guardrails.py`.

## Input check

`check_input` runs before LangGraph. It screens the question with **llm-guard**'s
`scan_prompt` using two **local** HuggingFace scanners:

1. **`Toxicity`** — harmful, abusive, or explicit content.
2. **`PromptInjection`** — prompt injection, jailbreaks, system probing.

If **either** scanner flags the text, a fixed refusal string is returned and LangGraph never
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

The scanners are **not** wrapped in a try/except, so an internal scanner error propagates
rather than failing open. If you want fail-open behaviour (degrade safety but never block the
agent on a guardrails outage), wrap the scan in `check_input` and return "not flagged" on
error.
