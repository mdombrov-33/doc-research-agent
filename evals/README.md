# RAG Evaluation

## Why this exists

A RAG system has many moving parts — the embedding model, chunk size, retrieval filters,
the generation prompt, the LLM. Change any one of them and quality can silently get worse.
This suite answers one question: **did this change make retrieval or answers worse?**

It runs the *real* pipeline against a fixed, labelled set of questions (the "golden set")
and measures quality with hard numbers. If a metric drops below its threshold, the run
fails — locally before you push, and in CI on every PR. That is the entire point: a
**regression gate** so quality can't quietly degrade.

This is different from `src/core/monitoring/`, which tracks *live production* traffic.
Here we measure quality offline against known-correct answers.

## The three levels

A RAG pipeline can fail independently at three stages, so we score all three:

| Level | Question it answers | Metrics |
|-------|---------------------|---------|
| **1. Retrieval** | Did we fetch the right documents? | recall@k, precision@k, MRR, NDCG, MAP |
| **2. Generation** | Was the answer grounded in the retrieved context, and did it address the question? | faithfulness, answer relevance (LLM-as-judge) |
| **3. Embeddings** | Does the embedding model still separate relevant from irrelevant text? | cosine separation guard |

They build on each other: if retrieval (1) is broken, generation (2) can't be right; the
embedding guard (3) catches a silent embedding-model swap that would quietly sink level 1.

## How a question flows through

For each line in `golden.jsonl`:

1. **Retrieve** with the real bounded `retrieve_evidence` path (the same retrieval the agent's
   `retrieve_documents` tool calls) → rank the returned filenames against the labelled
   `relevant_filenames` → recall/precision/MRR/NDCG/MAP. The eval scores retrieval **directly**,
   not through the agent's tool loop, so the numbers reflect retrieval quality rather than the
   LLM's tool-choice behaviour.
2. **Generate** an answer over the retrieved context using the eval's own `GENERATION_*` prompts
   (`src/core/agent/prompts.py`), decoupled from the serving agent loop → an LLM judge scores
   the answer's **faithfulness** (no hallucination vs. context) and **relevance** (does it
   answer the question), each 1–5, normalised to 0–1.
3. Separately, the **embedding guard** checks that each question embeds closer to its
   relevant document than to an irrelevant one.

Aggregates are compared to `THRESHOLDS` in `run_eval.py`; any gated metric below its bar
fails the run. `precision@5` is reported but not gated (with one relevant doc per question
it's capped at 1/k, so it reflects the question mix, not a regression).

## Files

| File | Role |
|------|------|
| `corpus/*.txt` | The fixed document set that gets ingested |
| `golden.jsonl` | Labelled questions: `question`, `relevant_filenames` |
| `graph_cases.jsonl` | Live graph cases for out-of-corpus, partial, multipart, and follow-up behavior |
| `ranking.py` | Pure retrieval metrics (recall@k, precision@k, MRR, MAP, NDCG) |
| `judges.py` | LLM-as-judge faithfulness + answer-relevance scorers |
| `embeddings_check.py` | Cosine separation guard for the embedding model |
| `run_eval.py` | Orchestrator: ingest → score all three levels → report → gate |
| `run_graph_eval.py` | Manual live graph path/citation evaluation against both case sets |
| `tests/integration/test_agent_graph.py` | Deterministic compiled-graph route and citation contracts |

The pure pieces (`ranking`, `judges`, `embeddings_check`) are unit-tested with no infra in
`tests/unit/` and run on every PR for free. The full `run_eval.py` needs live services.

## Two tiers: what runs where

The checks split by what's stable vs noisy, and they run in different places:

- **Retrieval + embeddings (default).** Deterministic and cheap — embeddings only, no
  generation, no LLM judges. This is the CI alarm: *did a change break retrieval?* It can't
  flake, so it gates every push to main (`.github/workflows/eval.yml`).
- **Generation (`--full`).** Adds the real LLM answer plus LLM-as-judge faithfulness and
  answer relevance. Expensive and noisy (a small judge can score the same answer 1/5 or 3/5),
  so it is **local-only** — run it by hand when you change a prompt or model. The judges run
  on a small model (`JUDGE_MODEL` in `judges.py`); generation stays on the production model.
- **Live graph (`make eval-graph`).** Runs the compiled serving graph: query model, document
  tool, evidence assessment, and answer model. Every golden question is expected to end as
`document_answer` and cite all its labelled filenames. Web fallback is deliberately disabled:
needing it for a question whose evidence is in the fixed corpus is a failure signal. This is a
manual check, never a CI gate. Pass `--limit N` to run a shorter live smoke check.
- **Graph contract (`make eval-graph-contract`).** Runs the compiled graph with corpus artifacts
  and scripted model outputs. It verifies document-answer, web-fallback, and abstention routes,
  evidence-source validation, and selected-evidence isolation. It has no service, API-key, or
  model dependency and runs as its own CI step on every PR.

## Running it

Needs Qdrant up. Retrieval needs `OPENAI_API_KEY` (embeddings); `--full` also needs
`OPENROUTER_API_KEY` (the LLM). Both live in `.env`.

```bash
make up              # boot Qdrant
make eval-retrieval  # retrieval + embeddings only (what CI runs on push to main)
make eval            # the full thing: + generation + judges (local)
make eval-graph-contract  # deterministic compiled-graph contracts (CI)
make eval-graph      # compiled graph: expected document outcome + cited filenames (local)
uv run python -m evals.run_graph_eval --limit 3  # quick live-graph smoke check
```

CI's `rag eval gate` job runs `make eval-retrieval`'s command on every push to main, using
`OPENAI_API_KEY` from repository secrets. Its normal check workflow also runs
`make eval-graph-contract` on every PR without any external service. Generation quality is
never run in CI.

## Adding a golden question

Append a line to `golden.jsonl`:

```json
{"question": "…", "relevant_filenames": ["some_doc.txt"]}
```

Use two or more `relevant_filenames` for questions answerable by multiple documents — those
are what make `precision@k` meaningful. To make the gate stricter, add questions whose
wrong-but-similar documents could plausibly be retrieved.
