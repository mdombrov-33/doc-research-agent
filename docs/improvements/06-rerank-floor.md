# 06 — Calibrated rerank-score floor

## Goal

Known regression: irrelevant chunks can ride into the answer context and the displayed
sources (architecture §7 note, past sources-regression incident). A raw score cutoff was
deliberately avoided because it was uncalibrated — so calibrate it on the golden set.

## Decided design

- **Sweep first**: a small script beside `evals/ranking.py` (with a make target) sweeps the
  cross-encoder score threshold over the golden set's labeled relevant/irrelevant chunks and
  reports, per candidate floor, how many irrelevant chunks are dropped vs. recall lost. Pick
  the value that drops irrelevant chunks at ~zero recall cost. Note: the cross-encoder
  (`Xenova/ms-marco-MiniLM-L-6-v2`) outputs raw logits, so the floor is a logit value.
- **Floor applies in `rerank()` to the evidence pool** — chunks below it are excluded before
  assessment/answer ever see them. Fixes the root cause, not just the display.
- Edge case is the desired behavior: if every chunk falls below the floor, assessment sees no
  evidence → insufficient → web fallback/abstain — consistent with evidence control.
- Display-only and two-floor variants were rejected (cosmetic / over-engineered).
- **Env `RERANK_SCORE_FLOOR`**, default `None` = current behavior. The swept value is set as
  the deployed default via env docs, not hardcoded.

## Files

- `src/core/retrieval/rerank.py` — floor filter after scoring.
- `src/config.py` — `RERANK_SCORE_FLOOR: float | None = None`.
- `evals/` + `Makefile` — sweep script and target.

## Verification

- Run the sweep; record the chosen floor and its tradeoff numbers in this file.
- Unit tests: floor excludes below-threshold docs; `None` preserves current behavior; empty
  result propagates.
- `make eval-retrieval` with the floor set: no recall regression.
- `make test`, `make lint`.

## Definition of done

Sweep numbers recorded here; code + tests green; `docs/architecture.md` §9 (reranking — the
"no uncalibrated cutoff" note gets updated to "calibrated floor") and env example docs
updated; `progress.md` marked.
