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

## Sweep result (recorded)

`make eval-rerank-sweep` over the 31-question golden set (147 relevant chunks, 1558 irrelevant;
cross-encoder score range `[-11.517, 8.662]`, baseline recall@5 = 1.000):

| floor | irrelevant dropped | relevant dropped | recall@5 |
| --- | --- | --- | --- |
| −11.517 (baseline) | 0 / 1558 | 0 / 147 | 1.000 |
| −9.902 | 1538 / 1558 | 33 / 147 | 1.000 |
| **−9.095** | **1542 / 1558** | 39 / 147 | **1.000** |
| −8.288 | 1545 / 1558 | 46 / 147 | 0.984 ← first regression |
| −1.831 | 1558 / 1558 | 88 / 147 | 0.887 |
| 8.662 | 1558 / 1558 | 146 / 147 | 0.032 |

recall@5 is document-level: dropping low-scored *relevant* chunks costs nothing while the
top-scored chunk per relevant document survives. Recall holds at 1.000 up to −9.095 and first
regresses at −8.288. **Deployed default: `RERANK_SCORE_FLOOR=-9.0`** (in `.env.example`) — a
round value just below the −8.288 cliff (~0.7-logit margin) that drops ~99% of labelled-irrelevant
chunks at zero recall cost on this set. The code default stays `None`. Recalibrate if the reranker
model changes.

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
