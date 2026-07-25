"""Calibrate the rerank score floor (item 06) on the golden set.

The cross-encoder scores every retrieved chunk (raw logits). A relevance floor drops chunks
below a threshold before they reach the evidence pool. This script sweeps candidate thresholds
and reports, per floor, how many labelled-irrelevant chunks it removes vs. how much retrieval
recall it costs — so we can pick the highest floor that drops irrelevant chunks at ~zero recall
cost and set it as the deployed `RERANK_SCORE_FLOOR` default.

A chunk is "relevant" iff its source file is in the query's `relevant_filenames`, matching how
`run_eval` scores retrieval. Reuses that module's corpus ingestion so the pipeline is identical.

Requires live Qdrant + OpenAI embeddings. Run from the repo root:
  uv run python -m evals.rerank_sweep
"""

import asyncio

# run_eval sets QDRANT_COLLECTION_NAME + LOG_LEVEL at import time; import it before anything
# under src/ so retrieval hits the isolated eval collection.
from evals import ranking
from evals.run_eval import (
    TOP_K,
    K,
    _ingest_corpus,
    _load_golden,
    _reset_collection,
)
from src.config import get_settings
from src.core.retrieval.rerank import _get_cross_encoder
from src.core.retrieval.search import hybrid_search

STEPS = 25  # candidate floors, evenly spaced across the observed score range


def _dedup(filenames: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for name in filenames:
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered


def _scored_pool(question: str) -> list[tuple[str, float]]:
    """The full reranked candidate pool as (filename, cross-encoder score) in ranked order."""
    settings = get_settings()
    docs = hybrid_search(question, settings.RERANK_FETCH_CAP)
    encoder = _get_cross_encoder(settings.RERANK_MODEL)
    scores = list(encoder.rerank(question, [doc["content"] for doc in docs]))
    return [(doc["filename"], float(score)) for doc, score in zip(docs, scores)]


def _recall_at_floor(
    pool: list[tuple[str, float]], relevant: set[str], floor: float | None
) -> float:
    """Document recall@K after applying the floor and the serving top-k chunk slice."""
    kept = [(name, score) for name, score in pool if floor is None or score >= floor]
    ranked = _dedup([name for name, _ in kept[:TOP_K]])
    relevances = [1 if name in relevant else 0 for name in ranked]
    return ranking.recall_at_k(relevances, len(relevant), K)


def main() -> int:
    _reset_collection()
    asyncio.run(_ingest_corpus())

    golden = _load_golden()
    print(f"Sweeping rerank floor over {len(golden)} questions\n")

    # Score every query's pool once, then evaluate every candidate floor against it.
    pools = [(set(row["relevant_filenames"]), _scored_pool(row["question"])) for row in golden]

    all_scores = [score for _, pool in pools for _, score in pool]
    rel_scores = [s for rel, pool in pools for name, s in pool if name in rel]
    irrel_scores = [s for rel, pool in pools for name, s in pool if name not in rel]
    lo, hi = min(all_scores), max(all_scores)
    floors = [lo + (hi - lo) * i / STEPS for i in range(STEPS + 1)]

    baseline_recall = sum(_recall_at_floor(pool, rel, None) for rel, pool in pools) / len(pools)
    print(f"chunks: {len(rel_scores)} relevant, {len(irrel_scores)} irrelevant")
    print(f"score range: [{lo:.3f}, {hi:.3f}]   baseline recall@{K}: {baseline_recall:.3f}\n")

    header = f"{'floor':>8}  {'irrel_dropped':>13}  {'rel_dropped':>11}  {'recall@' + str(K):>9}"
    print(header)
    print("-" * len(header))
    for floor in floors:
        irrel_dropped = sum(1 for s in irrel_scores if s < floor)
        rel_dropped = sum(1 for s in rel_scores if s < floor)
        recall = sum(_recall_at_floor(pool, rel, floor) for rel, pool in pools) / len(pools)
        flag = "  <- recall regressed" if recall < baseline_recall else ""
        print(
            f"{floor:8.3f}  {irrel_dropped:5d}/{len(irrel_scores):<7d}  "
            f"{rel_dropped:4d}/{len(rel_scores):<6d}  {recall:9.3f}{flag}"
        )

    # Recommend the highest floor that costs no recall.
    safe = [
        floor
        for floor in floors
        if sum(_recall_at_floor(pool, rel, floor) for rel, pool in pools) / len(pools)
        >= baseline_recall
    ]
    if safe:
        best = max(safe)
        dropped = sum(1 for s in irrel_scores if s < best)
        print(
            f"\nRecommended floor: {best:.3f} "
            f"(drops {dropped}/{len(irrel_scores)} irrelevant chunks at no recall cost)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
