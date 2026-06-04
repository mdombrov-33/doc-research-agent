"""
Level 1 - Retrieval

Did we fetch the right docs?

Each function takes ``relevances`` - the binary relevance labels (1/0) of the
retrieved items in rank order — and, where recall is involved, ``total_relevant``,
the number of relevant items that exist in the corpus. Keeping these pure makes
them trivial to unit-test and decouples them from how relevance is matched.
"""

import math
from statistics import mean


def recall_at_k(relevances: list[int], total_relevant: int, k: int) -> float:
    if total_relevant <= 0:
        return 0.0
    return sum(relevances[:k]) / total_relevant


def precision_at_k(relevances: list[int], k: int) -> float:
    if k <= 0:
        return 0.0
    return sum(relevances[:k]) / k


def reciprocal_rank(relevances: list[int]) -> float:
    for rank, rel in enumerate(relevances, start=1):
        if rel:
            return 1.0 / rank
    return 0.0


def average_precision(relevances: list[int], total_relevant: int) -> float:
    if total_relevant <= 0:
        return 0.0
    hits = 0
    running = 0.0
    for rank, rel in enumerate(relevances, start=1):
        if rel:
            hits += 1
            running += hits / rank
    return running / total_relevant


def mean_average_precision(queries: list[tuple[list[int], int]]) -> float:
    """``queries`` is a list of (relevances, total_relevant) per query."""
    if not queries:
        return 0.0
    return mean(average_precision(rels, total) for rels, total in queries)


def _dcg(relevances: list[int], k: int) -> float:
    return sum(rel / math.log2(rank + 1) for rank, rel in enumerate(relevances[:k], start=1))


def ndcg_at_k(relevances: list[int], total_relevant: int, k: int) -> float:
    ideal = [1] * min(total_relevant, k)
    idcg = _dcg(ideal, k)
    if idcg == 0:
        return 0.0
    return _dcg(relevances, k) / idcg
