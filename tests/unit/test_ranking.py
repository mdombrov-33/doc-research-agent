import pytest

from src.evals import ranking

# Reference case: relevances [0, 1, 0, 1] with 3 relevant docs in the corpus.
#   recall@4    = 2 / 3
#   precision@4 = 2 / 4
#   RR          = 1 / 2  (first hit at rank 2)
#   AP          = (1/2 + 2/4) / 3        = 0.33333...
#   DCG@4       = 1/log2(3) + 1/log2(5)  = 0.63093 + 0.43068 = 1.06160
#   IDCG@4      = 1/log2(2) + 1/log2(3) + 1/log2(4) = 2.13093   (ideal: 3 ones up front)
#   NDCG@4      = 1.06160 / 2.13093      = 0.49819
_RELS = [0, 1, 0, 1]
_TOTAL = 3


def test_recall_at_k():
    assert ranking.recall_at_k(_RELS, _TOTAL, 4) == pytest.approx(2 / 3)
    assert ranking.recall_at_k(_RELS, _TOTAL, 1) == pytest.approx(0.0)


def test_precision_at_k():
    assert ranking.precision_at_k(_RELS, 4) == pytest.approx(0.5)
    assert ranking.precision_at_k(_RELS, 2) == pytest.approx(0.5)


def test_reciprocal_rank():
    assert ranking.reciprocal_rank(_RELS) == pytest.approx(0.5)
    assert ranking.reciprocal_rank([0, 0, 0]) == 0.0
    assert ranking.reciprocal_rank([1, 0]) == 1.0


def test_average_precision():
    assert ranking.average_precision(_RELS, _TOTAL) == pytest.approx(1 / 3)


def test_ndcg_at_k():
    assert ranking.ndcg_at_k(_RELS, _TOTAL, 4) == pytest.approx(0.49819, abs=1e-4)


def test_perfect_ranking_scores_one():
    rels = [1, 1, 1]
    assert ranking.recall_at_k(rels, 3, 3) == 1.0
    assert ranking.precision_at_k(rels, 3) == 1.0
    assert ranking.reciprocal_rank(rels) == 1.0
    assert ranking.average_precision(rels, 3) == pytest.approx(1.0)
    assert ranking.ndcg_at_k(rels, 3, 3) == pytest.approx(1.0)


def test_no_relevant_results_scores_zero():
    rels = [0, 0, 0]
    assert ranking.recall_at_k(rels, 2, 3) == 0.0
    assert ranking.reciprocal_rank(rels) == 0.0
    assert ranking.average_precision(rels, 2) == 0.0
    assert ranking.ndcg_at_k(rels, 2, 3) == 0.0


def test_empty_corpus_guards():
    assert ranking.recall_at_k([], 0, 5) == 0.0
    assert ranking.average_precision([], 0) == 0.0
    assert ranking.ndcg_at_k([], 0, 5) == 0.0


def test_mean_average_precision():
    queries = [([1, 0], 1), ([0, 1], 1)]  # AP = 1.0 and 0.5
    assert ranking.mean_average_precision(queries) == pytest.approx(0.75)
    assert ranking.mean_average_precision([]) == 0.0
