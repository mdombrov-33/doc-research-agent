from src.core.retrieval.fusion_retriever import FusionRetriever


DOCS = [
    "machine learning is a subset of artificial intelligence",
    "neural networks are inspired by the human brain",
    "retrieval augmented generation combines search with LLMs",
    "vector databases store high dimensional embeddings",
    "BM25 is a keyword based ranking algorithm",
]


def test_returns_all_docs():
    retriever = FusionRetriever(alpha=0.6)
    vector_scores = [0.9, 0.7, 0.5, 0.3, 0.1]
    results = retriever.fuse_results(DOCS, vector_scores, "machine learning")
    assert len(results) == len(DOCS)


def test_results_sorted_descending():
    retriever = FusionRetriever(alpha=0.6)
    vector_scores = [0.9, 0.7, 0.5, 0.3, 0.1]
    results = retriever.fuse_results(DOCS, vector_scores, "retrieval")
    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True)


def test_alpha_1_preserves_vector_ranking():
    # At alpha=1.0 only vector scores matter — top doc must be the one with highest vector score
    retriever = FusionRetriever(alpha=1.0)
    # Doc index 2 has the highest vector score
    vector_scores = [0.3, 0.5, 0.95, 0.1, 0.2]
    results = retriever.fuse_results(DOCS, vector_scores, "anything")
    top_idx = results[0][0]
    assert top_idx == 2


def test_empty_documents_returns_empty():
    retriever = FusionRetriever(alpha=0.6)
    results = retriever.fuse_results([], [], "query")
    assert results == []


def test_single_document():
    retriever = FusionRetriever(alpha=0.6)
    results = retriever.fuse_results(["only doc"], [0.8], "query")
    assert len(results) == 1
    assert results[0][0] == 0


def test_fused_score_within_bounds():
    retriever = FusionRetriever(alpha=0.6)
    vector_scores = [0.9, 0.4, 0.1]
    results = retriever.fuse_results(DOCS[:3], vector_scores, "neural networks")
    for _, score in results:
        assert 0.0 <= score <= 1.0


def test_equal_vector_scores_still_ranks_by_bm25():
    # With equal vector scores, BM25 should differentiate.
    # Need 5+ docs so IDF is nonzero (log((N-df+0.5)/(df+0.5)) > 0 when df < N/2).
    retriever = FusionRetriever(alpha=0.5)
    docs = [
        "cooking recipes pasta sauce tomato",
        "weather forecast rain umbrella clouds",
        "football match score goals league",
        "stock market prices trading finance",
        "retrieval augmented generation vector search embeddings",
    ]
    vector_scores = [0.5, 0.5, 0.5, 0.5, 0.5]  # identical — BM25 must decide
    results = retriever.fuse_results(docs, vector_scores, "retrieval vector search")
    top_idx = results[0][0]
    assert top_idx == 4  # last doc contains all query terms
