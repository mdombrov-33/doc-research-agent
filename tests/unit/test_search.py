from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest
from qdrant_client import models
from qdrant_client.common.client_exceptions import ResourceExhaustedResponse
from qdrant_client.http.exceptions import ResponseHandlingException

from src.core.retrieval import search


class _Doc:
    def __init__(self, content, metadata=None):
        self.page_content = content
        self.metadata = metadata or {}


# --------------------------- _entity_filter ---------------------------


def test_entity_filter_none_for_empty():
    assert search._entity_filter([]) is None


def test_entity_filter_builds_match_any():
    flt = search._entity_filter(["Acme", "Bob"])
    expected = models.Filter(
        must=[
            models.FieldCondition(
                key="metadata.entities",
                match=models.MatchAny(any=["Acme", "Bob"]),
            )
        ]
    )
    assert flt == expected


# --------------------------- _fetch_k ---------------------------


def test_fetch_k_scales_with_top_k_and_is_capped():
    s = SimpleNamespace(RERANK_ENABLED=True, RERANK_MULTIPLIER=4, RERANK_FETCH_CAP=100)
    assert search._fetch_k(5, s) == 20
    assert search._fetch_k(40, s) == 100  # 40 * 4 = 160, capped to 100


def test_fetch_k_returns_top_k_when_reranking_disabled():
    s = SimpleNamespace(RERANK_ENABLED=False, RERANK_MULTIPLIER=4, RERANK_FETCH_CAP=100)
    assert search._fetch_k(5, s) == 5


# --------------------------- hybrid_search ---------------------------


def test_hybrid_search_maps_results_and_skips_blank(monkeypatch):
    store = MagicMock()
    store.similarity_search_with_score.return_value = [
        (
            _Doc(
                "real content",
                {
                    "document_id": "doc-a",
                    "chunk_id": "doc-a:2",
                    "filename": "a.pdf",
                    "chunk_index": 2,
                    "chunk_length": 12,
                },
            ),
            0.9,
        ),
        (_Doc("   ", {"filename": "blank.pdf"}), 0.5),  # blank → skipped
    ]
    monkeypatch.setattr(search, "get_retrieval_vector_store", lambda: store)
    monkeypatch.setattr(search, "extract_entities", lambda q: [])
    monkeypatch.setattr(search, "rerank", lambda q, docs, k: docs[:k])

    out = search.hybrid_search("q", top_k=5)

    assert len(out) == 1
    assert out[0]["filename"] == "a.pdf"
    assert out[0]["document_id"] == "doc-a"
    assert out[0]["chunk_id"] == "doc-a:2"
    assert out[0]["source"] == "document"


def test_hybrid_search_logs_counts_not_query_text(monkeypatch):
    store = MagicMock()
    store.similarity_search_with_score.return_value = []
    logger = MagicMock()
    monkeypatch.setattr(search, "get_retrieval_vector_store", lambda: store)
    monkeypatch.setattr(search, "extract_entities", lambda _query: ["Acme"])
    monkeypatch.setattr(
        search,
        "get_settings",
        lambda: SimpleNamespace(RERANK_ENABLED=False, RERANK_MULTIPLIER=4, RERANK_FETCH_CAP=100),
    )
    monkeypatch.setattr(search, "logger", logger)

    search.hybrid_search("private rollout details for Acme", top_k=5)

    logged = [call.kwargs for call in logger.info.call_args_list]
    assert logged == [
        {
            "top_k": 5,
            "fetch_k": 5,
            "query_entity_count": 1,
            "entity_supplement_active": False,
        },
        {
            "count": 0,
            "fetch_k": 5,
            "blank_results": 0,
            "query_entity_count": 1,
            "entity_supplement_count": 0,
        },
    ]


@pytest.mark.parametrize(
    "transient_error",
    [
        ResponseHandlingException(httpx.ReadTimeout("timed out")),
        ResourceExhaustedResponse("rate limited", retry_after_s=1),
    ],
)
def test_hybrid_search_retries_a_transient_qdrant_failure_once(monkeypatch, transient_error):
    store = MagicMock()
    store.similarity_search_with_score.side_effect = [
        transient_error,
        [(_Doc("recovered", {"filename": "recovered.txt"}), 0.9)],
    ]
    monkeypatch.setattr(search, "get_retrieval_vector_store", lambda: store)
    monkeypatch.setattr(search, "extract_entities", lambda q: [])
    monkeypatch.setattr(
        search,
        "get_settings",
        lambda: SimpleNamespace(RERANK_ENABLED=False, RERANK_MULTIPLIER=4, RERANK_FETCH_CAP=100),
    )

    out = search.hybrid_search("q", top_k=1)

    assert [doc["content"] for doc in out] == ["recovered"]
    assert store.similarity_search_with_score.call_count == 2


@pytest.mark.parametrize(
    ("question", "entity", "answer_chunk", "entity_chunk"),
    [
        (
            "What are Saturn's rings made of and how thin are they?",
            "Saturn",
            ("saturn:0", "rings are water ice and only tens of meters thick"),
            ("saturn:1", "rings formed from a moon or comet"),
        ),
        (
            "What chunk size and overlap does the article recommend for a RAG pipeline?",
            "RAG",
            ("rag:chunking", "chunks should be 500 to 800 tokens with 10 to 15 percent overlap"),
            ("rag:evaluation", "evaluate retrieval with recall and nDCG"),
        ),
        (
            "Why does adding a cross-encoder reranker improve a RAG pipeline?",
            "RAG",
            ("rag:reranking", "a cross-encoder puts the best candidate first"),
            ("rag:evaluation", "evaluate retrieval with recall and nDCG"),
        ),
        (
            "Which London coffee house grew into a famous insurance market?",
            "London",
            ("coffee:lloyd", "Edward Lloyd's coffee house became Lloyd's of London"),
            ("coffee:stock-exchange", "London stock dealers met at Jonathan's and Garraway's"),
        ),
    ],
)
def test_hybrid_search_preserves_unfiltered_candidates_with_entity_supplement(
    monkeypatch, question, entity, answer_chunk, entity_chunk
):
    store = MagicMock()
    # The entity match is real but incomplete. The answer chunk is only in the unfiltered pool
    # because ingestion did not tag it with the query entity.
    store.similarity_search_with_score.side_effect = [
        [
            (_Doc(answer_chunk[1], {"chunk_id": answer_chunk[0]}), 0.9),
            (_Doc("unrelated broad match", {"chunk_id": "other:0"}), 0.8),
        ],
        [
            (_Doc(answer_chunk[1], {"chunk_id": answer_chunk[0]}), 0.9),
            (_Doc(entity_chunk[1], {"chunk_id": entity_chunk[0]}), 0.7),
        ],
    ]
    monkeypatch.setattr(search, "get_retrieval_vector_store", lambda: store)
    monkeypatch.setattr(search, "extract_entities", lambda q: [entity])
    monkeypatch.setattr(
        search,
        "get_settings",
        lambda: SimpleNamespace(RERANK_ENABLED=True, RERANK_MULTIPLIER=4, RERANK_FETCH_CAP=100),
    )
    captured = {}

    def fake_rerank(question, docs, top_k):
        captured["chunk_ids"] = [doc["chunk_id"] for doc in docs]
        return docs[:top_k]

    monkeypatch.setattr(search, "rerank", fake_rerank)

    out = search.hybrid_search(question, top_k=2)

    assert store.similarity_search_with_score.call_count == 2
    assert store.similarity_search_with_score.call_args_list[0].kwargs == {"k": 8}
    assert store.similarity_search_with_score.call_args_list[1].kwargs["k"] == 2
    entity_call = store.similarity_search_with_score.call_args_list[1]
    assert entity_call.kwargs["filter"] == search._entity_filter([entity])
    assert captured["chunk_ids"] == [answer_chunk[0], "other:0", entity_chunk[0]]
    assert out[0]["chunk_id"] == answer_chunk[0]


def test_hybrid_search_overfetches_pool_and_truncates_to_top_k(monkeypatch):
    store = MagicMock()
    # Eight candidate chunks come back from hybrid search, in retrieval order.
    store.similarity_search_with_score.return_value = [
        (_Doc(f"doc {i}", {"filename": f"{i}.txt"}), 1.0 - i * 0.1) for i in range(8)
    ]
    monkeypatch.setattr(search, "get_retrieval_vector_store", lambda: store)
    monkeypatch.setattr(search, "extract_entities", lambda q: [])
    monkeypatch.setattr(
        search,
        "get_settings",
        lambda: SimpleNamespace(RERANK_ENABLED=True, RERANK_MULTIPLIER=4, RERANK_FETCH_CAP=100),
    )

    captured = {}

    def fake_rerank(question, docs, top_k):
        captured["candidates"] = len(docs)
        captured["top_k"] = top_k
        return docs[:top_k]

    monkeypatch.setattr(search, "rerank", fake_rerank)

    out = search.hybrid_search("q", top_k=3)

    # The pool requested from the store scales with top_k (3 * multiplier 4 = 12).
    assert store.similarity_search_with_score.call_args.kwargs["k"] == 12
    # The reranker saw every returned candidate and was asked for the user's top_k.
    assert captured["candidates"] == 8
    assert captured["top_k"] == 3
    # The final result is truncated to the user's top_k.
    assert len(out) == 3
