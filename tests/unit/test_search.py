import threading
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


def _settings(candidate_budget=40, evidence_budget=8):
    return SimpleNamespace(
        RETRIEVAL_CANDIDATE_BUDGET=candidate_budget,
        RETRIEVAL_EVIDENCE_BUDGET=evidence_budget,
    )


def _item(chunk_id: str) -> dict:
    return {
        "content": f"content {chunk_id}",
        "document_id": chunk_id.split(":")[0],
        "chunk_id": chunk_id,
        "filename": "doc.txt",
        "source": "document",
    }


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


def test_collect_candidates_maps_results_and_skips_blank(monkeypatch):
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
        (_Doc("   ", {"filename": "blank.pdf"}), 0.5),
    ]
    monkeypatch.setattr(search, "get_retrieval_vector_store", lambda: store)
    monkeypatch.setattr(search, "extract_entities", lambda _query: [])

    out = search._collect_candidates("q", candidate_budget=40, entity_budget=8)

    assert len(out) == 1
    assert out[0]["filename"] == "a.pdf"
    assert out[0]["document_id"] == "doc-a"
    assert out[0]["chunk_id"] == "doc-a:2"
    assert out[0]["source"] == "document"
    assert store.similarity_search_with_score.call_args.kwargs == {"k": 40}


@pytest.mark.parametrize(
    "transient_error",
    [
        ResponseHandlingException(httpx.ReadTimeout("timed out")),
        ResourceExhaustedResponse("rate limited", retry_after_s=1),
    ],
)
def test_collect_candidates_retries_a_transient_qdrant_failure_once(monkeypatch, transient_error):
    store = MagicMock()
    store.similarity_search_with_score.side_effect = [
        transient_error,
        [(_Doc("recovered", {"chunk_id": "doc:0"}), 0.9)],
    ]
    monkeypatch.setattr(search, "get_retrieval_vector_store", lambda: store)
    monkeypatch.setattr(search, "extract_entities", lambda _query: [])

    out = search._collect_candidates("q", candidate_budget=40, entity_budget=8)

    assert [doc["content"] for doc in out] == ["recovered"]
    assert store.similarity_search_with_score.call_count == 2


def test_collect_candidates_supplements_without_excluding_unfiltered_hits(monkeypatch):
    store = MagicMock()
    store.similarity_search_with_score.side_effect = [
        [
            (_Doc("answer", {"chunk_id": "doc:answer"}), 0.9),
            (_Doc("broad", {"chunk_id": "doc:broad"}), 0.8),
        ],
        [
            (_Doc("answer", {"chunk_id": "doc:answer"}), 0.9),
            (_Doc("entity", {"chunk_id": "doc:entity"}), 0.7),
        ],
    ]
    monkeypatch.setattr(search, "get_retrieval_vector_store", lambda: store)
    monkeypatch.setattr(search, "extract_entities", lambda _query: ["Acme"])

    out = search._collect_candidates("Acme answer", candidate_budget=40, entity_budget=8)

    assert [doc["chunk_id"] for doc in out] == ["doc:answer", "doc:broad", "doc:entity"]
    assert store.similarity_search_with_score.call_args_list[0].kwargs == {"k": 40}
    assert store.similarity_search_with_score.call_args_list[1].kwargs["k"] == 8
    assert store.similarity_search_with_score.call_args_list[1].kwargs[
        "filter"
    ] == search._entity_filter(["Acme"])


def test_merge_candidate_branches_is_rank_aware_deterministic_and_deduplicated():
    branch_a = [_item("doc:shared"), _item("doc:a"), _item("doc:c")]
    branch_b = [_item("doc:b"), _item("doc:shared"), _item("doc:d")]

    first = search._merge_candidate_branches([branch_a, branch_b], budget=4)
    second = search._merge_candidate_branches([branch_a, branch_b], budget=4)

    assert [doc["chunk_id"] for doc in first] == [
        "doc:shared",
        "doc:b",
        "doc:a",
        "doc:c",
    ]
    assert first == second
    assert len({doc["chunk_id"] for doc in first}) == 4


def test_two_query_branches_execute_concurrently(monkeypatch):
    barrier = threading.Barrier(2)

    def collect(query, _candidate_budget, _entity_budget):
        barrier.wait(timeout=1)
        return [_item(f"doc:{query}")]

    monkeypatch.setattr(search, "_collect_candidates", collect)
    monkeypatch.setattr(search, "get_settings", lambda: _settings())
    monkeypatch.setattr(search, "rerank", lambda _question, docs, limit: docs[:limit])

    result = search.retrieve_evidence(["first", "second"], "original question")

    assert result.metrics.query_shape == "multipart"
    assert result.metrics.query_count == 2
    assert len(result.documents) == 2


def test_retrieve_evidence_applies_global_candidate_budget_and_one_rerank(monkeypatch):
    branches = {
        "first": [_item(f"a:{index}") for index in range(35)],
        "second": [_item(f"b:{index}") for index in range(35)],
    }
    monkeypatch.setattr(
        search,
        "_collect_candidates",
        lambda query, _candidate_budget, _entity_budget: branches[query],
    )
    monkeypatch.setattr(search, "get_settings", lambda: _settings())
    rerank = MagicMock(side_effect=lambda _question, docs, limit: docs[:limit])
    monkeypatch.setattr(search, "rerank", rerank)

    result = search.retrieve_evidence(["first", "second"], "original question")

    rerank.assert_called_once()
    assert rerank.call_args.args[0] == "original question"
    assert len(rerank.call_args.args[1]) == 40
    assert rerank.call_args.args[2] == 8
    assert result.metrics.candidates == 40
    assert result.metrics.evidence == 8
    assert result.metrics.reranker_calls == 1


@pytest.mark.parametrize("queries", [[], ["one", "two", "three"], [" ", ""]])
def test_retrieve_evidence_rejects_unbounded_or_empty_plans(queries):
    with pytest.raises(ValueError, match="one or two"):
        search.retrieve_evidence(queries, "question")


def test_empty_candidate_pool_does_not_call_reranker(monkeypatch):
    monkeypatch.setattr(
        search, "_collect_candidates", lambda _query, _candidate_budget, _entity_budget: []
    )
    monkeypatch.setattr(search, "get_settings", lambda: _settings())
    rerank = MagicMock()
    monkeypatch.setattr(search, "rerank", rerank)

    result = search.retrieve_evidence(["question"], "question")

    assert result.documents == []
    assert result.metrics.candidates == 0
    assert result.metrics.evidence == 0
    assert result.metrics.reranker_calls == 0
    rerank.assert_not_called()


def test_duplicate_planned_queries_collapse_to_one_branch(monkeypatch):
    collect = MagicMock(return_value=[_item("doc:0")])
    monkeypatch.setattr(search, "_collect_candidates", collect)
    monkeypatch.setattr(search, "get_settings", lambda: _settings())
    monkeypatch.setattr(search, "rerank", lambda _question, docs, limit: docs[:limit])

    result = search.retrieve_evidence(["same query", "same query"], "question")

    collect.assert_called_once()
    assert result.metrics.query_count == 1
    assert result.metrics.query_shape == "single"
