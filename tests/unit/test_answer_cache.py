import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.config import Settings
from src.core import answer_cache


def _point(payload: dict) -> SimpleNamespace:
    return SimpleNamespace(payload=payload)


@pytest.fixture
def client(monkeypatch):
    fake = MagicMock()
    # Default: the reserved version point reports corpus_version=1.
    fake.retrieve.return_value = [_point({"corpus_version": 1})]
    fake.scroll.return_value = ([], None)
    fake.query_points.return_value = SimpleNamespace(points=[])
    monkeypatch.setattr(answer_cache, "get_ingestion_qdrant_client", lambda: fake)
    return fake


@pytest.fixture
def embeddings(monkeypatch):
    fake = MagicMock()
    fake.embed_query.return_value = [0.1, 0.2, 0.3]
    monkeypatch.setattr(answer_cache, "get_embeddings", lambda: fake)
    return fake


def test_question_hash_ignores_case_and_whitespace():
    assert answer_cache._question_hash("  Hello   World ") == answer_cache._question_hash(
        "hello world"
    )


def test_exact_hit_skips_the_embedding_call(client, embeddings):
    client.scroll.return_value = (
        [_point({"answer": "42", "sources": json.dumps([{"id": "S1"}])})],
        None,
    )

    result = answer_cache.lookup("q", model="a/x")

    assert result == {"answer": "42", "sources": [{"id": "S1"}]}
    embeddings.embed_query.assert_not_called()
    client.query_points.assert_not_called()


def test_semantic_hit_on_exact_miss(client, embeddings):
    client.query_points.return_value = SimpleNamespace(
        points=[_point({"answer": "42", "sources": json.dumps([])})]
    )

    result = answer_cache.lookup("q", model="a/x")

    assert result == {"answer": "42", "sources": []}
    embeddings.embed_query.assert_called_once_with("q")
    threshold = client.query_points.call_args.kwargs["score_threshold"]
    assert threshold == answer_cache._L2_MIN_SCORE


def test_miss_returns_none(client, embeddings):
    assert answer_cache.lookup("q", model="a/x") is None


def test_lookup_filters_on_model_and_current_version(client, embeddings):
    client.retrieve.return_value = [_point({"corpus_version": 7})]

    answer_cache.lookup("q", model="a/x")

    conditions = client.scroll.call_args.kwargs["scroll_filter"].must
    matched = {c.key: c.match.value for c in conditions if getattr(c, "match", None)}
    assert matched["model"] == "a/x"
    assert matched["corpus_version"] == 7


def test_lookup_disabled_returns_none_without_touching_qdrant(client, embeddings, monkeypatch):
    monkeypatch.setattr(answer_cache, "get_settings", lambda: Settings(ANSWER_CACHE_ENABLED=False))

    assert answer_cache.lookup("q", model="a/x") is None
    client.scroll.assert_not_called()


def test_store_upserts_entry_with_current_version(client, embeddings):
    client.retrieve.return_value = [_point({"corpus_version": 4})]

    answer_cache.store("q", answer="42", sources=[{"id": "S1"}], model="a/x")

    payload = client.upsert.call_args.kwargs["points"][0].payload
    assert payload["model"] == "a/x"
    assert payload["corpus_version"] == 4
    assert payload["namespace"] == "default"
    assert json.loads(payload["sources"]) == [{"id": "S1"}]


def test_bump_increments_version_and_deletes_stale_entries(client, embeddings):
    client.retrieve.return_value = [_point({"corpus_version": 3})]

    new_version = answer_cache.bump_corpus_version()

    assert new_version == 4
    written = client.upsert.call_args.kwargs["points"][0].payload
    assert written["corpus_version"] == 4
    stale_filter = client.delete.call_args.kwargs["points_selector"]
    assert stale_filter.must[0].range.lt == 4


def test_bump_when_no_version_point_starts_from_one(client, embeddings):
    client.retrieve.return_value = []

    assert answer_cache.bump_corpus_version() == 2


def test_existing_cache_with_legacy_retrieval_width_is_reset(client, embeddings):
    client.collection_exists.side_effect = [True, False]
    client.get_collection.return_value = SimpleNamespace(
        payload_schema={"top_k": SimpleNamespace()}
    )

    answer_cache.ensure_answer_cache_collection()

    client.delete_collection.assert_called_once_with("answer_cache")
    client.create_collection.assert_called_once()
    indexed_fields = {
        call.kwargs["field_name"] for call in client.create_payload_index.call_args_list
    }
    assert indexed_fields == {
        "question_hash",
        "model",
        "namespace",
        "corpus_version",
        "created_at",
    }
