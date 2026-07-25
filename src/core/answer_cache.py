"""Two-layer answer cache backed by Qdrant.

A repeated first-turn ``document_answer`` is served from here instead of re-running the graph.
One collection holds both cache layers in each entry: the question-embedding vector serves L2
semantic lookup, and the payload's ``question_hash`` serves L1 exact lookup.

Correctness is enforced by ``corpus_version``: every non-duplicate upload bumps it (see
``bump_corpus_version``), and lookups only match the current version — so a new document can
never surface a stale cached answer. A 24 h ``created_at`` floor is a growth/staleness backstop.

Functions are synchronous (Qdrant + embeddings clients are sync); the async stream handler runs
them in a worker thread.
"""

import json
import time
import uuid
from hashlib import sha256

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Condition,
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    Range,
    VectorParams,
)

from src.config import get_settings
from src.core.vectorstore import get_embeddings, get_ingestion_qdrant_client
from src.utils.logger import logger

_NAMESPACE = "default"
_TTL_SECONDS = 24 * 60 * 60
_L2_MIN_SCORE = 0.95

# Reserved point holding the current corpus_version. Its nil-UUID id and lack of a question_hash
# or model payload keep it out of every lookup filter; the vector is an arbitrary unit sentinel
# (a zero vector is invalid under cosine).
_VERSION_POINT_ID = "00000000-0000-0000-0000-000000000000"


def _client() -> QdrantClient:
    return get_ingestion_qdrant_client()


def _collection() -> str:
    return get_settings().ANSWER_CACHE_COLLECTION


def _normalize(question: str) -> str:
    return " ".join(question.strip().lower().split())


def _question_hash(question: str) -> str:
    return sha256(_normalize(question).encode("utf-8")).hexdigest()


def _key_filters(model: str, top_k: int, version: int) -> list[Condition]:
    """Filters shared by both lookup layers: the request knobs, version, tenancy, and TTL."""
    return [
        FieldCondition(key="model", match=MatchValue(value=model)),
        FieldCondition(key="top_k", match=MatchValue(value=top_k)),
        FieldCondition(key="corpus_version", match=MatchValue(value=version)),
        FieldCondition(key="namespace", match=MatchValue(value=_NAMESPACE)),
        FieldCondition(key="created_at", range=Range(gte=time.time() - _TTL_SECONDS)),
    ]


def _entry(payload: dict | None) -> dict:
    payload = payload or {}
    return {"answer": payload["answer"], "sources": json.loads(payload["sources"])}


def ensure_answer_cache_collection() -> None:
    client = _client()
    collection = _collection()
    if not client.collection_exists(collection):
        logger.info("answer_cache_collection_creating", collection=collection)
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(
                size=get_settings().EMBEDDING_DIMENSION,
                distance=Distance.COSINE,
            ),
        )
        for field, schema in (
            ("question_hash", PayloadSchemaType.KEYWORD),
            ("model", PayloadSchemaType.KEYWORD),
            ("namespace", PayloadSchemaType.KEYWORD),
            ("top_k", PayloadSchemaType.INTEGER),
            ("corpus_version", PayloadSchemaType.INTEGER),
            ("created_at", PayloadSchemaType.FLOAT),
        ):
            client.create_payload_index(
                collection_name=collection, field_name=field, field_schema=schema
            )
    _ensure_version_point(client, collection)


def _version_sentinel_vector() -> list[float]:
    return [1.0] + [0.0] * (get_settings().EMBEDDING_DIMENSION - 1)


def _ensure_version_point(client: QdrantClient, collection: str) -> None:
    existing = client.retrieve(collection, ids=[_VERSION_POINT_ID], with_payload=True)
    if not existing:
        _write_version_point(client, collection, 1)


def _write_version_point(client: QdrantClient, collection: str, version: int) -> None:
    client.upsert(
        collection_name=collection,
        points=[
            PointStruct(
                id=_VERSION_POINT_ID,
                vector=_version_sentinel_vector(),
                payload={"corpus_version": version, "namespace": _NAMESPACE},
            )
        ],
    )


def current_corpus_version() -> int:
    points = _client().retrieve(_collection(), ids=[_VERSION_POINT_ID], with_payload=True)
    if not points:
        return 1
    return int((points[0].payload or {})["corpus_version"])


def bump_corpus_version() -> int:
    """Advance the corpus version and drop now-stale entries. Called after a real upload."""
    client = _client()
    collection = _collection()
    new_version = current_corpus_version() + 1
    _write_version_point(client, collection, new_version)
    client.delete(
        collection_name=collection,
        points_selector=Filter(
            must=[FieldCondition(key="corpus_version", range=Range(lt=new_version))]
        ),
    )
    logger.info("answer_cache_version_bumped", corpus_version=new_version)
    return new_version


def lookup(question: str, model: str, top_k: int) -> dict | None:
    """Return the cached ``{answer, sources}`` for this question, or None on a miss."""
    if not get_settings().ANSWER_CACHE_ENABLED:
        return None
    client = _client()
    collection = _collection()
    version = current_corpus_version()
    shared = _key_filters(model, top_k, version)

    hash_match: Condition = FieldCondition(
        key="question_hash", match=MatchValue(value=_question_hash(question))
    )
    exact = client.scroll(
        collection_name=collection,
        scroll_filter=Filter(must=[hash_match, *shared]),
        limit=1,
        with_payload=True,
    )[0]
    if exact:
        return _entry(exact[0].payload)

    result = client.query_points(
        collection_name=collection,
        query=get_embeddings().embed_query(question),
        query_filter=Filter(must=shared),
        limit=1,
        score_threshold=_L2_MIN_SCORE,
        with_payload=True,
    )
    if result.points:
        return _entry(result.points[0].payload)
    return None


def store(question: str, answer: str, sources: list[dict], model: str, top_k: int) -> None:
    if not get_settings().ANSWER_CACHE_ENABLED:
        return
    _client().upsert(
        collection_name=_collection(),
        points=[
            PointStruct(
                id=str(uuid.uuid4()),
                vector=get_embeddings().embed_query(question),
                payload={
                    "question_hash": _question_hash(question),
                    "question": question,
                    "answer": answer,
                    "sources": json.dumps(sources),
                    "model": model,
                    "top_k": top_k,
                    "corpus_version": current_corpus_version(),
                    "namespace": _NAMESPACE,
                    "created_at": time.time(),
                },
            )
        ],
    )
