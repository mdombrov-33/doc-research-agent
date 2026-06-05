"""
Level 3 - Embeddings

Does the model still separate relevant from irrelevant?

A direct sanity check on the embedding model, independent of Qdrant and chunking: for a
set of (query, relevant_text, irrelevant_text) triples, the query embedding must be more
similar to the relevant text than to the irrelevant one. If a future embedding-model swap
silently degrades quality, the separation collapses and this guard fails.
"""

from typing import Protocol

import numpy as np


class Embedder(Protocol):
    def embed_query(self, text: str) -> list[float]: ...
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...


def cosine(a: list[float], b: list[float]) -> float:
    va, vb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    return float(np.dot(va, vb) / denom) if denom else 0.0


def check_separation(
    triples: list[tuple[str, str, str]],
    embedder: Embedder,
    margin: float = 0.0,
) -> tuple[int, int]:
    """Return (passed, total). A triple passes when
    cosine(query, relevant) > cosine(query, irrelevant) + margin.
    """
    passed = 0
    for query, relevant_text, irrelevant_text in triples:
        q = embedder.embed_query(query)
        rel, irr = embedder.embed_documents([relevant_text, irrelevant_text])
        if cosine(q, rel) > cosine(q, irr) + margin:
            passed += 1
    return passed, len(triples)
