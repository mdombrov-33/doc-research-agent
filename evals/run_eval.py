"""Offline RAG evaluation against a golden set.

Two tiers, by what's stable vs noisy:

  default  — retrieval metrics + embedding separation. Deterministic and cheap (embeddings
             only, no generation, no LLM judges). This is the CI alarm: "did retrieval break?"
  --full   — also runs generation and LLM-as-judge faithfulness / answer relevance. Expensive
             and noisy, so it is a local-only check we run when we touch a prompt or model.

Ingests evals/corpus/ into an isolated Qdrant collection, scores each golden question, prints
a report, and exits non-zero if any gated aggregate is below threshold.

Requires live Qdrant + OpenAI embeddings (and, with --full, an OpenRouter LLM).
Run from the repo root:  uv run python -m evals.run_eval [--full]
"""

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

# Set before importing anything under src/ so get_settings() picks these up on first
# construction: an isolated collection (never touch production data) and a quiet log level
# (the pipeline logs per query, which would otherwise drown the report).
os.environ["QDRANT_COLLECTION_NAME"] = "documents_eval"
os.environ.setdefault("LOG_LEVEL", "WARNING")

import asyncio

from evals import judges, ranking
from evals.embeddings_check import check_separation
from src.config import get_settings
from src.core.agent.prompts import GENERATION_SYSTEM_PROMPT, GENERATION_USER_PROMPT
from src.core.ingestion.pipeline import process_and_store
from src.core.llm import get_llm
from src.core.nlp import get_spacy_model
from src.core.retrieval.search import retrieve_evidence
from src.core.vectorstore import (
    ensure_collection_exists,
    get_embeddings,
    get_ingestion_qdrant_client,
    get_ingestion_vector_store,
)

CORPUS_DIR = Path(__file__).parent / "corpus"
GOLDEN_PATH = Path(__file__).parent / "golden.jsonl"

K = 5  # k for the @k metrics

# Gated retrieval + embedding metrics (deterministic, run in CI).
RETRIEVAL_THRESHOLDS = {"recall@5": 0.8, "mrr": 0.7, "ndcg@5": 0.7, "map": 0.6}
EMBEDDING_THRESHOLD = {"embedding_separation": 1.0}
# Noisy LLM-judge metrics, only gated under --full (local).
GENERATION_THRESHOLDS = {"faithfulness": 0.75, "answer_relevance": 0.75}


@dataclass
class QueryResult:
    question: str
    relevances: list[int]  # 1/0 per retrieved document, in rank order
    total_relevant: int
    faithfulness: float | None = None  # normalized 0-1, only set under --full
    answer_relevance: float | None = None

    @property
    def recall(self) -> float:
        return ranking.recall_at_k(self.relevances, self.total_relevant, K)

    @property
    def precision(self) -> float:
        return ranking.precision_at_k(self.relevances, K)

    @property
    def mrr(self) -> float:
        return ranking.reciprocal_rank(self.relevances)

    @property
    def ndcg(self) -> float:
        return ranking.ndcg_at_k(self.relevances, self.total_relevant, K)


def _load_golden() -> list[dict]:
    return [json.loads(line) for line in GOLDEN_PATH.read_text().splitlines() if line.strip()]


def _reset_collection() -> None:
    client = get_ingestion_qdrant_client()
    try:
        client.delete_collection(get_settings().QDRANT_COLLECTION_NAME)
    except Exception:
        pass  # didn't exist yet
    ensure_collection_exists()


async def _ingest_corpus() -> None:
    vector_store = get_ingestion_vector_store()
    nlp = get_spacy_model()
    settings = get_settings()
    for path in sorted(CORPUS_DIR.glob("*.txt")):
        file_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        await process_and_store(str(path), path.name, file_sha256, vector_store, nlp, settings)


def _dedup(filenames: list[str]) -> list[str]:
    """First occurrence wins, preserving retrieval rank order."""
    seen: set[str] = set()
    ordered: list[str] = []
    for name in filenames:
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered


def _evaluate_query(row: dict, full: bool) -> QueryResult:
    relevant = set(row["relevant_filenames"])

    # Score the retrieval layer directly (ungraded), independent of the agent's tool loop.
    docs = retrieve_evidence([row["question"]], row["question"]).documents
    ranked = _dedup([doc["filename"] for doc in docs])
    result = QueryResult(
        question=row["question"],
        relevances=[1 if name in relevant else 0 for name in ranked],
        total_relevant=len(relevant),
    )

    if full:
        context = "\n\n".join(doc["content"] for doc in docs)
        messages = [
            {"role": "system", "content": GENERATION_SYSTEM_PROMPT.format(context=context)},
            {"role": "user", "content": GENERATION_USER_PROMPT.format(question=row["question"])},
        ]
        response = get_llm().invoke(messages)  # temp 0, matching the serving agent
        answer = response.content if isinstance(response.content, str) else str(response.content)
        result.faithfulness = judges.normalize(judges.judge_faithfulness(context, answer).score)
        result.answer_relevance = judges.normalize(
            judges.judge_answer_relevance(row["question"], answer).score
        )

    return result


def _embedding_separation(golden: list[dict]) -> float:
    """Fraction of (query, relevant, irrelevant) triples the embedder separates correctly."""
    files = {path.name: path.read_text() for path in CORPUS_DIR.glob("*.txt")}
    triples: list[tuple[str, str, str]] = []
    for row in golden:
        relevant = row["relevant_filenames"]
        irrelevant = next(name for name in sorted(files) if name not in relevant)
        triples.append((row["question"], files[relevant[0]], files[irrelevant]))

    passed, total = check_separation(triples, get_embeddings())
    return passed / total if total else 0.0


def _aggregate(results: list[QueryResult], embedding_separation: float, full: bool) -> dict:
    aggregate = {
        "recall@5": mean(r.recall for r in results),
        "precision@5": mean(r.precision for r in results),
        "mrr": mean(r.mrr for r in results),
        "ndcg@5": mean(r.ndcg for r in results),
        "map": ranking.mean_average_precision([(r.relevances, r.total_relevant) for r in results]),
        "embedding_separation": embedding_separation,
    }
    if full:
        aggregate["faithfulness"] = mean(
            r.faithfulness for r in results if r.faithfulness is not None
        )
        aggregate["answer_relevance"] = mean(
            r.answer_relevance for r in results if r.answer_relevance is not None
        )
    return aggregate


def _render(results: list[QueryResult], aggregate: dict, full: bool) -> None:
    header = f"\n{'question':<54} R@5  P@5  MRR  NDCG" + ("  Faith  Rel" if full else "")
    print(header)
    print("-" * len(header))
    for r in results:
        row = f"{r.question[:52]:<54}{r.recall:4.2f} {r.precision:4.2f} {r.mrr:4.2f} {r.ndcg:4.2f}"
        if full:
            row += f"  {r.faithfulness:4.2f}  {r.answer_relevance:4.2f}"
        print(row)

    print("\nRetrieval:")
    for name in ("recall@5", "precision@5", "mrr", "ndcg@5", "map"):
        print(f"  {name:<20} {aggregate[name]:.3f}")
    print("Embeddings:")
    print(f"  {'separation':<20} {aggregate['embedding_separation']:.3f}")
    if full:
        print("Generation:")
        for name in ("faithfulness", "answer_relevance"):
            print(f"  {name:<20} {aggregate[name]:.3f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full", action="store_true", help="also run generation + LLM judges (local only)"
    )
    args = parser.parse_args()

    _reset_collection()
    asyncio.run(_ingest_corpus())

    golden = _load_golden()
    print(f"Running {'full' if args.full else 'retrieval'} eval: {len(golden)} questions")
    results = [_evaluate_query(row, args.full) for row in golden]
    aggregate = _aggregate(results, _embedding_separation(golden), args.full)
    _render(results, aggregate, args.full)

    thresholds = {**RETRIEVAL_THRESHOLDS, **EMBEDDING_THRESHOLD}
    if args.full:
        thresholds |= GENERATION_THRESHOLDS

    failures = [
        f"{name} {aggregate[name]:.3f} < {threshold}"
        for name, threshold in thresholds.items()
        if aggregate.get(name, 0.0) < threshold
    ]
    if failures:
        print("\nFAILED thresholds:")
        for failure in failures:
            print(f"  {failure}")
        return 1

    print("\nAll thresholds met.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
