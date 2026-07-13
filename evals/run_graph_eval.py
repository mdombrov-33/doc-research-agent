"""Manual live evaluation of the complete document-answer graph.

It rebuilds the isolated eval corpus, then runs each golden question through the real query,
retrieval, assessment, and answer nodes. A case passes only when it finishes as a document answer
and cites every labelled document. Web fallback is disabled: needing it for a known-corpus
question is the failure signal this evaluation is meant to expose.
"""

import argparse
import asyncio
import os
import uuid
from dataclasses import dataclass
from typing import Any
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.core.agent.outcomes import FinalOutcome, normalize_outcome
from src.core.citations import citations_referenced_by_answer


@dataclass(frozen=True)
class GraphResult:
    question: str
    expected_filenames: frozenset[str]
    outcome: FinalOutcome
    cited_filenames: frozenset[str]
    error: str | None = None

    @property
    def passed(self) -> bool:
        return (
            self.error is None
            and self.outcome == "document_answer"
            and self.expected_filenames <= self.cited_filenames
        )


def evaluate_graph_state(row: dict[str, Any], state: dict[str, Any]) -> GraphResult:
    """Evaluate one completed graph state against its labelled document evidence."""
    messages = state.get("messages", [])
    artifacts = [
        artifact
        for message in messages
        if isinstance(message, ToolMessage) and isinstance(message.artifact, list)
        for artifact in message.artifact
        if isinstance(artifact, dict)
    ]
    answer = next(
        (
            message.content
            for message in reversed(messages)
            if isinstance(message, AIMessage)
            and not message.tool_calls
            and isinstance(message.content, str)
        ),
        "",
    )
    cited_filenames = frozenset(
        citation.title
        for citation in citations_referenced_by_answer(artifacts, answer)
        if citation.source_type == "document"
    )
    return GraphResult(
        question=row["question"],
        expected_filenames=frozenset(row["relevant_filenames"]),
        outcome=normalize_outcome(state.get("outcome")),
        cited_filenames=cited_filenames,
    )


def evaluate_live_graph(graph: Any, golden: list[dict[str, Any]]) -> list[GraphResult]:
    """Invoke the supplied compiled graph once per golden question."""
    results: list[GraphResult] = []
    for index, row in enumerate(golden, start=1):
        print(f"[{index}/{len(golden)}] {row['question'][:72]}", flush=True)
        try:
            state = graph.invoke(
                {"messages": [HumanMessage(content=row["question"])]},
                config={"configurable": {"thread_id": str(uuid.uuid4())}},
            )
            results.append(evaluate_graph_state(row, state))
        except Exception as error:
            results.append(
                GraphResult(
                    question=row["question"],
                    expected_filenames=frozenset(row["relevant_filenames"]),
                    outcome="abstained",
                    cited_filenames=frozenset(),
                    error=type(error).__name__,
                )
            )
    return results


def _disabled_web_fallback(_: str) -> tuple[str, list[dict]]:
    return "Web fallback disabled during graph evaluation.", []


def _render(results: list[GraphResult]) -> None:
    print(f"Running live graph eval: {len(results)} questions")
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        expected = ", ".join(sorted(result.expected_filenames))
        cited = ", ".join(sorted(result.cited_filenames)) or "none"
        details = f"outcome={result.outcome}; expected={expected}; cited={cited}"
        if result.error:
            details += f"; error={result.error}"
        print(f"{status:<4} {result.question[:62]:<64} {details}")

    passed = sum(result.passed for result in results)
    print(f"\nDocument-answer coverage: {passed}/{len(results)}")


def main() -> int:
    # Set before importing the eval helpers or application settings so this never touches the
    # serving collection.
    os.environ["QDRANT_COLLECTION_NAME"] = "documents_eval"
    os.environ.setdefault("LOG_LEVEL", "WARNING")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, help="run only the first N golden questions")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")

    from langgraph.checkpoint.memory import MemorySaver

    from evals.run_eval import _ingest_corpus, _load_golden, _reset_collection
    from src.core.agent import nodes
    from src.core.agent.graph import build_graph

    _reset_collection()
    asyncio.run(_ingest_corpus())
    golden = _load_golden()
    if args.limit is not None:
        golden = golden[: args.limit]
    graph = build_graph(checkpointer=MemorySaver())
    with patch.object(nodes, "search_web", _disabled_web_fallback):
        results = evaluate_live_graph(graph, golden)
    _render(results)
    return 0 if all(result.passed for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
