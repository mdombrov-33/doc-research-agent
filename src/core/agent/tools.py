import json
import random
import time
from collections.abc import Iterable, Mapping
from typing import Annotated, Any

from ddgs import DDGS
from ddgs.exceptions import DDGSException, RatelimitException, TimeoutException
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from opentelemetry import trace
from pydantic import Field

from src.config import get_settings
from src.core.agent.state import AgentState
from src.core.citations import citation_from_artifact
from src.core.retrieval.search import retrieve_evidence
from src.utils.logger import logger

_tracer = trace.get_tracer(__name__)


def _is_transient_web_error(error: Exception) -> bool:
    if isinstance(error, (RatelimitException, TimeoutException)):
        return True
    return (
        isinstance(error, DDGSException)
        and bool(error.args)
        and isinstance(error.args[0], (RatelimitException, TimeoutException))
    )


def _ddgs_text_results(query: str, timeout_seconds: int) -> list[dict[str, Any]]:
    with DDGS(timeout=timeout_seconds) as ddgs:
        results = ddgs.text(
            query,
            region="wt-wt",
            safesearch="moderate",
            timelimit="y",
            max_results=5,
            backend="auto",
        )
    return [
        {
            "title": result["title"],
            "link": result["href"],
            "snippet": result["body"],
        }
        for result in results
    ]


def format_docs(docs: list[dict]) -> str:
    """Render source data as explicitly untrusted evidence for the model."""
    if not docs:
        return "No documents found."
    evidence = []
    for doc in docs:
        citation = citation_from_artifact(doc)
        is_web = doc.get("source") == "web"
        evidence.append(
            {
                "source_id": citation.source_id if citation else "unavailable",
                "source_type": "web" if is_web else "document",
                "title": str(doc.get("title" if is_web else "filename", "unknown")),
                "content": str(doc["content"]),
            }
        )
    return (
        "<untrusted_evidence_json>\n"
        + json.dumps(evidence, ensure_ascii=False)
        + "\n</untrusted_evidence_json>"
    )


def artifact_documents(artifact: Any) -> list[dict]:
    """Return citable documents from either retrieval or web tool artifacts."""
    if isinstance(artifact, list):
        return [item for item in artifact if isinstance(item, dict)]
    if isinstance(artifact, Mapping):
        documents = artifact.get("documents")
        if isinstance(documents, list):
            return [item for item in documents if isinstance(item, dict)]
    return []


def artifact_retrieval_metrics(artifact: Any) -> dict[str, int | str]:
    if not isinstance(artifact, Mapping):
        return {}
    metrics = artifact.get("retrieval")
    if not isinstance(metrics, Mapping):
        return {}
    return {
        key: value
        for key, value in metrics.items()
        if key in {"query_count", "query_shape", "candidates", "evidence", "reranker_calls"}
        and isinstance(value, (int, str))
    }


def _current_question(state: AgentState) -> str:
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage) and isinstance(message.content, str):
            return message.content
    raise ValueError("retrieval requires a current user question")


# response_format="content_and_artifact": the string goes into the ToolMessage the model
# reads; the structured docs ride along as ToolMessage.artifact for source metadata.
@tool(response_format="content_and_artifact")
def retrieve_documents(
    search_queries: Annotated[
        list[str],
        Field(
            min_length=1,
            max_length=2,
            description="One focused query, or two only for a genuinely multipart question.",
        ),
    ],
    state: Annotated[AgentState, InjectedState],
) -> tuple[str, dict[str, Any]]:
    """Run one bounded search plan over the user's uploaded documents.

    Resolve vague follow-ups into standalone terms. Use one query for a simple question and no
    more than two focused queries when separate parts genuinely need different searches.
    """
    current_question = _current_question(state)
    with _tracer.start_as_current_span("tool.retrieve_documents") as span:
        span.set_attribute("tool.query_count", len(search_queries))
        result = retrieve_evidence(search_queries, current_question)
        span.set_attribute("tool.candidates", result.metrics.candidates)
        span.set_attribute("tool.docs_returned", len(result.documents))
    logger.info("retrieve_tool", **result.metrics.as_dict())
    artifact = {
        "documents": result.documents,
        "retrieval": result.metrics.as_dict(),
    }
    return format_docs(result.documents), artifact


@tool(response_format="content_and_artifact")
def web_search(query: str) -> tuple[str, list[dict]]:
    """Search the live web for current or external information not in the user's documents.

    Use only when the documents lack the answer or the question needs up-to-date facts."""
    return search_web(query)


def search_web(query: str) -> tuple[str, list[dict]]:
    """Search the web and return model context plus independently citable artifacts."""
    timeout_seconds = get_settings().WEB_SEARCH_TIMEOUT_SECONDS
    with _tracer.start_as_current_span("tool.web_search") as span:
        span.set_attribute("tool.timeout_seconds", timeout_seconds)
        for attempt in range(1, 3):
            try:
                results = _ddgs_text_results(query, timeout_seconds)
                break
            except Exception as error:
                if attempt == 2 or not _is_transient_web_error(error):
                    span.set_status(trace.StatusCode.ERROR, "web search unavailable")
                    logger.error("web_search_tool_failed", failure_type=type(error).__name__)
                    return "Web search failed.", []
                logger.warning(
                    "web_search_retry",
                    attempt=attempt,
                    failure_type=type(error).__name__,
                )
                time.sleep(random.uniform(0.05, 0.25))
        docs = _web_evidence(results)
        span.set_attribute("tool.results_returned", len(docs))
    result = format_docs(docs)
    logger.info("web_search_tool", count=len(docs))
    return result, docs


def _web_evidence(results: Any) -> list[dict]:
    """Retain only DuckDuckGo results that can become honest, independently citable evidence."""
    if not isinstance(results, Iterable) or isinstance(results, (str, bytes, Mapping)):
        return []

    documents: list[dict] = []
    for rank, result in enumerate(results, start=1):
        if not isinstance(result, Mapping):
            continue
        title = result.get("title")
        url = result.get("link")
        snippet = result.get("snippet")
        if not isinstance(title, str) or not title.strip():
            continue
        if not isinstance(url, str) or not url.strip():
            continue
        if not isinstance(snippet, str) or not snippet.strip():
            continue
        evidence = {
            "content": snippet,
            "title": title.strip(),
            "url": url.strip(),
            "rank": rank,
            "source": "web",
        }
        if citation_from_artifact(evidence) is not None:
            documents.append(evidence)
    return documents
