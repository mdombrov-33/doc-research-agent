from collections.abc import Iterable, Mapping
from typing import Any

from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from opentelemetry import trace

from src.core.citations import citation_from_artifact
from src.core.retrieval.search import hybrid_search
from src.utils.logger import logger

_tracer = trace.get_tracer(__name__)


def format_docs(docs: list[dict]) -> str:
    """Render evidence for the model with the stable source IDs it may cite in its answer."""
    if not docs:
        return "No documents found."
    parts = []
    for doc in docs:
        citation = citation_from_artifact(doc)
        label = (
            f"[Web: {doc.get('title', 'unknown')}]"
            if doc.get("source") == "web"
            else f"[Document: {doc.get('filename', 'unknown')}]"
        )
        source_id = f"[Source ID: {citation.source_id}]" if citation else "[Source ID: unavailable]"
        parts.append(f"{source_id}\n{label}\n{doc['content']}")
    return "\n\n".join(parts)


# response_format="content_and_artifact": the string goes into the ToolMessage the model
# reads; the structured docs ride along as ToolMessage.artifact for source metadata.
@tool(response_format="content_and_artifact")
def retrieve_documents(query: str, config: RunnableConfig) -> tuple[str, list[dict]]:
    """Search the user's uploaded documents with hybrid vector + keyword search.

    Use this first for almost any question about the user's documents. `query` should be a
    focused search query (resolve vague follow-ups into standalone terms before calling)."""
    top_k = (config.get("configurable") or {}).get("top_k") or 10
    with _tracer.start_as_current_span("tool.retrieve_documents") as span:
        span.set_attribute("tool.query", query)
        span.set_attribute("tool.top_k", top_k)
        docs = hybrid_search(query, top_k)
        span.set_attribute("tool.docs_returned", len(docs))
    logger.info("retrieve_tool", count=len(docs))
    return format_docs(docs), docs


@tool(response_format="content_and_artifact")
def web_search(query: str) -> tuple[str, list[dict]]:
    """Search the live web for current or external information not in the user's documents.

    Use only when the documents lack the answer or the question needs up-to-date facts."""
    with _tracer.start_as_current_span("tool.web_search") as span:
        span.set_attribute("tool.query", query)
        try:
            results = DuckDuckGoSearchResults(num_results=5, output_format="list").invoke(query)
        except Exception as e:
            span.set_status(trace.StatusCode.ERROR, str(e))
            logger.error("web_search_tool_failed", error=str(e))
            return "Web search failed.", []
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


TOOLS = [retrieve_documents, web_search]
