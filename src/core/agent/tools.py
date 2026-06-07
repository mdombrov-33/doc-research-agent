from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from src.core.retrieval.search import hybrid_search
from src.utils.logger import logger

RETRIEVE_TOP_K = 5


def format_docs(docs: list[dict]) -> str:
    """Render docs for the model, labelling each by its source (file vs web)."""
    if not docs:
        return "No documents found."
    parts = []
    for doc in docs:
        label = (
            "[Web Search]"
            if doc.get("source") == "web"
            else f"[Document: {doc.get('filename', 'unknown')}]"
        )
        parts.append(f"{label}\n{doc['content']}")
    return "\n\n".join(parts)


# response_format="content_and_artifact": the string goes into the ToolMessage the model
# reads; the structured docs ride along as the ToolMessage.artifact, which the grade node
# pulls back out to score relevance and build source metadata.
@tool(response_format="content_and_artifact")
def retrieve_documents(query: str, config: RunnableConfig) -> tuple[str, list[dict]]:
    """Search the user's uploaded documents with hybrid vector + keyword search.

    Use this first for almost any question about the user's documents. `query` should be a
    focused search query (resolve vague follow-ups into standalone terms before calling)."""
    # config is injected by ToolNode, not exposed to the model. top_k is the per-request value.
    top_k = (config.get("configurable") or {}).get("top_k") or RETRIEVE_TOP_K
    docs = hybrid_search(query, top_k)
    logger.info("retrieve_tool", count=len(docs))
    return format_docs(docs), docs


@tool(response_format="content_and_artifact")
def web_search(query: str) -> tuple[str, list[dict]]:
    """Search the live web for current or external information not in the user's documents.

    Use only when the documents lack the answer or the question needs up-to-date facts."""
    try:
        result = str(DuckDuckGoSearchRun().invoke(query))
    except Exception as e:
        logger.error("web_search_tool_failed", error=str(e))
        return "Web search failed.", []
    doc = {
        "content": result,
        "filename": "web",
        "chunk_index": 0,
        "chunk_length": len(result),
        "source": "web",
    }
    logger.info("web_search_tool", chars=len(result))
    return result, [doc]


TOOLS = [retrieve_documents, web_search]
