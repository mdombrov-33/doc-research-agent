from src.config import get_settings
from src.core import prompts
from src.core.exceptions import FusionRetrievalError
from src.core.grading.graders import (
    grade_documents_batch,
    route_and_rewrite,
)
from src.core.llm import get_llm
from src.core.retrieval.fusion_retriever import FusionRetriever
from src.core.state import AgentState
from src.core.tools import get_vector_store_tool, get_web_search_tool
from src.utils.logger import logger

settings = get_settings()


def router_node(state: AgentState) -> dict:
    logger.info("--- ROUTING QUERY ---")

    question = state.get("question", "")
    result = route_and_rewrite(question, model=state.get("model"))

    web_search = result.datasource == "websearch"

    if web_search:
        logger.info("Routing to vectorstore + web search (parallel)")
    else:
        logger.info("Routing to vectorstore only")

    return {
        "web_search": web_search,
        "web_search_done": False,
        "question": result.rewritten_query,
        "raw_documents": None,
        "docs_retrieved_total": None,
    }


def retrieve_node(state: AgentState) -> dict[str, list[dict] | int]:
    logger.info("--- RETRIEVING FROM VECTOR STORE ---")

    question = state.get("question", "")

    top_k = state.get("top_k") or 5
    vector_store = get_vector_store_tool()
    results = vector_store.similarity_search_with_score(question, k=top_k)

    doc_items = []
    vector_scores = []

    for doc, score in results:
        content = doc.page_content if hasattr(doc, "page_content") else str(doc)
        metadata = doc.metadata if hasattr(doc, "metadata") else {}
        doc_items.append(
            {
                "content": content,
                "filename": metadata.get("filename", "unknown"),
                "chunk_index": metadata.get("chunk_index", 0),
                "chunk_length": metadata.get("chunk_length", len(content)),
                "source": "vectorstore",
            }
        )
        vector_scores.append(float(score))

    docs_retrieved_total = len(doc_items)

    logger.info(f"Retrieved {len(doc_items)} documents from vector search")
    if vector_scores:
        logger.info(
            f"Vector scores: min={min(vector_scores):.4f}, "
            f"max={max(vector_scores):.4f}, "
            f"mean={sum(vector_scores) / len(vector_scores):.4f}"
        )

    non_empty_items = [item for item in doc_items if item["content"].strip()]

    if len(non_empty_items) < len(doc_items):
        logger.warning(f"Filtered out {len(doc_items) - len(non_empty_items)} empty documents")
        doc_items = non_empty_items

    if doc_items:
        contents = [item["content"] for item in doc_items]
        fusion = FusionRetriever(alpha=0.6)
        try:
            fused_results = fusion.fuse_results(contents, vector_scores[: len(contents)], question)
            doc_items = [doc_items[idx] for idx, score in fused_results]
            logger.info(f"Reranked documents using fusion (top score: {fused_results[0][1]:.4f})")
        except FusionRetrievalError as e:
            logger.warning(f"Fusion failed: {e}, using vector scores only")
    else:
        logger.warning("No non-empty documents for fusion, skipping")

    return {"raw_documents": doc_items, "docs_retrieved_total": docs_retrieved_total}


def web_search_node(state: AgentState) -> dict[str, list[dict] | int]:
    logger.info("--- WEB SEARCH ---")

    question = state.get("question", "")
    web_search = get_web_search_tool()

    web_docs: list[dict] = []
    try:
        result = web_search.invoke(question)
        content = str(result)
        web_docs = [
            {
                "content": content,
                "filename": "web",
                "chunk_index": 0,
                "chunk_length": len(content),
                "source": "web",
            }
        ]
        logger.info(f"Web search completed, got {len(web_docs)} results")
    except Exception as e:
        logger.error(f"Web search failed: {e}")

    return {
        "raw_documents": web_docs,
        "docs_retrieved_total": len(web_docs),
        "web_search_done": True,
    }


def grade_documents_node(state: AgentState) -> dict[str, list[dict] | None]:
    logger.info("--- GRADING DOCUMENTS ---")

    question = state.get("question", "")
    documents = state.get("raw_documents", [])

    if not documents:
        logger.warning("No documents to grade")
        return {"documents": []}

    contents = [doc["content"] for doc in documents]
    scores = grade_documents_batch(question, contents)
    filtered_docs = [doc for doc, score in zip(documents, scores) if score == "yes"]

    logger.info(f"Filtered to {len(filtered_docs)} relevant documents from {len(documents)}")

    if len(filtered_docs) < 3 and not state.get("web_search_done", False):
        logger.info(f"Only {len(filtered_docs)} relevant docs, triggering web search fallback")
        return {"documents": [], "raw_documents": None}

    return {"documents": filtered_docs}


def generate_node(state: AgentState) -> dict[str, str | list]:
    logger.info("--- GENERATING ANSWER ---")

    question = state.get("question", "")
    documents = state.get("documents", [])
    chat_history = state.get("chat_history", [])

    context = "\n\n".join(doc["content"] for doc in documents)

    llm = get_llm(state.get("model"), temperature=0.7)

    messages = [
        {"role": "system", "content": prompts.GENERATION_SYSTEM_PROMPT.format(context=context)},
        *chat_history,
        {"role": "user", "content": prompts.GENERATION_USER_PROMPT.format(question=question)},
    ]

    response = llm.invoke(messages)

    generation = response.content if isinstance(response.content, str) else str(response.content)

    logger.info(f"Generated answer: {len(generation)} chars")

    updated_history = chat_history + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": generation},
    ]

    return {
        "generation": generation,
        "chat_history": updated_history,
    }
