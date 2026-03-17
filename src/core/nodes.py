from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from src.config import get_settings
from src.core import prompts
from src.core.exceptions import FusionRetrievalError
from src.core.grading.graders import (
    grade_documents_batch,
    route_and_rewrite,
)
from src.core.retrieval.fusion_retriever import FusionRetriever
from src.core.state import AgentState
from src.core.tools import get_vector_store_tool, get_web_search_tool
from src.utils.logger import logger

settings = get_settings()


def get_llm(model_override: str | None = None):
    model = model_override or settings.get_llm_model()
    logger.info(f"Using model: {model}")

    # Auto-detect provider: OpenRouter models contain '/', OpenAI models don't
    if model_override and "/" in model_override:
        llm = ChatOpenAI(
            api_key=SecretStr(settings.OPENROUTER_API_KEY),
            base_url="https://openrouter.ai/api/v1",
            model=model,
            temperature=0.7,
        )
    elif settings.LLM_PROVIDER == "openrouter":
        llm = ChatOpenAI(
            api_key=SecretStr(settings.OPENROUTER_API_KEY),
            base_url="https://openrouter.ai/api/v1",
            model=model,
            temperature=0.7,
        )
    else:
        llm = ChatOpenAI(
            api_key=SecretStr(settings.OPENAI_API_KEY),
            model=model,
            temperature=0.7,
        )

    return llm


def detect_explicit_web_search(question: str) -> bool:
    explicit_phrases = [
        "web search",
        "search web",
        "check web",
        "online search",
        "search online",
        "google",
        "search google",
        "look online",
        "check internet",
        "both storage and web",
        "also search",
    ]

    if any(phrase in question.lower() for phrase in explicit_phrases):
        logger.info("Explicit web search detected via pattern match")
        return True

    recent_indicators = ["today", "latest", "recent", "current", "now", "breaking", "this week", "this month"]

    if any(word in question.lower() for word in recent_indicators):
        logger.info("Recent info indicator detected, confirming with LLM...")

        llm = get_llm()
        prompt = f"""Determine if this question requires real-time or recent information from the web.

Answer YES if:
- Question asks about current events, breaking news, or today's information
- Question needs up-to-date data (prices, weather, scores, etc.)
- Question explicitly mentions time periods requiring recent data

Answer NO if:
- Question is about general knowledge or historical facts
- Time reference is not critical to answering

Question: "{question}"

Should we use web search for current information? Answer only YES or NO:"""

        response = llm.invoke([{"role": "user", "content": prompt}])
        answer = str(response.content).strip().upper()

        logger.info(f"LLM web search decision: {answer}")
        return "YES" in answer

    return False


def router_node(state: AgentState) -> dict:
    logger.info("--- ROUTING QUERY ---")

    question = state.get("question", "")
    explicit_web_request = detect_explicit_web_search(question)
    result = route_and_rewrite(question, model=state.get("model"))

    web_search = explicit_web_request or result.datasource == "websearch"

    if web_search:
        logger.info("Routing to vectorstore + web search (parallel)")
    else:
        logger.info("Routing to vectorstore only")

    return {
        "web_search": web_search,
        "question": result.rewritten_query,
        "raw_documents": None,
        "docs_retrieved_total": None,
    }


def retrieve_node(state: AgentState) -> dict[str, list[str] | int]:
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
        doc_items.append({
            "content": content,
            "filename": metadata.get("filename", "unknown"),
            "chunk_index": metadata.get("chunk_index", 0),
            "chunk_length": metadata.get("chunk_length", len(content)),
            "source": "vectorstore",
        })
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


def web_search_node(state: AgentState) -> dict[str, list[str] | int]:
    logger.info("--- WEB SEARCH ---")

    question = state.get("question", "")
    web_search = get_web_search_tool()

    web_docs: list[str] = []
    try:
        result = web_search.invoke(question)
        web_docs = [str(result)]
        logger.info(f"Web search completed, got {len(web_docs)} results")
    except Exception as e:
        logger.error(f"Web search failed: {e}")

    return {"raw_documents": web_docs, "docs_retrieved_total": len(web_docs)}


def grade_documents_node(state: AgentState) -> dict[str, list[dict]]:
    logger.info("--- GRADING DOCUMENTS ---")

    question = state.get("question", "")
    documents = state.get("raw_documents", [])

    if not documents:
        logger.warning("No documents to grade")
        return {"documents": []}

    contents = [doc["content"] for doc in documents]
    scores = grade_documents_batch(question, contents, model=state.get("model"))
    filtered_docs = [doc for doc, score in zip(documents, scores) if score == "yes"]

    logger.info(f"Filtered to {len(filtered_docs)} relevant documents from {len(documents)}")
    return {"documents": filtered_docs}


def generate_node(state: AgentState) -> dict[str, str | int | list]:
    logger.info("--- GENERATING ANSWER ---")

    question = state.get("question", "")
    documents = state.get("documents", [])
    attempts = state.get("generation_attempts", 0)
    chat_history = state.get("chat_history", [])

    context = "\n\n".join(doc["content"] for doc in documents)

    llm = get_llm(state.get("model"))

    messages = [
        {"role": "system", "content": prompts.GENERATION_SYSTEM_PROMPT.format(context=context)},
        *chat_history,
        {"role": "user", "content": prompts.GENERATION_USER_PROMPT.format(question=question)},
    ]

    response = llm.invoke(messages)

    generation = response.content if isinstance(response.content, str) else str(response.content)

    logger.info(f"Generated answer: {len(generation)} chars (attempt {attempts + 1})")

    updated_history = chat_history + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": generation},
    ]

    return {
        "generation": generation,
        "generation_attempts": attempts + 1,
        "chat_history": updated_history,
    }



