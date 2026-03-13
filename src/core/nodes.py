from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from src.config import get_settings
from src.core import prompts
from src.core.grading.graders import (
    grade_documents_batch,
    route_and_rewrite,
)
from src.core.retrieval.fusion_retriever import FusionRetriever
from src.core.state import AgentState
from src.core.tools import get_vector_store_tool, get_web_search_tool
from src.utils.logger import logger

settings = get_settings()


def get_llm():
    api_key = settings.get_llm_api_key()
    model = settings.get_llm_model()

    if settings.LLM_PROVIDER == "openrouter":
        llm = ChatOpenAI(
            api_key=SecretStr(api_key),
            base_url="https://openrouter.ai/api/v1",
            model=model,
            temperature=0.7,
        )
    else:
        llm = ChatOpenAI(
            api_key=SecretStr(api_key),
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


def router_node(state: AgentState) -> dict[str, bool | str]:
    logger.info("--- ROUTING QUERY ---")

    question = state.get("question", "")
    explicit_web_request = detect_explicit_web_search(question)

    result = route_and_rewrite(question)

    if explicit_web_request:
        logger.info("Routing to vector store (explicit web search request)")
        return {"web_search": False, "explicit_web_search": True, "question": result.rewritten_query}
    elif result.datasource == "websearch":
        logger.info("Routing to web search")
        return {"web_search": True, "explicit_web_search": False, "question": result.rewritten_query}
    else:
        logger.info("Routing to vector store")
        return {"web_search": False, "explicit_web_search": False, "question": result.rewritten_query}


def retrieve_node(state: AgentState) -> dict[str, list[str] | int]:
    logger.info("--- RETRIEVING FROM VECTOR STORE ---")

    question = state.get("question", "")

    vector_store = get_vector_store_tool()
    results = vector_store.similarity_search_with_score(question, k=5)

    doc_contents = []
    vector_scores = []

    for doc, score in results:
        content = doc.page_content if hasattr(doc, "page_content") else str(doc)
        doc_contents.append(content)
        vector_scores.append(float(score))

    docs_retrieved_total = len(doc_contents)

    logger.info(f"Retrieved {len(doc_contents)} documents from vector search")
    if vector_scores:
        logger.info(
            f"Vector scores: min={min(vector_scores):.4f}, "
            f"max={max(vector_scores):.4f}, "
            f"mean={sum(vector_scores) / len(vector_scores):.4f}"
        )

    # Step 3: Fusion retrieval (combine vector + BM25 scores)
    # Filter out empty documents first
    non_empty_docs = [doc for doc in doc_contents if doc and doc.strip()]

    if len(non_empty_docs) < len(doc_contents):
        logger.warning(f"Filtered out {len(doc_contents) - len(non_empty_docs)} empty documents")
        doc_contents = non_empty_docs

    if doc_contents and len(doc_contents) > 0:
        fusion = FusionRetriever(alpha=0.6)
        try:
            fused_results = fusion.fuse_results(
                doc_contents, vector_scores[: len(doc_contents)], question
            )
            doc_contents = [doc_contents[idx] for idx, score in fused_results]
            logger.info(f"Reranked documents using fusion (top score: {fused_results[0][1]:.4f})")
        except Exception as e:
            logger.warning(f"Fusion failed: {e}, using vector scores only")
    else:
        logger.warning("No non-empty documents for fusion, skipping")

    return {"documents": doc_contents, "docs_retrieved_total": docs_retrieved_total}


def web_search_node(state: AgentState) -> dict[str, list[str]]:
    logger.info("--- WEB SEARCH ---")

    question = state.get("question", "")
    existing_docs = state.get("documents", [])  # Keep relevant docs from vector store
    web_search = get_web_search_tool()

    try:
        result = web_search.invoke(question)
        web_docs = [result]
        logger.info(f"Web search completed, got {len(web_docs)} results")
    except Exception as e:
        logger.error(f"Web search failed: {e}")
        web_docs = []

    combined = existing_docs + web_docs
    logger.info(
        f"Combined {len(existing_docs)} vector docs + {len(web_docs)} web docs = {len(combined)} total"  # noqa: E501
    )

    return {"documents": combined}


def grade_documents_node(state: AgentState) -> dict[str, list[str] | bool | int]:
    logger.info("--- GRADING DOCUMENTS ---")

    question = state.get("question", "")
    documents = state.get("documents", [])
    attempts = state.get("retrieval_attempts", 0)
    explicit_web = state.get("explicit_web_search", False)

    if attempts == 0:
        scores = grade_documents_batch(question, documents)

        filtered_docs = []
        for doc, score in zip(documents, scores):
            if score == "yes":
                filtered_docs.append(doc)

        # Pure adaptive threshold: need at least 1 relevant doc OR explicit web request
        web_search_needed = explicit_web or len(filtered_docs) == 0

        logger.info(
            f"Filtered to {len(filtered_docs)} relevant documents. "
            f"Web search needed: {web_search_needed} "
            f"(explicit_web={explicit_web}, has_relevant_docs={len(filtered_docs) > 0})"
        )

        return {
            "documents": filtered_docs,
            "web_search": web_search_needed,
            "retrieval_attempts": attempts + 1,
        }
    else:
        existing_count = len([d for d in documents if d])
        logger.info(f"Grading {existing_count} total documents (vector + web)")

        scores = grade_documents_batch(question, documents)

        filtered_docs = []
        for doc, score in zip(documents, scores):
            if score == "yes":
                filtered_docs.append(doc)

        logger.info(
            f"Filtered to {len(filtered_docs)} relevant documents. Web search needed: False"
        )

        return {
            "documents": filtered_docs,
            "web_search": False,
            "retrieval_attempts": attempts + 1,
        }


def generate_node(state: AgentState) -> dict[str, str | int | list]:
    logger.info("--- GENERATING ANSWER ---")

    question = state.get("question", "")
    documents = state.get("documents", [])
    attempts = state.get("generation_attempts", 0)
    chat_history = state.get("chat_history", [])

    context = "\n\n".join(documents)

    llm = get_llm()

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


def decide_to_generate(state: AgentState) -> str:
    logger.info("--- DECIDING TO GENERATE OR WEB SEARCH ---")

    web_search = state.get("web_search", False)
    attempts = state.get("retrieval_attempts", 0)

    if web_search and attempts < 2:
        logger.info("Decision: Need web search")
        return "websearch"
    else:
        logger.info("Decision: Generate answer")
        return "generate"


