from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from src.config import get_settings
from src.core import prompts
from src.core.grading.graders import (
    check_hallucination,
    grade_answer_quality,
    grade_documents_batch,
    rewrite_query,
    route_question,
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


def router_node(state: AgentState) -> dict[str, bool]:
    logger.info("--- ROUTING QUERY ---")

    question = state.get("question", "")
    source = route_question(question)

    explicit_phrases = [
        "web search",
        "search web",
        "check web",
        "online search",
        "search online",
        "both storage and web",
        "also search",
    ]
    explicit_web_request = any(phrase in question.lower() for phrase in explicit_phrases)

    if explicit_web_request:
        logger.info("Routing to vector store (with explicit web search request)")
        return {"web_search": False, "explicit_web_search": True}
    elif source == "websearch":
        logger.info("Routing to web search (router decision)")
        return {"web_search": True, "explicit_web_search": False}
    else:
        logger.info("Routing to vector store")
        return {"web_search": False, "explicit_web_search": False}


def retrieve_node(state: AgentState) -> dict[str, list[str] | int]:
    logger.info("--- RETRIEVING FROM VECTOR STORE ---")

    question = state.get("question", "")

    preprocessed_query = rewrite_query(question)
    logger.info(f"Preprocessed query: '{question}' -> '{preprocessed_query}'")

    vector_store = get_vector_store_tool()
    results = vector_store.similarity_search_with_score(preprocessed_query, k=10)

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
                doc_contents, vector_scores[: len(doc_contents)], preprocessed_query
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

        threshold = settings.RELEVANCE_THRESHOLD
        web_search_needed = len(filtered_docs) < threshold or explicit_web

        logger.info(
            f"Filtered to {len(filtered_docs)} relevant documents (threshold: {threshold}). "
            f"Web search needed: {web_search_needed}"
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


def generate_node(state: AgentState) -> dict[str, str | int]:
    logger.info("--- GENERATING ANSWER ---")

    question = state.get("question", "")
    documents = state.get("documents", [])
    attempts = state.get("generation_attempts", 0)

    context = "\n\n".join(documents)

    llm = get_llm()

    messages = [
        {"role": "system", "content": prompts.GENERATION_SYSTEM_PROMPT.format(context=context)},
        {"role": "user", "content": prompts.GENERATION_USER_PROMPT.format(question=question)},
    ]

    response = llm.invoke(messages)

    generation = response.content if isinstance(response.content, str) else str(response.content)

    logger.info(f"Generated answer: {len(generation)} chars (attempt {attempts + 1})")

    return {"generation": generation, "generation_attempts": attempts + 1}


def rewrite_query_node(state: AgentState) -> dict[str, str]:
    logger.info("--- REWRITING QUERY ---")

    question = state.get("question", "")

    better_question = rewrite_query(question)

    return {"question": better_question}


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


def grade_generation_grounded_node(state: AgentState) -> dict[str, str]:
    logger.info("--- CHECKING HALLUCINATION ---")

    documents = state.get("documents", [])
    generation = state.get("generation", "")

    score = check_hallucination(documents, generation)

    return {"hallucination_grounded": score}


def grade_generation_grounded(state: AgentState) -> str:
    score = state.get("hallucination_grounded", "yes")

    if score == "yes":
        logger.info("Decision: Answer is grounded")
        return "useful"
    else:
        logger.info("Decision: Answer has hallucinations, regenerating")
        return "not useful"


def grade_answer_quality_node(state: AgentState) -> dict[str, str]:
    logger.info("--- CHECKING ANSWER QUALITY ---")

    question = state.get("question", "")
    generation = state.get("generation", "")
    attempts = state.get("generation_attempts", 0)

    if attempts >= 3:
        logger.warning(f"Max generation attempts ({attempts}) reached, accepting answer")
        return {"answer_quality": "yes"}

    score = grade_answer_quality(question, generation)

    return {"answer_quality": score}


def grade_generation_quality(state: AgentState) -> str:
    score = state.get("answer_quality", "yes")
    attempts = state.get("generation_attempts", 0)

    if score == "yes":
        logger.info("Decision: Answer is useful")
        return "useful"
    else:
        logger.info(f"Decision: Answer not useful, re-generating (attempt {attempts}/3)")
        return "not useful"
