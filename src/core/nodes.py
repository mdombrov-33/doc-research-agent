from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from src.config import get_settings
from src.core import prompts
from src.core.grading.graders import (
    check_hallucination,
    grade_answer_quality,
    grade_document_relevance,
    rewrite_query,
    route_question,
)
from src.core.state import AgentState
from src.core.tools import get_retriever_tool, get_web_search_tool
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

    if source == "websearch":
        logger.info("Routing to web search")
        return {"web_search": True}
    else:
        logger.info("Routing to vector store")
        return {"web_search": False}


def retrieve_node(state: AgentState) -> dict[str, list[str]]:
    logger.info("--- RETRIEVING FROM VECTOR STORE ---")

    question = state.get("question", "")

    # Step 1: Preprocess query to optimize for semantic search
    preprocessed_query = rewrite_query(question)
    logger.info(f"Preprocessed query: '{question}' -> '{preprocessed_query}'")

    # Step 2: Vector search with preprocessed query
    retriever = get_retriever_tool()
    documents = retriever.invoke(preprocessed_query)

    doc_contents = []
    vector_scores = []

    for doc in documents:
        content = doc.page_content if hasattr(doc, "page_content") else str(doc)
        doc_contents.append(content)

        score = 1.0  # default
        if hasattr(doc, "metadata") and isinstance(doc.metadata, dict):
            score = doc.metadata.get("score", 1.0)
        vector_scores.append(score)

    logger.info(f"Retrieved {len(doc_contents)} documents from vector search")

    # Step 3: Fusion retrieval (combine vector + BM25 scores)
    # Filter out empty documents first
    non_empty_docs = [doc for doc in doc_contents if doc and doc.strip()]

    if len(non_empty_docs) < len(doc_contents):
        logger.warning(f"Filtered out {len(doc_contents) - len(non_empty_docs)} empty documents")
        doc_contents = non_empty_docs

    if doc_contents and len(doc_contents) > 0:
        from src.core.retrieval.fusion_retriever import FusionRetriever

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

    return {"documents": doc_contents}


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

    # First pass: keep relevant docs from vector store
    # Second pass: add relevant web docs to existing relevant docs
    if attempts == 0:
        # First grading (vector docs only)
        filtered_docs = []
        for doc in documents:
            score = grade_document_relevance(question, doc)
            if score == "yes":
                logger.info("Document is relevant")
                filtered_docs.append(doc)
            else:
                logger.info("Document is not relevant")

        web_search_needed = len(filtered_docs) < 2
        logger.info(
            f"Filtered to {len(filtered_docs)} relevant documents. Web search needed: {web_search_needed}"  # noqa: E501
        )

        return {
            "documents": filtered_docs,
            "web_search": web_search_needed,
            "retrieval_attempts": attempts + 1,
        }
    else:
        # Second grading (web docs added to relevant vector docs)
        # Only grade the NEW web docs, keep existing relevant ones
        existing_count = len([d for d in documents if d])  # Rough heuristic
        logger.info(f"Grading {existing_count} total documents (vector + web)")

        filtered_docs = []
        for doc in documents:
            score = grade_document_relevance(question, doc)
            if score == "yes":
                logger.info("Document is relevant")
                filtered_docs.append(doc)
            else:
                logger.info("Document is not relevant")

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


def grade_generation_grounded(state: AgentState) -> str:
    logger.info("--- CHECKING HALLUCINATION ---")

    documents = state.get("documents", [])
    generation = state.get("generation", "")

    score = check_hallucination(documents, generation)

    if score == "yes":
        logger.info("Decision: Answer is grounded")
        return "useful"
    else:
        logger.info("Decision: Answer has hallucinations, regenerating")
        return "not useful"


def grade_generation_quality(state: AgentState) -> str:
    logger.info("--- CHECKING ANSWER QUALITY ---")

    question = state.get("question", "")
    generation = state.get("generation", "")
    attempts = state.get("generation_attempts", 0)

    # Max 3 attempts to generate a useful answer
    if attempts >= 3:
        logger.warning(f"Max generation attempts ({attempts}) reached, accepting answer")
        return "useful"

    score = grade_answer_quality(question, generation)

    if score == "yes":
        logger.info("Decision: Answer is useful")
        return "useful"
    else:
        logger.info(f"Decision: Answer not useful, re-generating (attempt {attempts}/3)")
        return "not useful"
