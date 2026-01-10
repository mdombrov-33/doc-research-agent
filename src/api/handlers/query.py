import time
from typing import Any

from fastapi import HTTPException

from src.api.schemas import QueryRequest
from src.core.agent import get_agent
from src.core.evaluation.metrics import QueryEvaluation, get_evaluation_tracker
from src.guardrails.guardrails_wrapper import get_guardrails
from src.utils.logger import logger


async def handle_query(request: QueryRequest) -> dict[str, Any]:
    start_time = time.time()

    try:
        logger.info(f"Received query: {request.question}")

        guardrails = get_guardrails()

        rag_result: dict[str, str | list[str] | int | bool] = {}

        async def run_rag_agent(question: str) -> str:
            agent = get_agent()
            inputs: dict[str, str | bool | list[str] | int] = {
                "question": question,
                "generation": "",
                "web_search": False,
                "explicit_web_search": False,
                "documents": [],
                "retrieval_attempts": 0,
                "generation_attempts": 0,
            }
            result = agent.invoke(inputs)  # type: ignore[arg-type]

            rag_result["generation"] = result.get("generation", "No answer generated")
            rag_result["documents"] = result.get("documents", [])
            rag_result["web_search"] = result.get("web_search", False)
            rag_result["generation_attempts"] = result.get("generation_attempts", 1)
            rag_result["hallucination_grounded"] = result.get("hallucination_grounded", "yes")
            rag_result["answer_quality"] = result.get("answer_quality", "yes")
            rag_result["docs_retrieved_total"] = result.get("docs_retrieved_total", 0)

            return rag_result["generation"]  # type: ignore[return-value]

        guardrails.register_rag_action(run_rag_agent)

        answer = await guardrails.generate_safe(request.question)

        latency_ms = (time.time() - start_time) * 1000

        documents_list = rag_result.get("documents", [])
        sources_count = len(documents_list) if isinstance(documents_list, list) else 0

        docs_retrieved_raw = rag_result.get("docs_retrieved_total", sources_count)
        docs_retrieved = (
            int(docs_retrieved_raw) if isinstance(docs_retrieved_raw, int) else sources_count
        )  # noqa: E501

        generation_attempts_raw = rag_result.get("generation_attempts", 1)
        generation_attempts = (
            int(generation_attempts_raw) if isinstance(generation_attempts_raw, int) else 1
        )  # noqa: E501

        evaluation = QueryEvaluation(
            question=request.question,
            retrieval_precision=(sources_count / docs_retrieved if docs_retrieved > 0 else 0.0),
            docs_retrieved=docs_retrieved,
            docs_relevant=sources_count,
            hallucination_check=str(rag_result.get("hallucination_grounded", "yes")),
            quality_check=str(rag_result.get("answer_quality", "yes")),
            web_search_triggered=bool(rag_result.get("web_search", False)),
            generation_attempts=generation_attempts,
            latency_ms=latency_ms,
        )

        tracker = get_evaluation_tracker()
        tracker.record(evaluation)

        logger.info(f"Evaluation: {evaluation.to_dict()}")
        logger.info(f"Query completed. Answer length: {len(answer)}, Sources: {sources_count}")

        return {
            "question": request.question,
            "answer": answer,
            "sources_count": sources_count,
        }

    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")
