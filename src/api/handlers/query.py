from typing import Any

from fastapi import HTTPException

from src.api.schemas import QueryRequest
from src.core.agent import get_agent
from src.guardrails.guardrails_wrapper import get_guardrails
from src.utils.logger import logger


async def handle_query(request: QueryRequest) -> dict[str, Any]:
    try:
        logger.info(f"Received query: {request.question}")

        guardrails = get_guardrails()

        rag_result: dict[str, str | list[str]] = {}

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

            return rag_result["generation"]  # type: ignore[return-value]

        guardrails.register_rag_action(run_rag_agent)

        answer = await guardrails.generate_safe(request.question)

        sources_count = len(rag_result.get("documents", [])) if rag_result else 0
        logger.info(f"Query completed. Answer length: {len(answer)}, Sources: {sources_count}")

        return {
            "question": request.question,
            "answer": answer,
            "sources_count": sources_count,
        }

    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")
