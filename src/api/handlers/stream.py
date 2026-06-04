import json
import time
from collections.abc import AsyncGenerator
from typing import Any

from fastapi.responses import StreamingResponse
from langchain_core.runnables import RunnableConfig

from src.api.schemas import QueryRequest
from src.core.monitoring.tracker import MetricsTracker, QueryMetrics
from src.guardrails.guardrails_wrapper import GuardrailsWrapper
from src.utils.logger import logger


async def _token_generator(
    request: QueryRequest,
    guardrails: GuardrailsWrapper,
    agent: Any,
    tracker: MetricsTracker,
) -> AsyncGenerator[str, None]:
    _t = time.monotonic()
    refusal = await guardrails.check_input(request.question)
    logger.info(
        "node_complete",
        node="guardrails_input",
        duration_ms=round((time.monotonic() - _t) * 1000, 1),
    )
    if refusal:
        yield f"data: {json.dumps({'token': refusal, 'done': True})}\n\n"
        return

    inputs = {
        "question": request.question,
        "web_search": False,
        "raw_documents": [],
        "documents": [],
        "model": request.model,
        "top_k": request.top_k,
    }
    config: RunnableConfig = {"configurable": {"thread_id": request.session_id}}

    accumulated: list[str] = []
    sources_count = 0
    sources_meta: list[dict] = []
    web_search_triggered = False
    docs_retrieved_total = 0
    start_ms = time.monotonic() * 1000

    try:
        async for event in agent.astream_events(inputs, config=config, version="v2"):
            kind = event["event"]

            if (
                kind == "on_chat_model_stream"
                and event.get("metadata", {}).get("langgraph_node") == "generate"
            ):
                token = event["data"]["chunk"].content
                if token:
                    accumulated.append(token)
                    yield f"data: {json.dumps({'token': token})}\n\n"

            elif kind == "on_chain_end" and event.get("name") == "LangGraph":
                output = event.get("data", {}).get("output", {})
                graded_docs = output.get("documents", [])
                sources_count = len(graded_docs)
                sources_meta = [
                    {
                        "filename": d.get("filename", "unknown"),
                        "chunk_index": d.get("chunk_index", 0),
                        "chunk_length": d.get("chunk_length", 0),
                        "source": d.get("source", "vectorstore"),
                    }
                    for d in graded_docs
                ]
                web_search_triggered = output.get("web_search_done", False) or output.get(
                    "web_search", False
                )
                docs_retrieved_total = output.get("docs_retrieved_total", sources_count)

    except Exception as e:
        logger.error("stream_failed", error=str(e), exc_info=True)
        yield f"data: {json.dumps({'error': str(e), 'done': True})}\n\n"
        return

    full_response = "".join(accumulated)
    _t = time.monotonic()
    correction = await guardrails.check_output(full_response)
    logger.info(
        "node_complete",
        node="guardrails_output",
        duration_ms=round((time.monotonic() - _t) * 1000, 1),
    )

    latency_ms = time.monotonic() * 1000 - start_ms
    tracker.record(
        QueryMetrics(
            question=request.question,
            retrieval_precision=sources_count / docs_retrieved_total
            if docs_retrieved_total
            else 0.0,
            docs_retrieved=docs_retrieved_total,
            docs_relevant=sources_count,
            web_search_triggered=web_search_triggered,
            latency_ms=latency_ms,
        )
    )

    if correction:
        yield f"data: {json.dumps({'token': correction, 'done': True, 'correction': True})}\n\n"
    else:
        yield f"data: {json.dumps({'done': True, 'sources_count': sources_count, 'sources': sources_meta, 'session_id': request.session_id})}\n\n"  # noqa: E501


async def handle_stream(
    request: QueryRequest,
    guardrails: GuardrailsWrapper,
    agent: Any,
    tracker: MetricsTracker,
) -> StreamingResponse:
    return StreamingResponse(
        _token_generator(request, guardrails, agent, tracker),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
