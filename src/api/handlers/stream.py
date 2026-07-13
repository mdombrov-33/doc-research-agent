import json
import time
from collections.abc import AsyncGenerator
from typing import Any

from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from src.api.schemas import QueryRequest
from src.core import guardrails
from src.core.citations import citations_from_artifacts
from src.core.monitoring.tracker import MetricsTracker, QueryMetrics
from src.utils.logger import logger


def _turn_sources(messages: list[Any]) -> tuple[list[dict], int, bool]:
    last_human_idx = max(
        (i for i, message in enumerate(messages) if isinstance(message, HumanMessage)),
        default=-1,
    )
    artifacts: list[dict] = []
    web_search_triggered = False
    for message in messages[last_human_idx + 1 :]:
        if not isinstance(message, ToolMessage) or not message.artifact:
            continue
        artifacts.extend(artifact for artifact in message.artifact if isinstance(artifact, dict))
        web_search_triggered |= message.name == "web_search"

    citations = citations_from_artifacts(artifacts)
    sources = [citation.model_dump(exclude_none=True) for citation in citations]
    return sources, len(artifacts), web_search_triggered


async def _token_generator(
    request: QueryRequest,
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
        "messages": [HumanMessage(content=request.question)],
        "tool_call_count": None,
    }
    # thread_id keys the persisted conversation; model/top_k are per-request knobs the agent
    # node and retrieve tool read back from config.
    config: RunnableConfig = {
        "configurable": {
            "thread_id": request.session_id,
            "model": request.model,
            "top_k": request.top_k,
        }
    }

    accumulated: list[str] = []
    sources_count = 0
    sources_meta: list[dict] = []
    web_search_triggered = False
    sources_retrieved_total = 0
    start_ms = time.monotonic() * 1000

    try:
        async for event in agent.astream_events(inputs, config=config, version="v2"):
            kind = event["event"]

            # The agent node both decides tools and writes the final answer. Tool-deciding
            # turns carry no content, so only the answer turn yields tokens here.
            if (
                kind == "on_chat_model_stream"
                and event.get("metadata", {}).get("langgraph_node") == "agent"
            ):
                token = event["data"]["chunk"].content
                if token:
                    accumulated.append(token)
                    yield f"data: {json.dumps({'token': token})}\n\n"

            elif kind == "on_chain_end" and event.get("name") == "LangGraph":
                output = event.get("data", {}).get("output", {})
                sources_meta, sources_retrieved_total, web_search_triggered = _turn_sources(
                    output.get("messages", [])
                )
                sources_count = len(sources_meta)

    except Exception as e:
        logger.error("stream_failed", error=str(e), exc_info=True)
        error_event = {
            "error": "Unable to complete the request. Please try again.",
            "done": True,
        }
        yield f"data: {json.dumps(error_event)}\n\n"
        return

    latency_ms = time.monotonic() * 1000 - start_ms
    tracker.record(
        QueryMetrics(
            question=request.question,
            sources_retrieved=sources_retrieved_total,
            web_search_triggered=web_search_triggered,
            latency_ms=latency_ms,
        )
    )

    yield f"data: {json.dumps({'done': True, 'sources_count': sources_count, 'sources': sources_meta, 'session_id': request.session_id})}\n\n"  # noqa: E501


async def handle_stream(
    request: QueryRequest,
    agent: Any,
    tracker: MetricsTracker,
) -> StreamingResponse:
    return StreamingResponse(
        _token_generator(request, agent, tracker),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
