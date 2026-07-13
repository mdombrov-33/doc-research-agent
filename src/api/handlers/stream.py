import json
import time
from collections.abc import AsyncGenerator
from typing import Any

from fastapi import Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from src.api.schemas import QueryRequest
from src.core import guardrails
from src.core.agent.outcomes import (
    FinalOutcome,
    FinalStopReason,
    normalize_outcome,
    normalize_stop_reason,
)
from src.core.citations import CitationMarkerRedactor, citations_referenced_by_answer
from src.core.monitoring.tracker import MetricsTracker, QueryMetrics
from src.utils.logger import logger


def _now_ms() -> float:
    return time.monotonic() * 1000


def _turn_sources(messages: list[Any]) -> tuple[list[dict], int, bool]:
    last_human_idx = max(
        (i for i, message in enumerate(messages) if isinstance(message, HumanMessage)),
        default=-1,
    )
    artifacts: list[dict] = []
    web_search_triggered = False
    turn_messages = messages[last_human_idx + 1 :]
    for message in turn_messages:
        if not isinstance(message, ToolMessage) or not message.artifact:
            continue
        artifacts.extend(artifact for artifact in message.artifact if isinstance(artifact, dict))
        web_search_triggered |= message.name == "web_search"

    citations = citations_referenced_by_answer(artifacts, _final_answer(turn_messages))
    sources = [citation.model_dump(exclude_none=True) for citation in citations]
    return sources, len(artifacts), web_search_triggered


def _final_answer(messages: list[Any]) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage) and not message.tool_calls:
            return message.content if isinstance(message.content, str) else ""
    return ""


async def _token_generator(
    request: QueryRequest,
    agent: Any,
    tracker: MetricsTracker,
    client_request: Request,
) -> AsyncGenerator[str, None]:
    if await client_request.is_disconnected():
        logger.info("stream_cancelled", stage="before_guardrails")
        return

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

    inputs = {"messages": [HumanMessage(content=request.question)]}
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
    citation_redactor = CitationMarkerRedactor()
    sources_count = 0
    sources_meta: list[dict] = []
    web_search_triggered = False
    sources_retrieved_total = 0
    outcome: FinalOutcome = "abstained"
    stop_reason: FinalStopReason = "unknown"
    start_ms = _now_ms()
    time_to_first_token_ms: float | None = None

    event_stream = agent.astream_events(inputs, config=config, version="v2")
    try:
        while True:
            if await client_request.is_disconnected():
                logger.info("stream_cancelled", stage="before_graph_event")
                return
            try:
                event = await anext(event_stream)
            except StopAsyncIteration:
                break

            kind = event["event"]

            # The query node only calls retrieval. The dedicated answer node is the sole
            # source of user-visible model text.
            if (
                kind == "on_chat_model_stream"
                and event.get("metadata", {}).get("langgraph_node") == "answer"
            ):
                token = event["data"]["chunk"].content
                if token:
                    visible_token = citation_redactor.push(token)
                    if visible_token:
                        if time_to_first_token_ms is None:
                            time_to_first_token_ms = _now_ms() - start_ms
                        accumulated.append(visible_token)
                        yield f"data: {json.dumps({'token': visible_token})}\n\n"

            elif kind == "on_chain_end" and event.get("name") == "LangGraph":
                output = event.get("data", {}).get("output", {})
                sources_meta, sources_retrieved_total, web_search_triggered = _turn_sources(
                    output.get("messages", [])
                )
                sources_count = len(sources_meta)
                outcome = normalize_outcome(output.get("outcome"))
                stop_reason = normalize_stop_reason(output.get("stop_reason"))

    except Exception as error:
        logger.error("stream_failed", failure_type=type(error).__name__)
        error_event = {
            "error": "Unable to complete the request. Please try again.",
            "done": True,
        }
        yield f"data: {json.dumps(error_event)}\n\n"
        return
    finally:
        aclose = getattr(event_stream, "aclose", None)
        if aclose is not None:
            await aclose()

    latency_ms = _now_ms() - start_ms
    visible_tail = citation_redactor.flush()
    if visible_tail:
        if time_to_first_token_ms is None:
            time_to_first_token_ms = _now_ms() - start_ms
        accumulated.append(visible_tail)
        yield f"data: {json.dumps({'token': visible_tail})}\n\n"
    tracker.record(
        QueryMetrics(
            sources_retrieved=sources_retrieved_total,
            web_search_triggered=web_search_triggered,
            latency_ms=latency_ms,
            time_to_first_token_ms=time_to_first_token_ms,
            outcome=outcome,
        )
    )
    logger.info(
        "query_completed",
        outcome=outcome,
        stop_reason=stop_reason,
        latency_ms=round(latency_ms, 1),
        time_to_first_token_ms=(
            round(time_to_first_token_ms, 1) if time_to_first_token_ms is not None else None
        ),
        sources_retrieved=sources_retrieved_total,
        sources_cited=sources_count,
        web_search_triggered=web_search_triggered,
    )

    yield f"data: {json.dumps({'done': True, 'sources_count': sources_count, 'sources': sources_meta, 'session_id': request.session_id, 'outcome': outcome, 'stop_reason': stop_reason})}\n\n"  # noqa: E501


async def handle_stream(
    request: QueryRequest,
    agent: Any,
    tracker: MetricsTracker,
    client_request: Request,
) -> StreamingResponse:
    return StreamingResponse(
        _token_generator(request, agent, tracker, client_request),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
