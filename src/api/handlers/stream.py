import asyncio
import json
import time
from collections.abc import AsyncGenerator
from typing import Any

from fastapi import Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from src.api.schemas import QueryRequest
from src.config import get_settings
from src.core import answer_cache, conversational, guardrails
from src.core.agent.outcomes import (
    FinalOutcome,
    FinalStopReason,
    normalize_outcome,
    normalize_stop_reason,
)
from src.core.agent.tools import artifact_documents, artifact_retrieval_metrics
from src.core.citations import (
    CitationMarkerRedactor,
    citations_referenced_by_answer,
    strip_source_ids,
)
from src.core.monitoring.tracker import MetricsTracker, QueryMetrics
from src.utils.logger import logger


def _now_ms() -> float:
    return time.monotonic() * 1000


async def _is_first_turn(agent: Any, config: RunnableConfig) -> bool:
    """A turn is cacheable only when the session's checkpoint holds no prior messages.

    A failed state probe degrades to "not first turn" so the request still runs the graph.
    """
    try:
        state = await agent.aget_state(config)
    except Exception as error:
        logger.warning("answer_cache_state_probe_failed", failure_type=type(error).__name__)
        return False
    return not state.values.get("messages")


async def _consult_cache(question: str, model: str) -> dict | None:
    try:
        return await asyncio.to_thread(answer_cache.lookup, question, model)
    except Exception as error:
        logger.warning("answer_cache_lookup_failed", failure_type=type(error).__name__)
        return None


async def _populate_cache(question: str, answer: str, sources: list[dict], model: str) -> None:
    try:
        await asyncio.to_thread(answer_cache.store, question, answer, sources, model)
    except Exception as error:
        logger.warning("answer_cache_store_failed", failure_type=type(error).__name__)


async def _write_cache_turn(agent: Any, config: RunnableConfig, question: str, answer: str) -> None:
    """Record a cache-served turn in the checkpoint so follow-ups still see the history."""
    try:
        await agent.aupdate_state(
            config, {"messages": [HumanMessage(content=question), AIMessage(content=answer)]}
        )
    except Exception as error:
        logger.warning("answer_cache_checkpoint_write_failed", failure_type=type(error).__name__)


def _message_usage(message: Any) -> tuple[int, int, float | None]:
    """Tokens and provider cost from one LLM call's end event.

    Cost comes from OpenRouter's usage accounting, but LangChain drops it on streamed
    calls, and under ``astream_events`` every call is streamed — so cost is None here
    until that changes. Tokens always survive.
    """
    metadata = getattr(message, "usage_metadata", None) or {}
    token_usage = getattr(message, "response_metadata", {}).get("token_usage") or {}
    return (
        metadata.get("input_tokens") or 0,
        metadata.get("output_tokens") or 0,
        token_usage.get("cost"),
    )


def _turn_messages(messages: list[Any]) -> list[Any]:
    last_human_idx = max(
        (i for i, message in enumerate(messages) if isinstance(message, HumanMessage)),
        default=-1,
    )
    return messages[last_human_idx + 1 :]


def _turn_sources(messages: list[Any]) -> tuple[list[dict], int, bool, dict[str, int | str]]:
    artifacts: list[dict] = []
    web_search_triggered = False
    retrieval_metrics: dict[str, int | str] = {}
    turn_messages = _turn_messages(messages)
    for message in turn_messages:
        if not isinstance(message, ToolMessage) or not message.artifact:
            continue
        artifacts.extend(artifact_documents(message.artifact))
        web_search_triggered |= message.name == "web_search"
        if message.name == "retrieve_documents":
            retrieval_metrics.update(artifact_retrieval_metrics(message.artifact))

    citations = citations_referenced_by_answer(artifacts, _final_answer(turn_messages))
    sources = [citation.model_dump(exclude_none=True) for citation in citations]
    return sources, len(artifacts), web_search_triggered, retrieval_metrics


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
    request_start_ms: float | None = None,
) -> AsyncGenerator[str, None]:
    start_ms = request_start_ms if request_start_ms is not None else _now_ms()
    settings = get_settings()
    model = request.model or settings.LLM_MODEL
    time_to_first_token_ms: float | None = None

    if await client_request.is_disconnected():
        logger.info("stream_cancelled", stage="before_guardrails")
        return

    timeout_seconds = settings.QUERY_TIMEOUT_SECONDS
    elapsed_seconds = max(0.0, (_now_ms() - start_ms) / 1000)
    deadline_at = asyncio.get_running_loop().time() + max(0.0, timeout_seconds - elapsed_seconds)
    guardrail_started_at = _now_ms()
    try:
        async with asyncio.timeout_at(deadline_at):
            refusal = await guardrails.check_input(request.question)
    except TimeoutError:
        completion_ms = _now_ms() - start_ms
        logger.warning("stream_timed_out", timeout_seconds=timeout_seconds)
        tracker.record(
            QueryMetrics(
                sources_retrieved=0,
                web_search_triggered=False,
                latency_ms=completion_ms,
                time_to_first_token_ms=None,
                outcome="abstained",
                model=model,
                path="timeout",
                planner_model=settings.PLANNER_MODEL,
                assessor_model=settings.ASSESSOR_MODEL,
            )
        )
        logger.info(
            "query_completed",
            path="timeout",
            outcome="abstained",
            latency_ms=round(completion_ms, 1),
            time_to_first_token_ms=None,
            model=model,
            planner_model=settings.PLANNER_MODEL,
            assessor_model=settings.ASSESSOR_MODEL,
        )
        error_event = {
            "error": "The request timed out. Please try again.",
            "done": True,
        }
        yield f"data: {json.dumps(error_event)}\n\n"
        return
    guardrail_duration_ms = _now_ms() - guardrail_started_at
    logger.info(
        "node_complete",
        node="guardrails_input",
        duration_ms=round(guardrail_duration_ms, 1),
    )
    if refusal:
        time_to_first_token_ms = _now_ms() - start_ms
        completion_ms = time_to_first_token_ms
        tracker.record(
            QueryMetrics(
                sources_retrieved=0,
                web_search_triggered=False,
                latency_ms=completion_ms,
                time_to_first_token_ms=time_to_first_token_ms,
                outcome="abstained",
                model=model,
                path="refusal",
                planner_model=settings.PLANNER_MODEL,
                assessor_model=settings.ASSESSOR_MODEL,
            )
        )
        logger.info(
            "query_completed",
            path="refusal",
            outcome="abstained",
            latency_ms=round(completion_ms, 1),
            time_to_first_token_ms=round(time_to_first_token_ms, 1),
            guardrail_duration_ms=round(guardrail_duration_ms, 1),
            model=model,
            planner_model=settings.PLANNER_MODEL,
            assessor_model=settings.ASSESSOR_MODEL,
        )
        yield f"data: {json.dumps({'token': refusal, 'done': True})}\n\n"
        return

    # A greeting or "what can you do?" gets a fixed reply without touching the graph, so it
    # never lands as a hard abstention. Conservative matching means content questions fall
    # through. No checkpoint write — losing "hi" from the history is harmless.
    conversational_reply = conversational.match(request.question)
    if conversational_reply is not None:
        time_to_first_token_ms = _now_ms() - start_ms
        yield f"data: {json.dumps({'token': conversational_reply})}\n\n"
        completion_ms = _now_ms() - start_ms
        tracker.record(
            QueryMetrics(
                sources_retrieved=0,
                web_search_triggered=False,
                latency_ms=completion_ms,
                time_to_first_token_ms=time_to_first_token_ms,
                outcome="conversational",
                model=model,
                input_tokens=0,
                output_tokens=0,
                reported_cost=0.0,
                path="conversational",
                planner_model=settings.PLANNER_MODEL,
                assessor_model=settings.ASSESSOR_MODEL,
            )
        )
        logger.info(
            "query_completed",
            path="conversational",
            outcome="conversational",
            latency_ms=round(completion_ms, 1),
            time_to_first_token_ms=round(time_to_first_token_ms, 1),
            guardrail_duration_ms=round(guardrail_duration_ms, 1),
            model=model,
            planner_model=settings.PLANNER_MODEL,
            assessor_model=settings.ASSESSOR_MODEL,
        )
        yield f"data: {json.dumps({'done': True, 'sources_count': 0, 'sources': [], 'session_id': request.session_id, 'outcome': 'conversational', 'stop_reason': 'retrieval_not_requested'})}\n\n"  # noqa: E501
        return

    inputs = {"messages": [HumanMessage(content=request.question)]}
    # thread_id keys persisted conversation; model is invocation-scoped answer configuration.
    config: RunnableConfig = {
        "configurable": {
            "thread_id": request.session_id,
            "model": request.model,
        }
    }

    # Consult the cache only on a session's first turn; a follow-up ("expand on that") depends on
    # history the cache has no way to reproduce. first_turn is reused to gate populate below.
    preflight_started_at = _now_ms()
    first_turn = settings.ANSWER_CACHE_ENABLED and await _is_first_turn(agent, config)
    if first_turn:
        cached = await _consult_cache(request.question, model)
        if cached is not None:
            answer = cached["answer"]
            cached_sources = cached["sources"]
            preflight_duration_ms = _now_ms() - preflight_started_at
            time_to_first_token_ms = _now_ms() - start_ms
            yield f"data: {json.dumps({'token': answer})}\n\n"
            await _write_cache_turn(agent, config, request.question, answer)
            completion_ms = _now_ms() - start_ms
            tracker.record(
                QueryMetrics(
                    sources_retrieved=0,
                    web_search_triggered=False,
                    latency_ms=completion_ms,
                    time_to_first_token_ms=time_to_first_token_ms,
                    outcome="document_answer",
                    model=model,
                    input_tokens=0,
                    output_tokens=0,
                    reported_cost=0.0,
                    cache_hit=True,
                    path="cache",
                    planner_model=settings.PLANNER_MODEL,
                    assessor_model=settings.ASSESSOR_MODEL,
                )
            )
            logger.info(
                "query_completed",
                path="cache",
                outcome="document_answer",
                cache_hit=True,
                latency_ms=round(completion_ms, 1),
                time_to_first_token_ms=round(time_to_first_token_ms, 1),
                guardrail_duration_ms=round(guardrail_duration_ms, 1),
                preflight_duration_ms=round(preflight_duration_ms, 1),
                model=model,
                planner_model=settings.PLANNER_MODEL,
                assessor_model=settings.ASSESSOR_MODEL,
            )
            yield f"data: {json.dumps({'done': True, 'sources_count': len(cached_sources), 'sources': cached_sources, 'session_id': request.session_id, 'outcome': 'document_answer', 'stop_reason': 'document_evidence_sufficient'})}\n\n"  # noqa: E501
            return
    preflight_duration_ms = _now_ms() - preflight_started_at

    accumulated: list[str] = []
    citation_redactor = CitationMarkerRedactor()
    sources_count = 0
    sources_meta: list[dict] = []
    web_search_triggered = False
    sources_retrieved_total = 0
    outcome: FinalOutcome = "abstained"
    stop_reason: FinalStopReason = "unknown"
    answer_started_at_ms: float | None = None
    answer_provider_ttft_ms: float | None = None
    answer_completion_ms: float | None = None
    final_answer = ""
    input_tokens = 0
    output_tokens = 0
    reported_cost: float | None = None
    retrieval_metrics: dict[str, int | str] = {}

    event_stream = agent.astream_events(inputs, config=config, version="v2")
    try:
        while True:
            if await client_request.is_disconnected():
                logger.info("stream_cancelled", stage="before_graph_event")
                return
            try:
                async with asyncio.timeout_at(deadline_at):
                    event = await anext(event_stream)
            except StopAsyncIteration:
                break

            kind = event["event"]

            if (
                kind == "on_chat_model_start"
                and event.get("metadata", {}).get("langgraph_node") == "answer"
            ):
                answer_started_at_ms = _now_ms()

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
                            first_visible_at_ms = _now_ms()
                            time_to_first_token_ms = first_visible_at_ms - start_ms
                            if answer_started_at_ms is not None:
                                answer_provider_ttft_ms = (
                                    first_visible_at_ms - answer_started_at_ms
                                )
                        accumulated.append(visible_token)
                        yield f"data: {json.dumps({'token': visible_token})}\n\n"

            elif kind == "on_chat_model_end":
                if (
                    event.get("metadata", {}).get("langgraph_node") == "answer"
                    and answer_started_at_ms is not None
                ):
                    answer_completion_ms = _now_ms() - answer_started_at_ms
                turn_input, turn_output, turn_cost = _message_usage(
                    event.get("data", {}).get("output")
                )
                input_tokens += turn_input
                output_tokens += turn_output
                if turn_cost is not None:
                    reported_cost = (reported_cost or 0.0) + turn_cost

            elif kind == "on_chain_end" and event.get("name") == "LangGraph":
                output = event.get("data", {}).get("output", {})
                (
                    sources_meta,
                    sources_retrieved_total,
                    web_search_triggered,
                    retrieval_metrics,
                ) = _turn_sources(output.get("messages", []))
                final_answer = _final_answer(_turn_messages(output.get("messages", [])))
                sources_count = len(sources_meta)
                outcome = normalize_outcome(output.get("outcome"))
                stop_reason = normalize_stop_reason(output.get("stop_reason"))

    except TimeoutError:
        completion_ms = _now_ms() - start_ms
        logger.warning("stream_timed_out", timeout_seconds=timeout_seconds)
        tracker.record(
            QueryMetrics(
                sources_retrieved=sources_retrieved_total,
                web_search_triggered=web_search_triggered,
                latency_ms=completion_ms,
                time_to_first_token_ms=time_to_first_token_ms,
                outcome="abstained",
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                reported_cost=reported_cost,
                path="timeout",
                planner_model=settings.PLANNER_MODEL,
                assessor_model=settings.ASSESSOR_MODEL,
                **_retrieval_metric_fields(retrieval_metrics),
            )
        )
        logger.info(
            "query_completed",
            path="timeout",
            outcome="abstained",
            latency_ms=round(completion_ms, 1),
            time_to_first_token_ms=(
                round(time_to_first_token_ms, 1) if time_to_first_token_ms is not None else None
            ),
            model=model,
            planner_model=settings.PLANNER_MODEL,
            assessor_model=settings.ASSESSOR_MODEL,
        )
        error_event = {
            "error": "The request timed out. Please try again.",
            "done": True,
        }
        yield f"data: {json.dumps(error_event)}\n\n"
        return
    except Exception as error:
        completion_ms = _now_ms() - start_ms
        logger.error("stream_failed", failure_type=type(error).__name__)
        tracker.record(
            QueryMetrics(
                sources_retrieved=sources_retrieved_total,
                web_search_triggered=web_search_triggered,
                latency_ms=completion_ms,
                time_to_first_token_ms=time_to_first_token_ms,
                outcome="abstained",
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                reported_cost=reported_cost,
                path="error",
                planner_model=settings.PLANNER_MODEL,
                assessor_model=settings.ASSESSOR_MODEL,
                **_retrieval_metric_fields(retrieval_metrics),
            )
        )
        logger.info(
            "query_completed",
            path="error",
            outcome="abstained",
            latency_ms=round(completion_ms, 1),
            time_to_first_token_ms=(
                round(time_to_first_token_ms, 1) if time_to_first_token_ms is not None else None
            ),
            model=model,
            planner_model=settings.PLANNER_MODEL,
            assessor_model=settings.ASSESSOR_MODEL,
        )
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

    visible_tail = citation_redactor.flush()
    if visible_tail:
        if time_to_first_token_ms is None:
            time_to_first_token_ms = _now_ms() - start_ms
        accumulated.append(visible_tail)
        yield f"data: {json.dumps({'token': visible_tail})}\n\n"

    # Terminal nodes that answer without calling a model (abstain) emit no stream events, so
    # nothing above ever reaches the client. Send their text instead of an empty response.
    if not accumulated:
        unstreamed = strip_source_ids(final_answer).strip()
        if unstreamed:
            time_to_first_token_ms = _now_ms() - start_ms
            accumulated.append(unstreamed)
            yield f"data: {json.dumps({'token': unstreamed})}\n\n"
    # Cache only first-turn document answers; web answers and abstentions can change by tomorrow.
    if first_turn and outcome == "document_answer" and accumulated:
        await _populate_cache(request.question, "".join(accumulated), sources_meta, model)
    completion_ms = _now_ms() - start_ms
    path = (
        "web"
        if outcome == "web_answer"
        else "document"
        if outcome == "document_answer"
        else "abstention"
    )
    tracker.record(
        QueryMetrics(
            sources_retrieved=sources_retrieved_total,
            web_search_triggered=web_search_triggered,
            latency_ms=completion_ms,
            time_to_first_token_ms=time_to_first_token_ms,
            outcome=outcome,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            reported_cost=reported_cost,
            path=path,
            planner_model=settings.PLANNER_MODEL,
            assessor_model=settings.ASSESSOR_MODEL,
            **_retrieval_metric_fields(retrieval_metrics),
        )
    )
    logger.info(
        "query_completed",
        path=path,
        outcome=outcome,
        stop_reason=stop_reason,
        latency_ms=round(completion_ms, 1),
        time_to_first_token_ms=(
            round(time_to_first_token_ms, 1) if time_to_first_token_ms is not None else None
        ),
        guardrail_duration_ms=round(guardrail_duration_ms, 1),
        preflight_duration_ms=round(preflight_duration_ms, 1),
        answer_provider_ttft_ms=(
            round(answer_provider_ttft_ms, 1) if answer_provider_ttft_ms is not None else None
        ),
        answer_completion_ms=(
            round(answer_completion_ms, 1) if answer_completion_ms is not None else None
        ),
        sources_retrieved=sources_retrieved_total,
        sources_cited=sources_count,
        web_search_triggered=web_search_triggered,
        model=model,
        planner_model=settings.PLANNER_MODEL,
        assessor_model=settings.ASSESSOR_MODEL,
        **retrieval_metrics,
    )

    yield f"data: {json.dumps({'done': True, 'sources_count': sources_count, 'sources': sources_meta, 'session_id': request.session_id, 'outcome': outcome, 'stop_reason': stop_reason})}\n\n"  # noqa: E501


def _retrieval_metric_fields(metrics: dict[str, int | str]) -> dict[str, Any]:
    return {
        "query_shape": metrics.get("query_shape"),
        "candidate_count": metrics.get("candidates", 0),
        "evidence_count": metrics.get("evidence", 0),
        "reranker_calls": metrics.get("reranker_calls", 0),
    }


async def handle_stream(
    request: QueryRequest,
    agent: Any,
    tracker: MetricsTracker,
    client_request: Request,
) -> StreamingResponse:
    request_start_ms = _now_ms()
    return StreamingResponse(
        _token_generator(request, agent, tracker, client_request, request_start_ms),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
