import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.api.handlers.stream import _token_generator, _turn_sources
from src.api.schemas import QueryRequest
from src.core.monitoring.tracker import MetricsTracker


class ConnectedRequest:
    async def is_disconnected(self) -> bool:
        return False


@pytest.fixture(autouse=True)
def _no_cache(monkeypatch):
    """Treat every turn as a follow-up so the cache path stays out of non-cache tests.

    The dedicated cache tests below re-patch _is_first_turn to opt back in.
    """

    async def not_first_turn(*_args, **_kwargs) -> bool:
        return False

    monkeypatch.setattr("src.api.handlers.stream._is_first_turn", not_first_turn)


def test_turn_sources_preserves_document_and_web_evidence():
    messages = [
        HumanMessage(content="What does the document say?"),
        ToolMessage(
            content="document result",
            tool_call_id="retrieve-1",
            name="retrieve_documents",
            artifact=[
                {
                    "content": "The document says the rollout is complete.",
                    "document_id": "doc-report",
                    "chunk_id": "doc-report:2",
                    "filename": "report.pdf",
                    "page": 3,
                    "source": "document",
                }
            ],
        ),
        AIMessage(
            content=(
                "The announcement confirms it [web:https://example.com/announcement]. "
                "The rollout is complete [document:doc-report:2]."
            )
        ),
        ToolMessage(
            content="web result",
            tool_call_id="web-1",
            name="web_search",
            artifact=[
                {
                    "content": "The official announcement confirms it.",
                    "title": "Official announcement",
                    "url": "https://example.com/announcement",
                    "source": "web",
                }
            ],
        ),
    ]

    sources, retrieved_total, web_search_triggered, retrieval_metrics = _turn_sources(messages)

    assert sources == [
        {
            "source_id": "web:https://example.com/announcement",
            "source_type": "web",
            "title": "Official announcement",
            "url": "https://example.com/announcement",
            "excerpt": "The official announcement confirms it.",
        },
        {
            "source_id": "document:doc-report:2",
            "source_type": "document",
            "title": "report.pdf",
            "document_id": "doc-report",
            "chunk_id": "doc-report:2",
            "page": 3,
            "excerpt": "The document says the rollout is complete.",
        },
    ]
    assert retrieved_total == 2
    assert web_search_triggered is True
    assert retrieval_metrics == {}


def test_turn_sources_deduplicates_repeated_artifacts():
    artifact = {
        "content": "Repeated document evidence.",
        "document_id": "doc-report",
        "chunk_id": "doc-report:2",
        "filename": "report.pdf",
        "source": "document",
    }
    messages = [
        HumanMessage(content="What does the document say?"),
        ToolMessage(
            content="document result",
            tool_call_id="retrieve-1",
            name="retrieve_documents",
            artifact=[artifact],
        ),
        AIMessage(
            content="The evidence is repeated [document:doc-report:2] [document:doc-report:2]."
        ),
        ToolMessage(
            content="same document result",
            tool_call_id="retrieve-2",
            name="retrieve_documents",
            artifact=[artifact],
        ),
    ]

    sources, retrieved_total, web_search_triggered, retrieval_metrics = _turn_sources(messages)

    assert sources == [
        {
            "source_id": "document:doc-report:2",
            "source_type": "document",
            "title": "report.pdf",
            "document_id": "doc-report",
            "chunk_id": "doc-report:2",
            "excerpt": "Repeated document evidence.",
        }
    ]
    assert retrieved_total == 2
    assert web_search_triggered is False
    assert retrieval_metrics == {}


@pytest.mark.asyncio
async def test_stream_hides_internal_source_ids_from_answer_tokens(monkeypatch):
    async def no_refusal(_: str) -> None:
        return None

    class Agent:
        async def astream_events(self, *_args, **_kwargs):
            for content in ["The rollout is complete [doc", "ument:doc-report:2]."]:
                yield {
                    "event": "on_chat_model_stream",
                    "metadata": {"langgraph_node": "answer"},
                    "data": {"chunk": SimpleNamespace(content=content)},
                }

    monkeypatch.setattr("src.api.handlers.stream.guardrails.check_input", no_refusal)
    events = [
        event
        async for event in _token_generator(
            QueryRequest(question="What is the rollout status?"),
            Agent(),
            MetricsTracker(),
            ConnectedRequest(),
        )
    ]
    visible_answer = "".join(
        json.loads(event.removeprefix("data: ").strip())["token"]
        for event in events
        if "token" in json.loads(event.removeprefix("data: ").strip())
    )

    assert visible_answer == "The rollout is complete."


@pytest.mark.asyncio
async def test_stream_sends_answer_text_from_nodes_that_never_call_a_model(monkeypatch):
    """Abstain builds its message in Python, so no token would otherwise reach the client."""

    async def no_refusal(_: str) -> None:
        return None

    class Agent:
        async def astream_events(self, *_args, **_kwargs):
            yield {
                "event": "on_chain_end",
                "name": "LangGraph",
                "data": {
                    "output": {
                        "messages": [
                            HumanMessage(content="What is the status?"),
                            AIMessage(content="I couldn't find enough reliable evidence."),
                        ],
                        "outcome": "abstained",
                        "stop_reason": "insufficient_evidence_after_web",
                    }
                },
            }

    monkeypatch.setattr("src.api.handlers.stream.guardrails.check_input", no_refusal)
    events = [
        json.loads(event.removeprefix("data: ").strip())
        async for event in _token_generator(
            QueryRequest(question="What is the status?"),
            Agent(),
            MetricsTracker(),
            ConnectedRequest(),
        )
    ]

    assert [event["token"] for event in events if "token" in event] == [
        "I couldn't find enough reliable evidence."
    ]
    assert events[-1]["outcome"] == "abstained"


@pytest.mark.asyncio
async def test_stream_reports_graph_outcome_and_records_it(monkeypatch):
    async def no_refusal(_: str) -> None:
        return None

    class Agent:
        async def astream_events(self, *_args, **_kwargs):
            yield {
                "event": "on_chain_end",
                "name": "LangGraph",
                "data": {
                    "output": {
                        "messages": [HumanMessage(content="What is the status?")],
                        "outcome": "abstained",
                        "stop_reason": "insufficient_evidence_after_web",
                    }
                },
            }

    monkeypatch.setattr("src.api.handlers.stream.guardrails.check_input", no_refusal)
    tracker = MetricsTracker()
    events = [
        event
        async for event in _token_generator(
            QueryRequest(question="What is the status?"), Agent(), tracker, ConnectedRequest()
        )
    ]

    final = json.loads(events[-1].removeprefix("data: ").strip())
    assert final["outcome"] == "abstained"
    assert final["stop_reason"] == "insufficient_evidence_after_web"
    assert tracker.get_stats()["abstention_rate"] == 1.0
    assert tracker.get_stats()["avg_time_to_first_token_ms"] == 0.0


@pytest.mark.asyncio
async def test_stream_short_circuits_greetings_without_running_the_graph(monkeypatch):
    async def no_refusal(_: str) -> None:
        return None

    class Agent:
        async def astream_events(self, *_args, **_kwargs):
            raise AssertionError("graph must not run for a conversational turn")
            yield  # pragma: no cover — makes this an async generator

    monkeypatch.setattr("src.api.handlers.stream.guardrails.check_input", no_refusal)
    tracker = MetricsTracker()
    events = [
        json.loads(event.removeprefix("data: ").strip())
        async for event in _token_generator(
            QueryRequest(question="hi"), Agent(), tracker, ConnectedRequest()
        )
    ]

    assert events[0]["token"].startswith("I'm a document research assistant.")
    assert events[-1]["outcome"] == "conversational"
    assert events[-1]["stop_reason"] == "retrieval_not_requested"
    assert events[-1]["sources"] == []
    assert tracker.get_stats()["conversational_rate"] == 1.0
    assert tracker.get_stats()["abstention_rate"] == 0.0


@pytest.mark.asyncio
async def test_stream_records_time_to_first_visible_answer_token(monkeypatch):
    async def no_refusal(_: str) -> None:
        return None

    class Agent:
        async def astream_events(self, *_args, **_kwargs):
            yield {
                "event": "on_chat_model_stream",
                "metadata": {"langgraph_node": "answer"},
                "data": {"chunk": SimpleNamespace(content="Visible answer")},
            }

    now_ms = MagicMock(side_effect=[10_000.0] * 5 + [11_500.0, 13_000.0])
    monkeypatch.setattr("src.api.handlers.stream.guardrails.check_input", no_refusal)
    monkeypatch.setattr("src.api.handlers.stream._now_ms", now_ms)
    tracker = MetricsTracker()

    events = [
        event
        async for event in _token_generator(
            QueryRequest(question="What is the status?"),
            Agent(),
            tracker,
            ConnectedRequest(),
            10_000.0,
        )
    ]

    assert any("Visible answer" in event for event in events)
    assert tracker.get_stats()["avg_time_to_first_token_ms"] == 1500.0


@pytest.mark.asyncio
async def test_stream_collects_token_usage_from_model_end_events(monkeypatch):
    async def no_refusal(_: str) -> None:
        return None

    class Agent:
        async def astream_events(self, *_args, **_kwargs):
            yield {
                "event": "on_chat_model_stream",
                "metadata": {"langgraph_node": "answer"},
                "data": {"chunk": SimpleNamespace(content="Answer")},
            }
            # Two model calls in the turn (e.g. agent + answer); tokens sum, cost stays
            # None because streamed calls carry no provider cost.
            yield {
                "event": "on_chat_model_end",
                "data": {
                    "output": SimpleNamespace(
                        usage_metadata={"input_tokens": 100, "output_tokens": 10},
                        response_metadata={"token_usage": None},
                    )
                },
            }
            yield {
                "event": "on_chat_model_end",
                "data": {
                    "output": SimpleNamespace(
                        usage_metadata={"input_tokens": 40, "output_tokens": 8},
                        response_metadata={},
                    )
                },
            }

    monkeypatch.setattr("src.api.handlers.stream.guardrails.check_input", no_refusal)
    tracker = MetricsTracker()
    async for _ in _token_generator(
        QueryRequest(question="What is the status?", model="openai/gpt-5.6-luna"),
        Agent(),
        tracker,
        ConnectedRequest(),
    ):
        pass

    stats = tracker.get_stats()
    assert stats["avg_input_tokens"] == 140.0
    assert stats["avg_output_tokens"] == 18.0
    assert stats["avg_cost_per_query"] is None
    assert stats["models"]["openai/gpt-5.6-luna"]["queries"] == 1


@pytest.mark.asyncio
async def test_stream_logs_safe_completion_metadata(monkeypatch):
    async def no_refusal(_: str) -> None:
        return None

    answer = "The confidential rollout is complete."
    source_text = "Confidential evidence from the rollout plan."

    class Agent:
        async def astream_events(self, *_args, **_kwargs):
            yield {
                "event": "on_chat_model_stream",
                "metadata": {"langgraph_node": "answer"},
                "data": {"chunk": SimpleNamespace(content=answer)},
            }
            yield {
                "event": "on_chain_end",
                "name": "LangGraph",
                "data": {
                    "output": {
                        "messages": [
                            HumanMessage(content="What is the confidential rollout status?"),
                            ToolMessage(
                                content="retrieved evidence",
                                tool_call_id="retrieve-1",
                                name="retrieve_documents",
                                artifact={
                                    "documents": [
                                        {
                                            "content": source_text,
                                            "document_id": "rollout-plan",
                                            "chunk_id": "rollout-plan:1",
                                            "filename": "confidential.pdf",
                                            "source": "document",
                                        }
                                    ],
                                    "retrieval": {
                                        "query_count": 1,
                                        "query_shape": "single",
                                        "candidates": 40,
                                        "evidence": 1,
                                        "reranker_calls": 1,
                                    },
                                },
                            ),
                            AIMessage(content=f"{answer} [document:rollout-plan:1]"),
                        ],
                        "outcome": "document_answer",
                        "stop_reason": "document_evidence_sufficient",
                    }
                },
            }

    log_info = MagicMock()
    monkeypatch.setattr("src.api.handlers.stream.guardrails.check_input", no_refusal)
    monkeypatch.setattr("src.api.handlers.stream.logger.info", log_info)

    events = [
        event
        async for event in _token_generator(
            QueryRequest(question="What is the confidential rollout status?"),
            Agent(),
            MetricsTracker(),
            ConnectedRequest(),
        )
    ]

    completion = next(call for call in log_info.call_args_list if call.args == ("query_completed",))
    assert completion.kwargs["path"] == "document"
    assert completion.kwargs["outcome"] == "document_answer"
    assert completion.kwargs["stop_reason"] == "document_evidence_sufficient"
    assert completion.kwargs["sources_retrieved"] == 1
    assert completion.kwargs["sources_cited"] == 1
    assert completion.kwargs["web_search_triggered"] is False
    assert completion.kwargs["query_shape"] == "single"
    assert completion.kwargs["candidates"] == 40
    assert completion.kwargs["evidence"] == 1
    assert completion.kwargs["reranker_calls"] == 1
    assert completion.kwargs["latency_ms"] >= 0
    assert completion.kwargs["time_to_first_token_ms"] >= 0
    assert all(
        value not in str(completion.kwargs)
        for value in (answer, source_text, "confidential rollout", "confidential.pdf")
    )
    assert events[-1].startswith("data: ")


@pytest.mark.asyncio
async def test_stream_stops_before_guardrails_when_client_is_disconnected(monkeypatch):
    class DisconnectedRequest:
        async def is_disconnected(self) -> bool:
            return True

    check_input = AsyncMock()
    agent = MagicMock()
    monkeypatch.setattr("src.api.handlers.stream.guardrails.check_input", check_input)

    events = [
        event
        async for event in _token_generator(
            QueryRequest(question="What is the rollout status?"),
            agent,
            MetricsTracker(),
            DisconnectedRequest(),
        )
    ]

    assert events == []
    check_input.assert_not_awaited()
    agent.astream_events.assert_not_called()


@pytest.mark.asyncio
async def test_stream_stops_before_next_graph_event_after_client_disconnect(monkeypatch):
    async def no_refusal(_: str) -> None:
        return None

    class DisconnectAfterFirstEvent:
        def __init__(self):
            self.checks = 0

        async def is_disconnected(self) -> bool:
            self.checks += 1
            return self.checks >= 3

    class Agent:
        def __init__(self):
            self.events_consumed = 0
            self.closed = False

        async def astream_events(self, *_args, **_kwargs):
            try:
                for token in ["First", " second"]:
                    self.events_consumed += 1
                    yield {
                        "event": "on_chat_model_stream",
                        "metadata": {"langgraph_node": "answer"},
                        "data": {"chunk": SimpleNamespace(content=token)},
                    }
            finally:
                self.closed = True

    monkeypatch.setattr("src.api.handlers.stream.guardrails.check_input", no_refusal)
    agent = Agent()
    events = [
        event
        async for event in _token_generator(
            QueryRequest(question="What is the rollout status?"),
            agent,
            MetricsTracker(),
            DisconnectAfterFirstEvent(),
        )
    ]

    assert agent.events_consumed == 1
    assert agent.closed is True
    assert all('"done": true' not in event for event in events)


@pytest.mark.asyncio
async def test_stream_propagates_cancellation_to_an_in_flight_graph(monkeypatch):
    async def no_refusal(_: str) -> None:
        return None

    class Agent:
        def __init__(self):
            self.started = asyncio.Event()
            self.cancelled = asyncio.Event()

        async def astream_events(self, *_args, **_kwargs):
            try:
                self.started.set()
                await asyncio.Event().wait()
                yield {}
            except asyncio.CancelledError:
                self.cancelled.set()
                raise

    monkeypatch.setattr("src.api.handlers.stream.guardrails.check_input", no_refusal)
    agent = Agent()
    generator = _token_generator(
        QueryRequest(question="What is the rollout status?"),
        agent,
        MetricsTracker(),
        ConnectedRequest(),
    )
    next_event = asyncio.create_task(anext(generator))
    await agent.started.wait()

    next_event.cancel()
    with pytest.raises(asyncio.CancelledError):
        await next_event

    assert agent.cancelled.is_set()


@pytest.mark.asyncio
async def test_stream_stops_at_whole_query_deadline_before_graph(monkeypatch):
    cancelled = asyncio.Event()

    async def slow_guardrail(_: str) -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    agent = MagicMock()
    monkeypatch.setattr("src.api.handlers.stream.guardrails.check_input", slow_guardrail)
    monkeypatch.setattr(
        "src.api.handlers.stream.get_settings",
        lambda: SimpleNamespace(
            QUERY_TIMEOUT_SECONDS=0.01,
            LLM_MODEL="anthropic/claude-sonnet-4.6",
            ANSWER_CACHE_ENABLED=False,
            PLANNER_MODEL="openai/gpt-5.6-luna",
            ASSESSOR_MODEL="openai/gpt-5.6-luna",
        ),
    )

    events = [
        event
        async for event in _token_generator(
            QueryRequest(question="What is the rollout status?"),
            agent,
            MetricsTracker(),
            ConnectedRequest(),
        )
    ]

    assert cancelled.is_set()
    assert json.loads(events[-1].removeprefix("data: ").strip()) == {
        "error": "The request timed out. Please try again.",
        "done": True,
    }
    agent.astream_events.assert_not_called()


async def _first_turn(*_args, **_kwargs) -> bool:
    return True


class _DocumentAnswerAgent:
    """Streams a single document_answer turn with one cited source."""

    def __init__(self, answer: str):
        self._answer = answer

    async def astream_events(self, *_args, **_kwargs):
        yield {
            "event": "on_chat_model_stream",
            "metadata": {"langgraph_node": "answer"},
            "data": {"chunk": SimpleNamespace(content=self._answer)},
        }
        yield {
            "event": "on_chain_end",
            "name": "LangGraph",
            "data": {
                "output": {
                    "messages": [
                        HumanMessage(content="What is the rollout status?"),
                        ToolMessage(
                            content="retrieved evidence",
                            tool_call_id="retrieve-1",
                            name="retrieve_documents",
                            artifact={
                                "documents": [
                                    {
                                        "content": "The rollout plan is complete.",
                                        "document_id": "rollout-plan",
                                        "chunk_id": "rollout-plan:1",
                                        "filename": "plan.pdf",
                                        "source": "document",
                                    }
                                ],
                                "retrieval": {
                                    "query_count": 1,
                                    "query_shape": "single",
                                    "candidates": 1,
                                    "evidence": 1,
                                    "reranker_calls": 1,
                                },
                            },
                        ),
                        AIMessage(content=f"{self._answer} [document:rollout-plan:1]"),
                    ],
                    "outcome": "document_answer",
                    "stop_reason": "document_evidence_sufficient",
                }
            },
        }


@pytest.mark.asyncio
async def test_stream_serves_cached_answer_on_first_turn(monkeypatch):
    async def no_refusal(_: str) -> None:
        return None

    cached = {"answer": "The rollout is complete.", "sources": [{"source_id": "document:x"}]}
    monkeypatch.setattr("src.api.handlers.stream.guardrails.check_input", no_refusal)
    monkeypatch.setattr("src.api.handlers.stream._is_first_turn", _first_turn)
    monkeypatch.setattr(
        "src.api.handlers.stream.answer_cache.lookup", MagicMock(return_value=cached)
    )

    agent = MagicMock()
    agent.aupdate_state = AsyncMock()
    tracker = MetricsTracker()
    events = [
        json.loads(event.removeprefix("data: ").strip())
        async for event in _token_generator(
            QueryRequest(question="What is the rollout status?"),
            agent,
            tracker,
            ConnectedRequest(),
        )
    ]

    assert [e["token"] for e in events if "token" in e] == ["The rollout is complete."]
    done = events[-1]
    assert done["outcome"] == "document_answer"
    assert done["stop_reason"] == "document_evidence_sufficient"
    assert done["sources"] == [{"source_id": "document:x"}]
    agent.astream_events.assert_not_called()
    agent.aupdate_state.assert_awaited_once()
    assert tracker.get_stats()["cache_hit_rate"] == 1.0
    assert tracker.get_stats()["document_answer_rate"] == 1.0


@pytest.mark.asyncio
async def test_stream_populates_cache_after_first_turn_document_answer(monkeypatch):
    async def no_refusal(_: str) -> None:
        return None

    store = MagicMock()
    monkeypatch.setattr("src.api.handlers.stream.guardrails.check_input", no_refusal)
    monkeypatch.setattr("src.api.handlers.stream._is_first_turn", _first_turn)
    monkeypatch.setattr("src.api.handlers.stream.answer_cache.lookup", MagicMock(return_value=None))
    monkeypatch.setattr("src.api.handlers.stream.answer_cache.store", store)

    async for _ in _token_generator(
        QueryRequest(question="What is the rollout status?"),
        _DocumentAnswerAgent("The rollout is complete."),
        MetricsTracker(),
        ConnectedRequest(),
    ):
        pass

    store.assert_called_once()
    question, answer, sources = store.call_args.args[:3]
    assert question == "What is the rollout status?"
    assert answer == "The rollout is complete."
    assert sources[0]["chunk_id"] == "rollout-plan:1"


@pytest.mark.asyncio
async def test_stream_skips_cache_on_follow_up_turn(monkeypatch):
    async def no_refusal(_: str) -> None:
        return None

    lookup = MagicMock()
    store = MagicMock()
    monkeypatch.setattr("src.api.handlers.stream.guardrails.check_input", no_refusal)
    # _is_first_turn stays patched to False by the autouse fixture.
    monkeypatch.setattr("src.api.handlers.stream.answer_cache.lookup", lookup)
    monkeypatch.setattr("src.api.handlers.stream.answer_cache.store", store)

    async for _ in _token_generator(
        QueryRequest(question="What is the rollout status?"),
        _DocumentAnswerAgent("The rollout is complete."),
        MetricsTracker(),
        ConnectedRequest(),
    ):
        pass

    lookup.assert_not_called()
    store.assert_not_called()


@pytest.mark.asyncio
async def test_stream_stops_at_whole_query_deadline_during_graph(monkeypatch):
    async def no_refusal(_: str) -> None:
        return None

    class Agent:
        def __init__(self):
            self.cancelled = asyncio.Event()

        async def astream_events(self, *_args, **_kwargs):
            try:
                await asyncio.Event().wait()
                yield {}
            except asyncio.CancelledError:
                self.cancelled.set()
                raise

    agent = Agent()
    tracker = MetricsTracker()
    monkeypatch.setattr("src.api.handlers.stream.guardrails.check_input", no_refusal)
    monkeypatch.setattr(
        "src.api.handlers.stream.get_settings",
        lambda: SimpleNamespace(
            QUERY_TIMEOUT_SECONDS=0.01,
            LLM_MODEL="anthropic/claude-sonnet-4.6",
            ANSWER_CACHE_ENABLED=False,
            PLANNER_MODEL="openai/gpt-5.6-luna",
            ASSESSOR_MODEL="openai/gpt-5.6-luna",
        ),
    )

    events = [
        event
        async for event in _token_generator(
            QueryRequest(question="What is the rollout status?"),
            agent,
            tracker,
            ConnectedRequest(),
        )
    ]

    assert agent.cancelled.is_set()
    assert json.loads(events[-1].removeprefix("data: ").strip()) == {
        "error": "The request timed out. Please try again.",
        "done": True,
    }
    assert tracker.get_stats()["total_queries"] == 1
    assert tracker._events[-1].path == "timeout"
