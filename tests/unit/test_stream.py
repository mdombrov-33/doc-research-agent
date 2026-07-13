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

    sources, retrieved_total, web_search_triggered = _turn_sources(messages)

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

    sources, retrieved_total, web_search_triggered = _turn_sources(messages)

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
