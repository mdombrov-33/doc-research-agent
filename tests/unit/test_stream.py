import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.api.handlers.stream import _token_generator, _turn_sources
from src.api.schemas import QueryRequest
from src.core.monitoring.tracker import MetricsTracker


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
                    "metadata": {"langgraph_node": "agent"},
                    "data": {"chunk": SimpleNamespace(content=content)},
                }

    monkeypatch.setattr("src.api.handlers.stream.guardrails.check_input", no_refusal)
    events = [
        event
        async for event in _token_generator(
            QueryRequest(question="What is the rollout status?"), Agent(), MetricsTracker()
        )
    ]
    visible_answer = "".join(
        json.loads(event.removeprefix("data: ").strip())["token"]
        for event in events
        if "token" in json.loads(event.removeprefix("data: ").strip())
    )

    assert visible_answer == "The rollout is complete."
