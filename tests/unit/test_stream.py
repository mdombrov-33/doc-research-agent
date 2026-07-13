from langchain_core.messages import HumanMessage, ToolMessage

from src.api.handlers.stream import _turn_sources


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
            "source_id": "document:doc-report:2",
            "source_type": "document",
            "title": "report.pdf",
            "document_id": "doc-report",
            "chunk_id": "doc-report:2",
            "page": 3,
            "excerpt": "The document says the rollout is complete.",
        },
        {
            "source_id": "web:https://example.com/announcement",
            "source_type": "web",
            "title": "Official announcement",
            "url": "https://example.com/announcement",
            "excerpt": "The official announcement confirms it.",
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
