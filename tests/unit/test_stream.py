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
                    "filename": "report.pdf",
                    "chunk_index": 2,
                    "chunk_length": 120,
                    "source": "vectorstore",
                }
            ],
        ),
        ToolMessage(
            content="web result",
            tool_call_id="web-1",
            name="web_search",
            artifact=[
                {
                    "filename": "web",
                    "chunk_index": 0,
                    "chunk_length": 50,
                    "source": "web",
                }
            ],
        ),
    ]

    sources, retrieved_total, web_search_triggered = _turn_sources(messages)

    assert sources == [
        {
            "filename": "report.pdf",
            "chunk_index": 2,
            "chunk_length": 120,
            "source": "vectorstore",
        },
        {"filename": "web", "chunk_index": 0, "chunk_length": 50, "source": "web"},
    ]
    assert retrieved_total == 2
    assert web_search_triggered is True


def test_turn_sources_deduplicates_repeated_artifacts():
    artifact = {
        "filename": "report.pdf",
        "chunk_index": 2,
        "chunk_length": 120,
        "source": "vectorstore",
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

    assert sources == [artifact]
    assert retrieved_total == 2
    assert web_search_triggered is False
