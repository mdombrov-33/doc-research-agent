from unittest.mock import MagicMock

from ddgs.exceptions import TimeoutException

from src.config import Settings
from src.core.agent import tools
from src.core.agent.tools import _web_evidence, format_docs


def _ddgs_client(*, text_result=None, text_error=None):
    client = MagicMock()
    client.__enter__.return_value = client
    if text_error is not None:
        client.text.side_effect = text_error
    else:
        client.text.return_value = text_result
    return client


def test_web_evidence_keeps_only_results_that_can_be_cited():
    evidence = _web_evidence(
        [
            {
                "title": "Official source",
                "link": "https://example.com/release",
                "snippet": "The release is available now.",
            },
            {
                "title": "Invalid scheme",
                "link": "ftp://example.com/file",
                "snippet": "Not a web citation.",
            },
            {"title": "Missing snippet", "link": "https://example.com/missing"},
        ]
    )

    assert evidence == [
        {
            "content": "The release is available now.",
            "title": "Official source",
            "url": "https://example.com/release",
            "rank": 1,
            "source": "web",
        }
    ]


def test_format_docs_exposes_only_real_source_ids_to_the_model():
    formatted = format_docs(
        [
            {
                "content": "Document evidence",
                "document_id": "doc-1",
                "chunk_id": "doc-1:0",
                "filename": "notes.txt",
                "source": "document",
            },
            {
                "content": "Web evidence",
                "title": "Official source",
                "url": "https://example.com/release",
                "source": "web",
            },
        ]
    )

    assert "[Source ID: document:doc-1:0]" in formatted
    assert "[Source ID: web:https://example.com/release]" in formatted


def test_search_web_retries_a_transient_ddgs_timeout_once(monkeypatch):
    first = _ddgs_client(text_error=TimeoutException("timed out"))
    second = _ddgs_client(
        text_result=[
            {
                "title": "Official source",
                "href": "https://example.com/release",
                "body": "The release is available now.",
            }
        ]
    )
    ddgs = MagicMock(side_effect=[first, second])
    monkeypatch.setattr(tools, "DDGS", ddgs, raising=False)
    monkeypatch.setattr(
        tools,
        "DuckDuckGoSearchResults",
        MagicMock(),
        raising=False,
    )
    monkeypatch.setattr(
        tools,
        "get_settings",
        lambda: Settings(WEB_SEARCH_TIMEOUT_SECONDS=11),
        raising=False,
    )
    monkeypatch.setattr(tools.time, "sleep", MagicMock())

    _, evidence = tools.search_web("release")

    assert evidence == [
        {
            "content": "The release is available now.",
            "title": "Official source",
            "url": "https://example.com/release",
            "rank": 1,
            "source": "web",
        }
    ]
    assert [call.kwargs["timeout"] for call in ddgs.call_args_list] == [11, 11]


def test_search_web_does_not_retry_a_permanent_failure(monkeypatch):
    ddgs = MagicMock(side_effect=[_ddgs_client(text_error=ValueError("invalid query"))])
    monkeypatch.setattr(tools, "DDGS", ddgs)
    monkeypatch.setattr(tools, "get_settings", lambda: Settings(WEB_SEARCH_TIMEOUT_SECONDS=11))

    content, evidence = tools.search_web("release")

    assert (content, evidence) == ("Web search failed.", [])
    ddgs.assert_called_once_with(timeout=11)
