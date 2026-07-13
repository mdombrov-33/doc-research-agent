from src.core.agent.tools import _web_evidence, format_docs


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
