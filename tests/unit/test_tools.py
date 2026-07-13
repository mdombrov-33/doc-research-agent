from src.core.agent.tools import _web_evidence


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
