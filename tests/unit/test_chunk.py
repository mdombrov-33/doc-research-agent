from src.core.ingestion.chunk import chunk_text


def test_drops_short_chunks():
    assert chunk_text("too short") == []


def test_keeps_substantial_chunks():
    text = "Lorem ipsum dolor sit amet. " * 200
    chunks = chunk_text(text)
    assert chunks
    assert all(len(c.strip()) >= 100 for c in chunks)
