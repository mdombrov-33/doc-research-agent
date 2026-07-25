import asyncio
import hashlib
import os
from unittest.mock import AsyncMock, MagicMock

from src.config import Settings, get_settings


def test_eval_ingestion_passes_current_settings(tmp_path, monkeypatch):
    previous_collection = os.environ.get("QDRANT_COLLECTION_NAME")
    from evals import run_eval

    try:
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "example.txt").write_text("document", encoding="utf-8")
        vector_store = MagicMock()
        nlp = MagicMock()
        settings = Settings()
        process = AsyncMock()
        monkeypatch.setattr(run_eval, "CORPUS_DIR", corpus)
        monkeypatch.setattr(run_eval, "get_ingestion_vector_store", lambda: vector_store)
        monkeypatch.setattr(run_eval, "get_spacy_model", lambda: nlp)
        monkeypatch.setattr(run_eval, "get_settings", lambda: settings)
        monkeypatch.setattr(run_eval, "process_and_store", process)

        asyncio.run(run_eval._ingest_corpus())

        expected_sha = hashlib.sha256(b"document").hexdigest()
        process.assert_awaited_once_with(
            str(corpus / "example.txt"), "example.txt", expected_sha, vector_store, nlp, settings
        )
    finally:
        if previous_collection is None:
            os.environ.pop("QDRANT_COLLECTION_NAME", None)
        else:
            os.environ["QDRANT_COLLECTION_NAME"] = previous_collection
        get_settings.cache_clear()
