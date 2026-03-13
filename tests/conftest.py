from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def mock_spacy_tokenize():
    """Replace spaCy tokenizer with simple whitespace split so tests don't require the model."""
    with patch(
        "src.core.retrieval.bm25_indexer.tokenize",
        side_effect=lambda text: text.lower().split(),
    ):
        yield