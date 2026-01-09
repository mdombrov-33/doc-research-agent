import spacy
from spacy.language import Language

from src.utils.logger import logger

_nlp: Language | None = None


def get_spacy_model() -> Language:
    """Get or load spaCy model (cached)."""
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_sm")
            logger.info("Loaded spaCy model for tokenization")
        except OSError:
            logger.error(
                "spaCy model 'en_core_web_sm' not found"  # noqa: E501
            )
            raise
    return _nlp


def tokenize(text: str) -> list[str]:
    """
    Tokenize text using spaCy for BM25 indexing.

    Args:
        text: Input text to tokenize

    Returns:
        List of tokens (lemmatized, filtered for relevance)
    """
    if not text or not text.strip():
        return []

    nlp = get_spacy_model()
    doc = nlp(text.lower())

    tokens = [
        token.lemma_
        for token in doc
        if not token.is_stop  # Remove stop words
        and not token.is_punct  # Remove punctuation
        and not token.is_space  # Remove whitespace
        and len(token.text) > 1  # Remove single chars
        and token.text.strip()  # Remove empty
    ]

    # Fallback: if no tokens after filtering, use all alphabetic tokens
    if not tokens:
        tokens = [
            token.text.lower()
            for token in doc
            if token.is_alpha and len(token.text) > 1
        ]

    return tokens
