import spacy
from spacy.language import Language

from src.core.exceptions import ModelLoadError
from src.utils.logger import logger

_nlp: Language | None = None


def get_spacy_model() -> Language:
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_sm")
            logger.info("spacy_model_loaded", model="en_core_web_sm", context="tokenizer")
        except OSError as exc:
            logger.error("spacy_model_not_found", model="en_core_web_sm")
            raise ModelLoadError("spaCy model 'en_core_web_sm' not found") from exc
    return _nlp


def tokenize(text: str) -> list[str]:
    if not text or not text.strip():
        return []

    nlp = get_spacy_model()
    doc = nlp(text.lower())

    tokens = [
        token.lemma_
        for token in doc
        if not token.is_stop
        and not token.is_punct
        and not token.is_space
        and len(token.text) > 1
        and token.text.strip()
    ]

    if not tokens:
        tokens = [
            token.text.lower()
            for token in doc
            if token.is_alpha and len(token.text) > 1
        ]

    return tokens
