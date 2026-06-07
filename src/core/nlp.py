from functools import lru_cache

import spacy

from src.core.exceptions import ModelLoadError
from src.utils.logger import logger


@lru_cache(maxsize=1)
def get_spacy_model():
    try:
        nlp = spacy.load("en_core_web_sm")
        logger.info("spacy_model_loaded", model="en_core_web_sm")
        return nlp
    except OSError as exc:
        raise ModelLoadError("spaCy model 'en_core_web_sm' not found") from exc


def extract_entities(text: str) -> list[str]:
    """Named entities in `text`, deduped. Same NER used at ingestion, so a
    query's entities match the `entities` stored in each chunk's payload."""
    nlp = get_spacy_model()
    doc = nlp(text)
    return list({ent.text for ent in doc.ents})
