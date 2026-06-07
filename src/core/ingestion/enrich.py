from pathlib import Path

from spacy.language import Language


def enrich_chunks(chunks: list[str], filename: str, nlp: Language) -> list[dict]:
    enriched = []
    file_ext = Path(filename).suffix.lower()

    for i, chunk in enumerate(chunks):
        doc = nlp(chunk)

        entities = []
        entity_labels = []
        for ent in doc.ents:
            entities.append(ent.text)
            entity_labels.append(ent.label_)

        keywords = [
            token.text
            for token in doc
            if token.pos_ in {"NOUN", "PROPN"} and not token.is_stop and len(token.text) > 2
        ]

        enriched.append(
            {
                "text": chunk,
                "chunk_index": i,
                "chunk_length": len(chunk),
                "entities": entities[:10],
                "entity_types": entity_labels[:10],
                "keywords": list(set(keywords))[:15],
                "file_extension": file_ext,
            }
        )

    return enriched
