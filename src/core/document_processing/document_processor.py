import uuid
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client.models import PointStruct

from src.config import get_settings
from src.core.document_processing.text_processor import TextExtractor, get_spacy_model
from src.core.vector_store import get_embeddings, get_qdrant_client
from src.utils.logger import logger

settings = get_settings()


class DocumentProcessor:
    def __init__(self):
        self.extractor = TextExtractor()
        self.embeddings = get_embeddings()
        self.qdrant_client = get_qdrant_client()
        self.nlp = get_spacy_model()

    async def process_and_store(self, file_path: str, filename: str) -> dict:
        document_id = str(uuid.uuid4())
        logger.info(f"Processing document {filename} with ID {document_id}")

        raw_text = await self.extractor.extract_from_file(file_path, filename)
        if not raw_text.strip():
            raise ValueError("No text extracted from document")

        chunks = self._chunk_text(raw_text)
        logger.info(f"Created {len(chunks)} chunks")

        enriched_chunks = self._enrich_chunks(chunks, filename)
        logger.info(f"Enriched {len(enriched_chunks)} chunks with metadata")

        chunk_texts = [chunk["text"] for chunk in enriched_chunks]
        vectors = [self.embeddings.embed_query(text) for text in chunk_texts]
        logger.info(f"Generated {len(vectors)} embeddings")

        self._store_in_qdrant(document_id, filename, enriched_chunks, vectors)

        return {
            "document_id": document_id,
            "filename": filename,
            "chunks_created": len(chunks),
            "file_size": Path(file_path).stat().st_size,
        }

    def _chunk_text(self, text: str) -> list[str]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=240,
            separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
            length_function=len,
        )
        chunks = splitter.split_text(text)

        chunks = [chunk for chunk in chunks if len(chunk.strip()) >= 100]

        logger.info(
            f"Chunked into {len(chunks)} pieces, avg size: {
                sum(len(c) for c in chunks) // len(chunks) if chunks else 0
            } chars"
        )
        return chunks

    def _enrich_chunks(self, chunks: list[str], filename: str) -> list[dict]:
        enriched = []
        file_ext = Path(filename).suffix.lower()

        for i, chunk in enumerate(chunks):
            doc = self.nlp(chunk)

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

    def _store_in_qdrant(
        self,
        document_id: str,
        filename: str,
        enriched_chunks: list[dict],
        vectors: list[list[float]],
    ):
        points = []
        for chunk_data, vector in zip(enriched_chunks, vectors):
            payload = {
                "document_id": document_id,
                "filename": filename,
                "page_content": chunk_data["text"],  # LangChain expects "page_content"
                "chunk_index": chunk_data["chunk_index"],
                "chunk_length": chunk_data["chunk_length"],
                "entities": chunk_data["entities"],
                "entity_types": chunk_data["entity_types"],
                "keywords": chunk_data["keywords"],
                "file_extension": chunk_data["file_extension"],
            }

            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload=payload,
                )
            )

        self.qdrant_client.upsert(collection_name=settings.QDRANT_COLLECTION_NAME, points=points)
        logger.info(f"Stored {len(points)} enriched points in Qdrant")
