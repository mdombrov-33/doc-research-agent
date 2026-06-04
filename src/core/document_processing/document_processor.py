import uuid
from pathlib import Path

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from spacy.language import Language

from src.core.document_processing.text_processor import TextExtractor
from src.core.exceptions import EmptyDocumentError
from src.utils.logger import logger


class DocumentProcessor:
    def __init__(self, vector_store: QdrantVectorStore, nlp: Language):
        self.extractor = TextExtractor()
        self.vector_store = vector_store
        self.nlp = nlp

    async def process_and_store(self, file_path: str, filename: str) -> dict:
        document_id = str(uuid.uuid4())

        raw_text = await self.extractor.extract_from_file(file_path, filename)
        if not raw_text.strip():
            raise EmptyDocumentError("No text extracted from document")

        chunks = self._chunk_text(raw_text)
        enriched_chunks = self._enrich_chunks(chunks, filename)

        self._store_in_qdrant(document_id, filename, enriched_chunks)

        logger.info(
            "document_processed",
            document_id=document_id,
            filename=filename,
            chunks=len(chunks),
        )

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
    ):
        documents = [
            Document(
                page_content=chunk_data["text"],
                metadata={
                    "document_id": document_id,
                    "filename": filename,
                    "chunk_index": chunk_data["chunk_index"],
                    "chunk_length": chunk_data["chunk_length"],
                    "entities": chunk_data["entities"],
                    "entity_types": chunk_data["entity_types"],
                    "keywords": chunk_data["keywords"],
                    "file_extension": chunk_data["file_extension"],
                },
            )
            for chunk_data in enriched_chunks
        ]

        # add_documents computes both the dense (OpenAI) and sparse (BM25) vectors.
        self.vector_store.add_documents(documents)
