import uuid
from pathlib import Path

from langchain_qdrant import QdrantVectorStore
from spacy.language import Language

from src.core.exceptions import EmptyDocumentError
from src.core.ingestion.chunk import chunk_text
from src.core.ingestion.enrich import enrich_chunks
from src.core.ingestion.extract import extract_from_file
from src.core.ingestion.index import index_chunks
from src.utils.logger import logger


async def process_and_store(
    file_path: str,
    filename: str,
    vector_store: QdrantVectorStore,
    nlp: Language,
) -> dict:
    document_id = str(uuid.uuid4())

    raw_text = await extract_from_file(file_path, filename)
    if not raw_text.strip():
        raise EmptyDocumentError("No text extracted from document")

    chunks = chunk_text(raw_text)
    enriched_chunks = enrich_chunks(chunks, filename, nlp)
    index_chunks(vector_store, document_id, filename, enriched_chunks)

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
