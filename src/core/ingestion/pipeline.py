import asyncio
import uuid
from pathlib import Path

from langchain_qdrant import QdrantVectorStore
from spacy.language import Language

from src.config import Settings
from src.core.exceptions import DocumentLimitError, EmptyDocumentError
from src.core.ingestion.chunk import chunk_text
from src.core.ingestion.enrich import enrich_chunks
from src.core.ingestion.extract import extract_from_file
from src.core.ingestion.index import index_chunks
from src.utils.logger import logger


async def process_and_store(
    file_path: str,
    filename: str,
    file_sha256: str,
    vector_store: QdrantVectorStore,
    nlp: Language,
    settings: Settings,
) -> dict:
    document_id = str(uuid.uuid4())

    raw_text = await extract_from_file(
        file_path,
        filename,
        max_pdf_pages=settings.MAX_PDF_PAGES,
        max_extracted_characters=settings.MAX_EXTRACTED_CHARACTERS,
    )
    if not raw_text.strip():
        raise EmptyDocumentError("No text extracted from document")

    chunks = chunk_text(raw_text)
    if len(chunks) > settings.MAX_CHUNKS_PER_DOCUMENT:
        raise DocumentLimitError("Chunk limit exceeded")

    enriched_chunks = enrich_chunks(chunks, filename, nlp)
    await asyncio.to_thread(
        index_chunks, vector_store, document_id, filename, file_sha256, enriched_chunks
    )

    logger.info(
        "document_processed",
        document_id=document_id,
        file_extension=Path(filename).suffix.lower(),
        chunks=len(chunks),
    )

    return {
        "document_id": document_id,
        "filename": filename,
        "chunks_created": len(chunks),
        "file_size": Path(file_path).stat().st_size,
    }
