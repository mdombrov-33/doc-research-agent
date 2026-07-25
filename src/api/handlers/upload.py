import asyncio
import hashlib
import os
import uuid
from pathlib import Path
from typing import Any

from fastapi import HTTPException, UploadFile
from langchain_qdrant import QdrantVectorStore
from spacy.language import Language

from src.config import Settings
from src.core import answer_cache
from src.core.exceptions import DocumentLimitError, DocumentProcessingError, EmptyDocumentError
from src.core.ingestion.dedupe import find_duplicate
from src.core.ingestion.pipeline import process_and_store
from src.utils.logger import logger


def _file_size_bucket(size_bytes: int) -> str:
    if size_bytes <= 1024:
        return "0-1KiB"
    if size_bytes <= 1024 * 1024:
        return "1KiB-1MiB"
    if size_bytes <= 10 * 1024 * 1024:
        return "1MiB-10MiB"
    return "10MiB+"


async def handle_upload(
    file: UploadFile,
    settings: Settings,
    vector_store: QdrantVectorStore,
    nlp: Language,
) -> dict[str, Any]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    allowed_extensions = {".pdf", ".docx", ".txt"}
    file_ext = Path(file.filename).suffix.lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}",
        )

    upload_dir = Path(settings.UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)

    temp_filename = f"{uuid.uuid4()}{file_ext}"
    file_path = upload_dir / temp_filename

    try:
        hasher = hashlib.sha256()
        with open(file_path, "wb") as f:
            bytes_written = 0
            while content := await file.read(settings.UPLOAD_READ_CHUNK_BYTES):
                bytes_written += len(content)
                if bytes_written > settings.MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail="File exceeds the upload size limit.",
                    )
                hasher.update(content)
                f.write(content)
        file_sha256 = hasher.hexdigest()

        # Identical bytes are already indexed: return the existing document without re-embedding,
        # and leave the corpus version untouched so the answer cache is not needlessly flushed.
        duplicate_id = await asyncio.to_thread(find_duplicate, vector_store, file_sha256)
        if duplicate_id:
            logger.info(
                "upload_duplicate",
                document_id=duplicate_id,
                file_extension=file_ext,
                file_size_bucket=_file_size_bucket(bytes_written),
            )
            return {
                "document_id": duplicate_id,
                "filename": file.filename,
                "chunks_created": 0,
                "file_size": bytes_written,
                "duplicate": True,
            }

        result = await process_and_store(
            str(file_path), file.filename, file_sha256, vector_store, nlp, settings
        )
        # A new document can invalidate any cached answer; bump the corpus version so only
        # answers formed after this upload are served. Never fail an indexed upload over this.
        if settings.ANSWER_CACHE_ENABLED:
            try:
                await asyncio.to_thread(answer_cache.bump_corpus_version)
            except Exception as error:
                logger.warning(
                    "answer_cache_version_bump_failed", failure_type=type(error).__name__
                )
        logger.info(
            "upload_complete",
            document_id=result["document_id"],
            file_extension=file_ext,
            file_size_bucket=_file_size_bucket(bytes_written),
            chunks_created=result["chunks_created"],
        )
        return result

    except HTTPException:
        raise
    except EmptyDocumentError:
        logger.info("upload_rejected", file_extension=file_ext, reason="empty_document")
        raise HTTPException(
            status_code=400,
            detail="No text could be extracted from this document.",
        )
    except DocumentLimitError:
        logger.info("upload_rejected", file_extension=file_ext, reason="processing_limit")
        raise HTTPException(status_code=413, detail="Document exceeds processing limits.")
    except DocumentProcessingError as error:
        logger.error("upload_processing_failed", failure_type=type(error).__name__)
        raise HTTPException(
            status_code=500,
            detail="Unable to process the document. Please try again.",
        )
    except Exception as error:
        logger.error("upload_failed", failure_type=type(error).__name__)
        raise HTTPException(
            status_code=500,
            detail="Unable to process the document. Please try again.",
        )
    finally:
        if file_path.exists():
            os.remove(file_path)
