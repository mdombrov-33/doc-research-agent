import os
import uuid
from pathlib import Path
from typing import Any

from fastapi import HTTPException, UploadFile
from langchain_qdrant import QdrantVectorStore
from spacy.language import Language

from src.config import Settings
from src.core.exceptions import DocumentProcessingError, EmptyDocumentError
from src.core.ingestion.pipeline import process_and_store
from src.utils.logger import logger


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
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)

        result = await process_and_store(str(file_path), file.filename, vector_store, nlp)
        logger.info("upload_complete", **result)
        return result

    except EmptyDocumentError as e:
        logger.info("upload_rejected", filename=file.filename, reason=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except DocumentProcessingError as e:
        logger.error("upload_processing_failed", filename=file.filename, error=str(e))
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    except Exception as e:
        logger.error("upload_failed", filename=file.filename, error=str(e))
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    finally:
        if file_path.exists():
            os.remove(file_path)
