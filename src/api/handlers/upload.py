import os
import uuid
from pathlib import Path
from typing import Any

from fastapi import HTTPException, UploadFile

from src.config import get_settings
from src.core.document_processing.document_processor import DocumentProcessor
from src.utils.logger import logger

settings = get_settings()


async def handle_upload(file: UploadFile) -> dict[str, Any]:
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

        logger.info(f"Saved uploaded file to {file_path}")

        processor = DocumentProcessor()
        result = await processor.process_and_store(str(file_path), file.filename)

        return result

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to process document: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    finally:
        if file_path.exists():
            os.remove(file_path)
            logger.info(f"Cleaned up temp file {file_path}")
