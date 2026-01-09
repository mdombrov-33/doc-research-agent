import os
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from src.api.schemas import QueryRequest, QueryResponse, UploadResponse
from src.config import get_settings
from src.core.agent import get_agent
from src.core.document_processing.document_processor import DocumentProcessor
from src.utils.logger import logger

settings = get_settings()
router = APIRouter()


@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
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

        return UploadResponse(**result)

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


@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    try:
        logger.info(f"Received query: {request.question}")

        agent = get_agent()

        inputs: dict[str, str | bool | list[str] | int] = {
            "question": request.question,
            "generation": "",
            "web_search": False,
            "documents": [],
            "retrieval_attempts": 0,
            "generation_attempts": 0,
        }

        result = agent.invoke(inputs)  # type: ignore[arg-type]

        answer = result.get("generation", "No answer generated")
        sources_count = len(result.get("documents", []))

        logger.info(f"Query completed. Answer length: {len(answer)}, Sources: {sources_count}")

        return QueryResponse(question=request.question, answer=answer, sources_count=sources_count)

    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")
