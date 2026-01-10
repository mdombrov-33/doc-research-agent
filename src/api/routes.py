from typing import Any

from fastapi import APIRouter, File, UploadFile

from src.api.handlers.query import handle_query
from src.api.handlers.upload import handle_upload
from src.api.schemas import QueryRequest, QueryResponse, UploadResponse
from src.core.evaluation.metrics import get_evaluation_tracker

router = APIRouter()


@router.head("/ping")
async def ping():
    return {"message": "pong"}


@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    result = await handle_upload(file)
    return UploadResponse(**result)


@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    result = await handle_query(request)
    return QueryResponse(**result)


@router.get("/evaluation/stats")
async def get_evaluation_stats() -> dict[str, Any]:
    tracker = get_evaluation_tracker()
    return tracker.get_stats()
