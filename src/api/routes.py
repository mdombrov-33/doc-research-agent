from fastapi import APIRouter, File, UploadFile

from src.api.handlers.query import handle_query
from src.api.handlers.upload import handle_upload
from src.api.schemas import QueryRequest, QueryResponse, UploadResponse

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
