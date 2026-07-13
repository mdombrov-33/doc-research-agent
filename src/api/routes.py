from typing import Any

from fastapi import APIRouter, Depends, File, Request, UploadFile
from langchain_qdrant import QdrantVectorStore
from spacy.language import Language

from src.api.dependencies import (
    Settings,
    get_agent,
    get_metrics_tracker,
    get_nlp,
    get_settings,
    get_vector_store,
)
from src.api.handlers.stream import handle_stream
from src.api.handlers.upload import handle_upload
from src.api.rate_limit import limiter, rate_limit
from src.api.schemas import QueryRequest, UploadResponse
from src.core.monitoring.tracker import MetricsTracker

router = APIRouter()


@router.post("/upload", response_model=UploadResponse)
@limiter.limit(rate_limit)
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    settings: Settings = Depends(get_settings),
    vector_store: QdrantVectorStore = Depends(get_vector_store),
    nlp: Language = Depends(get_nlp),
):
    result = await handle_upload(file, settings, vector_store, nlp)
    return UploadResponse(**result)


@router.post("/stream")
@limiter.limit(rate_limit)
async def stream_query(
    request: Request,
    payload: QueryRequest,
    agent: Any = Depends(get_agent),
    tracker: MetricsTracker = Depends(get_metrics_tracker),
):
    return await handle_stream(payload, agent, tracker, request)


@router.get("/monitoring/stats")
async def get_monitoring_stats(
    tracker: MetricsTracker = Depends(get_metrics_tracker),
) -> dict[str, Any]:
    return tracker.get_stats()
