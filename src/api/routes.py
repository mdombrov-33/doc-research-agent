from typing import Any

from fastapi import APIRouter, Depends, File, UploadFile
from langchain_qdrant import QdrantVectorStore
from spacy.language import Language

from src.api.dependencies import (
    Settings,
    get_agent,
    get_eval_tracker,
    get_guardrails,
    get_nlp,
    get_settings,
    get_vector_store,
)
from src.api.handlers.stream import handle_stream
from src.api.handlers.upload import handle_upload
from src.api.schemas import QueryRequest, UploadResponse
from src.core.evaluation.metrics import EvaluationTracker
from src.guardrails.guardrails_wrapper import GuardrailsWrapper

router = APIRouter()


@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    settings: Settings = Depends(get_settings),
    vector_store: QdrantVectorStore = Depends(get_vector_store),
    nlp: Language = Depends(get_nlp),
):
    result = await handle_upload(file, settings, vector_store, nlp)
    return UploadResponse(**result)


@router.post("/stream")
async def stream_query(
    payload: QueryRequest,
    guardrails: GuardrailsWrapper = Depends(get_guardrails),
    agent: Any = Depends(get_agent),
    tracker: EvaluationTracker = Depends(get_eval_tracker),
):
    return await handle_stream(payload, guardrails, agent, tracker)


@router.get("/evaluation/stats")
async def get_evaluation_stats(
    tracker: EvaluationTracker = Depends(get_eval_tracker),
) -> dict[str, Any]:
    return tracker.get_stats()
