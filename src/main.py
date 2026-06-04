from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI

from src.api.middleware import RequestLoggingMiddleware
from src.api.routes import router
from src.api.schemas import HealthResponse
from src.config import Settings, get_settings
from src.core.agent import build_graph
from src.core.document_processing.text_processor import get_spacy_model
from src.core.monitoring.tracker import MetricsTracker
from src.core.retrieval.search import get_vector_store
from src.core.vector_store import ensure_collection_exists
from src.guardrails.guardrails_wrapper import GuardrailsWrapper
from src.utils.logger import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("startup")
    settings = get_settings()
    ensure_collection_exists()

    # Build expensive, process-wide resources once and stash them on app.state;
    # request handlers read them back via the providers in api/dependencies.py.
    app.state.vector_store = get_vector_store()
    app.state.nlp = get_spacy_model()
    app.state.agent = build_graph()
    app.state.guardrails = GuardrailsWrapper()
    app.state.metrics_tracker = MetricsTracker(db_path=settings.METRICS_DB_PATH)

    logger.info("startup_complete")
    yield
    logger.info("shutdown")


app = FastAPI(
    title="Document Research Agent",
    description="Agentic document research assistant using LangGraph",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(RequestLoggingMiddleware)
app.include_router(router, prefix="/api", tags=["documents"])


@app.get("/health", response_model=HealthResponse)
async def health_check(settings: Settings = Depends(get_settings)):
    return {
        "status": "healthy",
        "environment": settings.APP_ENV,
        "llm_model": settings.LLM_MODEL,
    }
