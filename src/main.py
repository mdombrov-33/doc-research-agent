from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request, Response
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from src.api.middleware import RequestLoggingMiddleware
from src.api.rate_limit import limiter
from src.api.routes import router
from src.api.schemas import HealthResponse
from src.config import Settings, get_settings
from src.core import guardrails
from src.core.agent.checkpointer import open_checkpointer
from src.core.agent.graph import build_graph
from src.core.monitoring.tracker import MetricsTracker
from src.core.nlp import get_spacy_model
from src.core.retrieval import rerank
from src.core.tracing import setup_tracing
from src.core.vectorstore import ensure_collection_exists, get_vector_store
from src.utils.logger import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("startup")
    settings = get_settings()
    setup_tracing(settings.OTEL_ENDPOINT)
    HTTPXClientInstrumentor().instrument()
    ensure_collection_exists()

    # Build expensive, process-wide resources once and stash them on app.state;
    # request handlers read them back via the providers in api/dependencies.py.
    async with open_checkpointer(settings) as checkpointer:
        app.state.vector_store = get_vector_store()
        app.state.nlp = get_spacy_model()
        app.state.agent = build_graph(checkpointer)
        guardrails.warmup()
        rerank.warmup()
        app.state.metrics_tracker = MetricsTracker.from_settings(settings)

        logger.info("startup_complete")
        try:
            yield
        finally:
            app.state.metrics_tracker.close()
            logger.info("shutdown")


app = FastAPI(
    title="Document Research Agent",
    description="Agentic document research assistant using LangGraph",
    version="1.0.0",
    lifespan=lifespan,
)
FastAPIInstrumentor.instrument_app(app)


def _on_rate_limit(request: Request, exc: Exception) -> Response:
    # slowapi types its handler's exc as RateLimitExceeded, narrower than the base Exception
    # Starlette's add_exception_handler expects. Adapt the signature instead of suppressing it;
    # Starlette only routes RateLimitExceeded here, so the assert always holds at runtime.
    assert isinstance(exc, RateLimitExceeded)
    return _rate_limit_exceeded_handler(request, exc)


app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _on_rate_limit)
app.add_middleware(RequestLoggingMiddleware)
app.include_router(router, prefix="/api", tags=["documents"])


@app.get("/health", response_model=HealthResponse)
async def health_check(settings: Settings = Depends(get_settings)):
    return {
        "status": "healthy",
        "environment": settings.APP_ENV,
        "llm_model": settings.LLM_MODEL,
    }
