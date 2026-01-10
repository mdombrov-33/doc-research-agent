from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.routes import router
from src.api.schemas import HealthResponse
from src.config import get_settings
from src.core.vector_store import ensure_collection_exists
from src.utils.logger import logger

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up application")
    ensure_collection_exists()
    logger.info("Application startup complete")
    yield
    logger.info("Shutting down application")


app = FastAPI(
    title="Document Research Agent",
    description="Agentic document research assistant using LangGraph",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(router, prefix="/api", tags=["documents"])


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return {
        "status": "healthy",
        "environment": settings.APP_ENV,
        "llm_provider": settings.LLM_PROVIDER,
    }
