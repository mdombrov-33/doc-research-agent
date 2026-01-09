from fastapi import FastAPI
from pydantic import BaseModel

from src.config import get_settings

settings = get_settings()

app = FastAPI(
    title="Document Research Agent",
    description="Agentic document research assistant using LangGraph",
    version="0.1.0",
)


class HealthResponse(BaseModel):
    status: str
    environment: str
    llm_provider: str


@app.get("/")
async def root():
    return {"message": "Document Research Agent API"}


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return {
        "status": "healthy",
        "environment": settings.APP_ENV,
        "llm_provider": settings.LLM_PROVIDER,
    }
