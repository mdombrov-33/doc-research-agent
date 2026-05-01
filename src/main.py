import time
import uuid
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from starlette.types import ASGIApp, Receive, Scope, Send

from src.api.routes import router
from src.api.schemas import HealthResponse
from src.config import get_settings
from src.core.vector_store import ensure_collection_exists
from src.utils.logger import logger

settings = get_settings()


class RequestLoggingMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app
        self._logger = structlog.get_logger("doc-research-agent")

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        raw_headers: dict[bytes, bytes] = dict(scope.get("headers", []))
        request_id = raw_headers.get(b"x-request-id", b"").decode() or str(uuid.uuid4())

        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)

        start = time.monotonic()
        status_code = 200

        async def send_wrapper(message: dict) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                headers = list(message.get("headers", []))
                headers.append((b"x-request-id", request_id.encode()))
                message = {**message, "headers": headers}
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            duration_ms = round((time.monotonic() - start) * 1000, 1)
            self._logger.info(
                "served",
                method=scope.get("method", ""),
                path=scope.get("path", ""),
                status=status_code,
                duration_ms=duration_ms,
            )
            structlog.contextvars.clear_contextvars()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("startup")
    ensure_collection_exists()
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
async def health_check():
    return {
        "status": "healthy",
        "environment": settings.APP_ENV,
        "llm_provider": settings.LLM_PROVIDER,
    }
