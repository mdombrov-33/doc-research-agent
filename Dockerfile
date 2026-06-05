FROM python:3.13-slim AS base

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_SYSTEM_PYTHON=1

RUN pip install uv

FROM base AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    g++ \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock* ./
RUN uv sync --frozen --no-install-project

FROM base AS runtime

COPY --from=builder /app/.venv /app/.venv
COPY . .

ENV PATH="/app/.venv/bin:$PATH"

# Bake the reranker model into the image so the first request skips a ~9s download.
# Not the fastembed default (/tmp): Cloud Run mounts a fresh tmpfs over /tmp at runtime.
ENV FASTEMBED_CACHE_PATH=/app/.model_cache
RUN python -c "from src.config import get_settings; from src.core.retrieval.reranker import _get_cross_encoder; _get_cross_encoder(get_settings().RERANK_MODEL)"

ARG GIT_SHA=unknown
ARG APP_VERSION=unknown
ENV GIT_SHA=${GIT_SHA} \
    APP_VERSION=${APP_VERSION}

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
