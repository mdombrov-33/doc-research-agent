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

ENV PATH="/app/.venv/bin:$PATH"

# Bake the reranker model into the image so the first request skips a ~9s download.
# Not the fastembed default (/tmp): Cloud Run mounts a fresh tmpfs over /tmp at runtime.
# Done before `COPY . .` and with the model name inline (not imported from src) so this
# layer — and the HuggingFace download — stays cached across code changes. It only re-runs
# when the venv or this model name changes. Keep in sync with config.RERANK_MODEL.
ENV FASTEMBED_CACHE_PATH=/app/.model_cache
RUN python -c "from fastembed.rerank.cross_encoder import TextCrossEncoder; TextCrossEncoder(model_name='Xenova/ms-marco-MiniLM-L-6-v2')"

# Bake llm-guard's Toxicity model so the first request skips a cold-start download. Same
# rationale as the reranker bake above: the scanner name is inlined (not imported from src)
# so this layer — and the download — stays
# cached across code changes. HF_HOME is set here so it also applies at runtime, where
# the models must be found. Keep in sync with src/core/guardrails.py.
ENV HF_HOME=/app/.hf_cache
RUN python -c "from llm_guard.input_scanners import Toxicity; Toxicity()"

COPY . .

ARG GIT_SHA=unknown
ARG APP_VERSION=unknown
ENV GIT_SHA=${GIT_SHA} \
    APP_VERSION=${APP_VERSION}

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
