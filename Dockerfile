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

RUN uv pip install --python /app/.venv/bin/python https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
