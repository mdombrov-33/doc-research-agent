import uuid

from pydantic import BaseModel, Field, field_validator

from src.config import SUPPORTED_MODELS


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    model: str | None = None
    top_k: int = Field(default=5, ge=1, le=20)

    @field_validator("model")
    @classmethod
    def validate_model(cls, model: str | None) -> str | None:
        if model is not None and model not in SUPPORTED_MODELS:
            raise ValueError("Unsupported model")
        return model


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    chunks_created: int
    file_size: int


class HealthResponse(BaseModel):
    status: str
    environment: str
    llm_model: str
