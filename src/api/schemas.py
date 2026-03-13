import uuid

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources_count: int
    session_id: str


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    chunks_created: int
    file_size: int


class HealthResponse(BaseModel):
    status: str
    environment: str
    llm_provider: str
