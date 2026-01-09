from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., description="User question to answer", min_length=1)


class QueryResponse(BaseModel):
    question: str = Field(..., description="Original question")
    answer: str = Field(..., description="Generated answer")
    sources_count: int = Field(..., description="Number of documents used")


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    chunks_created: int
    file_size: int
