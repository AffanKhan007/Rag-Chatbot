from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    document_ids: list[int] | None = None
    vector_top_k: int | None = Field(default=None, ge=1, le=50)
    keyword_top_k: int | None = Field(default=None, ge=1, le=50)
    final_top_k: int | None = Field(default=None, ge=1, le=10)
    debug: bool = False


class AskDocsResponse(BaseModel):
    question: str
    answer: str
    filenames: list[str]
    found_in_documents: bool
    mode: str
    groq_error: str | None = None
    debug: dict | None = None
