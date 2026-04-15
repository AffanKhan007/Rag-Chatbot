from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    vector_top_k: int | None = Field(default=None, ge=1, le=50)
    keyword_top_k: int | None = Field(default=None, ge=1, le=50)
    final_top_k: int | None = Field(default=None, ge=1, le=10)
