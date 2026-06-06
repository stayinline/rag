from uuid import UUID

from pydantic import BaseModel, Field, model_validator


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=5000)
    kb_ids: list[UUID] = Field(default_factory=list)
    top_k: int = Field(default=10, ge=1, le=50)
    limit: int | None = Field(default=None, ge=1, le=50, exclude=True)
    filters: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def apply_limit_alias(self) -> "SearchRequest":
        if self.limit is not None:
            self.top_k = self.limit
        return self


class SearchResultItem(BaseModel):
    chunk_id: str
    document_id: UUID
    document_title: str
    section_path: str | None
    page_start: int | None
    page_end: int | None
    content_preview: str
    vector_score: float
    bm25_score: float
    combined_score: float
    metadata_score: float = 0.0
    feedback_score: float = 0.0
    hybrid_score: float = 0.0


class SearchResponse(BaseModel):
    query: str
    total: int
    results: list[SearchResultItem]
