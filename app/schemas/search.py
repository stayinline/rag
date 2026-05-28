from uuid import UUID

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=5000)
    kb_ids: list[UUID] = Field(default_factory=list)
    top_k: int = Field(default=10, ge=1, le=50)
    filters: dict[str, str] = Field(default_factory=dict)


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


class SearchResponse(BaseModel):
    query: str
    total: int
    results: list[SearchResultItem]
