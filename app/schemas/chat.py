from uuid import UUID

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=5000)
    kb_ids: list[UUID] = Field(default_factory=list, description="Target knowledge base IDs")
    conversation_id: UUID | None = None
    stream: bool = True


class ChatSource(BaseModel):
    chunk_id: str
    document_id: str
    document_title: str
    section_path: str | None
    page_start: int | None
    page_end: int | None
    score: float
    content_preview: str


class ChatResponse(BaseModel):
    answer: str
    trace_id: str
    message_id: UUID | None = Field(
        default=None,
        description="Answer identifier for feedback; equals trace_id when present",
    )
    conversation_id: UUID | None
    sources: list[ChatSource]
    model: str
    prompt_version: str


class ChatStreamChunk(BaseModel):
    delta: str = ""
    done: bool = False
    trace_id: str | None = None
    message_id: UUID | None = None
    sources: list[ChatSource] = []
    conversation_id: UUID | None = None
