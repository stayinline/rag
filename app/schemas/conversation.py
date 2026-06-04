from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field

from app.schemas.chat import ChatSource


class ConversationListItem(BaseModel):
    id: UUID
    title: str
    kb_ids: list[UUID] = Field(default_factory=list)
    message_count: int
    last_message_at: datetime | None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ConversationListResponse(BaseModel):
    items: list[ConversationListItem]
    total: int


class ConversationMessageResponse(BaseModel):
    id: UUID
    conversation_id: UUID
    role: str
    content: str
    sequence: int
    trace_id: str | None = None
    sources: list[ChatSource] = Field(default_factory=list)
    kb_ids: list[UUID] = Field(default_factory=list)
    model: str | None = None
    prompt_version: str | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


class ConversationDetailResponse(BaseModel):
    id: UUID
    title: str
    kb_ids: list[UUID] = Field(default_factory=list)
    message_count: int
    last_message_at: datetime | None
    created_at: datetime
    updated_at: datetime
    messages: list[ConversationMessageResponse] = Field(default_factory=list)

    model_config = {"from_attributes": True}
