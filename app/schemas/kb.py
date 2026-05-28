from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class KBCreate(BaseModel):
    name: str = Field(..., max_length=200)
    description: str = ""


class KBUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    is_active: bool | None = None


class KBResponse(BaseModel):
    id: UUID
    org_id: UUID
    name: str
    description: str
    is_active: bool
    created_by: UUID
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class KBListResponse(BaseModel):
    items: list[KBResponse]
    total: int
