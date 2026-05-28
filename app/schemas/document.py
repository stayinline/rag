from datetime import datetime
from uuid import UUID

from pydantic import BaseModel


class DocumentVersionInfo(BaseModel):
    id: UUID
    version: int
    index_status: str
    chunk_count: int
    storage_path: str
    created_at: datetime

    model_config = {"from_attributes": True}


class DocumentResponse(BaseModel):
    id: UUID
    org_id: UUID
    kb_id: UUID
    title: str
    file_name: str | None
    file_type: str | None
    source_type: str
    current_version: int
    status: str
    security_level: str
    created_by: UUID
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class DocumentListResponse(BaseModel):
    items: list[DocumentResponse]
    total: int


class IngestionJobResponse(BaseModel):
    id: UUID
    document_id: UUID
    version_id: UUID
    job_type: str
    status: str
    retry_count: int
    error_code: str | None
    error_message: str | None
    started_at: datetime | None
    finished_at: datetime | None
    created_at: datetime

    model_config = {"from_attributes": True}
