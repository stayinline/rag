from sqlalchemy import Column, String, Text, Integer, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID, JSONB

from app.models.base import Base, TimestampMixin, UUIDMixin


class DocumentChunk(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "document_chunks"
    __table_args__ = (UniqueConstraint("document_version_id", "chunk_index"),)

    org_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    kb_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    document_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    document_version_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    chunk_index = Column(Integer, nullable=False)
    parent_chunk_id = Column(UUID(as_uuid=True))
    weaviate_id = Column(String(100))
    content_preview = Column(Text)
    token_count = Column(Integer)
    page_start = Column(Integer)
    page_end = Column(Integer)
    section_path = Column(Text)
    source_locator = Column(JSONB, default=dict)
    acl_hash = Column(String(64))
