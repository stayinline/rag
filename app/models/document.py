from sqlalchemy import Column, String, Text, Integer, DateTime, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID, JSONB

from app.models.base import Base, TimestampMixin, UUIDMixin


class Document(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "documents"

    org_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    kb_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    title = Column(String(500), nullable=False)
    file_name = Column(String(500))
    file_type = Column(String(50))
    source_type = Column(String(50), default="upload")
    source_uri = Column(Text)
    current_version = Column(Integer, default=1)
    status = Column(String(50), default="draft")
    security_level = Column(String(50), default="internal")
    content_hash = Column(String(64))
    metadata_ = Column("metadata", JSONB, default=dict)
    created_by = Column(UUID(as_uuid=True), nullable=False)
    document_type = Column(String(50), default="general")  # general, paper, sop, guideline
    deleted_at = Column(DateTime(timezone=True))


class DocumentVersion(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "document_versions"
    __table_args__ = (UniqueConstraint("document_id", "version"),)

    org_id = Column(UUID(as_uuid=True), nullable=False)
    document_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    version = Column(Integer, nullable=False)
    storage_path = Column(Text, nullable=False)
    parsed_path = Column(Text)
    content_hash = Column(String(64))
    parser_version = Column(String(50))
    chunker_version = Column(String(50))
    embedding_model = Column(String(100))
    index_status = Column(String(50), default="pending")
    chunk_count = Column(Integer, default=0)
    quality_score = Column(Integer)
