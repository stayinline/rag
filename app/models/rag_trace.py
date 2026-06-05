from sqlalchemy import Column, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID

from app.models.base import Base, TimestampMixin, UUIDMixin


class RAGTraceDetail(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "rag_trace_details"

    org_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    user_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    trace_id = Column(String(100), nullable=False, unique=True, index=True)
    conversation_id = Column(UUID(as_uuid=True), nullable=True, index=True)
    message_id = Column(UUID(as_uuid=True), nullable=True, index=True)
    query = Column(Text, nullable=False)
    answer_preview = Column(Text, nullable=False, default="")
    answer_length = Column(Integer, nullable=False, default=0)
    kb_ids = Column(JSONB, default=list)
    steps = Column(JSONB, default=list)
    total_latency_ms = Column(Integer, nullable=False, default=0)
    model = Column(String(100), nullable=True)
    prompt_version = Column(String(50), nullable=True)
