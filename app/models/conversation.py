from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID

from app.models.base import Base, TimestampMixin, UUIDMixin


class Conversation(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "conversations"

    org_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    user_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    title = Column(String(255), nullable=False)
    kb_ids = Column(JSONB, default=list)
    message_count = Column(Integer, default=0)
    last_message_at = Column(DateTime(timezone=True), index=True)
    deleted_at = Column(DateTime(timezone=True), index=True)


class ConversationMessage(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "conversation_messages"

    org_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    conversation_id = Column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    user_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    role = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    sequence = Column(Integer, nullable=False)
    trace_id = Column(String(100), index=True)
    sources = Column(JSONB, default=list)
    kb_ids = Column(JSONB, default=list)
    model = Column(String(100))
    prompt_version = Column(String(50))
