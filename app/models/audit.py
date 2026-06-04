from sqlalchemy import Column, String, Text, Integer
from sqlalchemy.dialects.postgresql import UUID, JSONB

from app.models.base import Base, TimestampMixin, UUIDMixin


class AuditLog(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "audit_logs"

    org_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    user_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    action = Column(String(100), nullable=False, index=True)  # login, upload, delete, search, chat, etc.
    resource_type = Column(String(50))  # document, kb, paper, chat, search
    resource_id = Column(UUID(as_uuid=True))
    details = Column(JSONB, default=dict)  # action-specific metadata
    ip_address = Column(String(50))
    user_agent = Column(String(500))
    status_code = Column(Integer)  # HTTP status code


class AnswerFeedback(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "answer_feedback"

    org_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    message_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    trace_id = Column(String(100))
    rating = Column(Integer)  # 1-5
    reason_tags = Column(JSONB, default=list)  # ["incorrect", "missing_source", "outdated", "unsafe"]
    comment = Column(Text)
    created_by = Column(UUID(as_uuid=True), nullable=False, index=True)


class EvaluationSet(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "evaluation_sets"

    org_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    scenario = Column(String(50))  # qa, search_summary, paper_extract
    description = Column(Text)
    dataset_path = Column(Text, nullable=False)  # path to JSON/CSV dataset
    status = Column(String(20), default="active")  # active, archived


class EvaluationRun(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "evaluation_runs"

    org_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    eval_set_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    status = Column(String(20), default="pending")  # pending, running, completed, failed
    metrics = Column(JSONB, default=dict)  # computed metrics
    config = Column(JSONB, default=dict)  # run configuration (model, prompt_version, etc.)
    error_message = Column(Text)


class EvaluationQuestion(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "evaluation_questions"

    org_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    eval_set_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    question = Column(Text, nullable=False)
    expected_kb_ids = Column(JSONB, default=list)  # expected knowledge base IDs
    expected_doc_ids = Column(JSONB, default=list)  # expected document IDs
    expected_answer = Column(Text)  # optional reference answer
    category = Column(String(50))  # drug, disease, guideline, paper, sop, regulation
    difficulty = Column(String(20))  # easy, medium, hard
