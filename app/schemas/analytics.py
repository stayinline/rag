from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


# --- Audit Log ---

class AuditLogResponse(BaseModel):
    id: UUID
    org_id: UUID
    user_id: UUID
    action: str
    resource_type: str | None = None
    resource_id: UUID | None = None
    details: dict = Field(default_factory=dict)
    ip_address: str | None = None
    status_code: int | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


class AuditLogListResponse(BaseModel):
    items: list[AuditLogResponse]
    total: int


# --- Feedback ---

class FeedbackCreate(BaseModel):
    message_id: UUID
    rating: int = Field(..., ge=1, le=5)
    reason_tags: list[str] = Field(default_factory=list)
    comment: str | None = None


class FeedbackResponse(BaseModel):
    id: UUID
    message_id: UUID
    trace_id: str | None
    rating: int
    reason_tags: list[str]
    comment: str | None
    created_by: UUID
    created_at: datetime

    model_config = {"from_attributes": True}


# --- Evaluation Sets ---

class EvalQuestionCreate(BaseModel):
    question: str = Field(..., min_length=1)
    expected_kb_ids: list[UUID] = Field(default_factory=list)
    expected_doc_ids: list[UUID] = Field(default_factory=list)
    expected_answer: str | None = None
    category: str | None = None
    difficulty: str | None = None


class EvalSetCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    scenario: str | None = None
    description: str | None = None
    dataset_path: str | None = None
    questions: list[EvalQuestionCreate] = Field(default_factory=list)


class EvalQuestionResponse(BaseModel):
    id: UUID
    question: str
    expected_kb_ids: list[UUID]
    expected_doc_ids: list[UUID]
    expected_answer: str | None
    category: str | None
    difficulty: str | None
    created_at: datetime

    model_config = {"from_attributes": True}


class EvalSetResponse(BaseModel):
    id: UUID
    org_id: UUID
    name: str
    scenario: str | None = None
    description: str | None = None
    dataset_path: str | None = None
    status: str
    question_count: int = 0
    created_at: datetime

    model_config = {"from_attributes": True}


class EvalSetListResponse(BaseModel):
    items: list[EvalSetResponse]
    total: int


# --- Evaluation Runs ---

class EvalRunResponse(BaseModel):
    id: UUID
    org_id: UUID
    eval_set_id: UUID
    status: str
    metrics: dict = Field(default_factory=dict)
    config: dict = Field(default_factory=dict)
    error_message: str | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


class EvalRunCreate(BaseModel):
    eval_set_id: UUID
    config: dict = Field(default_factory=dict)


# --- Analytics ---

class ZeroResultQueryItem(BaseModel):
    query: str
    org_id: UUID
    user_id: UUID
    kb_ids: list[UUID]
    count: int
    last_seen: datetime


class ZeroResultQueryResponse(BaseModel):
    items: list[ZeroResultQueryItem]
    total: int


class LowRatedAnswerItem(BaseModel):
    message_id: UUID
    rating: int
    reason_tags: list[str]
    comment: str | None
    created_at: datetime | None = None


class LowRatedAnswerResponse(BaseModel):
    items: list[LowRatedAnswerItem]
    total: int


class FeedbackWeightsResponse(BaseModel):
    sample_count: int
    chunk_weights: dict[str, float] = Field(default_factory=dict)
    document_weights: dict[str, float] = Field(default_factory=dict)


class RAGAnalyticsSummary(BaseModel):
    total_queries: int
    avg_latency_ms: float
    zero_result_rate: float
    avg_rating: float
    low_rating_rate: float
    avg_retrieved_count: float
    avg_reranked_count: float


class RAGTraceStepResponse(BaseModel):
    step: str
    duration_ms: float = 0.0
    started_at: str | None = None
    details: dict = Field(default_factory=dict)


class RAGTraceDetailResponse(BaseModel):
    id: UUID
    trace_id: str
    conversation_id: UUID | None = None
    message_id: UUID | None = None
    query: str
    answer_preview: str
    answer_length: int
    kb_ids: list[str] = Field(default_factory=list)
    steps: list[RAGTraceStepResponse] = Field(default_factory=list)
    total_latency_ms: int
    model: str | None = None
    prompt_version: str | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


class RAGTraceListResponse(BaseModel):
    items: list[RAGTraceDetailResponse]
    total: int
