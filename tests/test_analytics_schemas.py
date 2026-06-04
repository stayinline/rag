"""Tests for Phase 3 analytics schemas."""
import uuid
from datetime import datetime, timezone

import pytest

from app.schemas.analytics import (
    AuditLogResponse,
    AuditLogListResponse,
    FeedbackCreate,
    FeedbackResponse,
    EvalQuestionCreate,
    EvalSetCreate,
    EvalSetResponse,
    EvalSetListResponse,
    EvalRunResponse,
    EvalRunCreate,
    ZeroResultQueryItem,
    ZeroResultQueryResponse,
    LowRatedAnswerItem,
    LowRatedAnswerResponse,
    RAGAnalyticsSummary,
)


class TestFeedbackCreate:
    def test_valid(self):
        data = FeedbackCreate(
            message_id=uuid.uuid4(),
            rating=3,
            reason_tags=["incorrect"],
            comment="Not helpful",
        )
        assert data.rating == 3
        assert "incorrect" in data.reason_tags

    def test_rating_out_of_range(self):
        with pytest.raises(Exception):
            FeedbackCreate(message_id=uuid.uuid4(), rating=0)
        with pytest.raises(Exception):
            FeedbackCreate(message_id=uuid.uuid4(), rating=6)

    def test_minimal(self):
        data = FeedbackCreate(message_id=uuid.uuid4(), rating=1)
        assert data.reason_tags == []
        assert data.comment is None


class TestFeedbackResponse:
    def test_valid(self):
        data = FeedbackResponse(
            id=uuid.uuid4(),
            message_id=uuid.uuid4(),
            trace_id="t1",
            rating=4,
            reason_tags=["missing_source"],
            comment="Could be better",
            created_by=uuid.uuid4(),
            created_at=datetime.now(timezone.utc),
        )
        assert data.rating == 4
        assert data.trace_id == "t1"


class TestEvalQuestionCreate:
    def test_valid(self):
        data = EvalQuestionCreate(
            question="What is the mechanism of EGFR inhibitors?",
            expected_kb_ids=[uuid.uuid4()],
            expected_doc_ids=[uuid.uuid4()],
            expected_answer="EGFR inhibitors block...",
            category="drug",
            difficulty="medium",
        )
        assert data.question.startswith("What")
        assert data.category == "drug"

    def test_minimal(self):
        data = EvalQuestionCreate(question="test")
        assert data.expected_kb_ids == []
        assert data.difficulty is None

    def test_empty_question(self):
        with pytest.raises(Exception):
            EvalQuestionCreate(question="")


class TestEvalSetCreate:
    def test_valid_with_questions(self):
        kb = uuid.uuid4()
        data = EvalSetCreate(
            name="Drug QA Set",
            scenario="qa",
            description="Test drug questions",
            questions=[
                EvalQuestionCreate(question="Q1", expected_kb_ids=[kb]),
                EvalQuestionCreate(question="Q2"),
            ],
        )
        assert len(data.questions) == 2

    def test_name_too_long(self):
        with pytest.raises(Exception):
            EvalSetCreate(name="x" * 256)


class TestEvalSetResponse:
    def test_valid(self):
        data = EvalSetResponse(
            id=uuid.uuid4(),
            org_id=uuid.uuid4(),
            name="Test Set",
            scenario="qa",
            status="active",
            question_count=10,
            created_at=datetime.now(timezone.utc),
        )
        assert data.question_count == 10


class TestEvalSetListResponse:
    def test_valid(self):
        data = EvalSetListResponse(
            items=[
                EvalSetResponse(
                    id=uuid.uuid4(), org_id=uuid.uuid4(), name="Set1",
                    status="active", created_at=datetime.now(timezone.utc),
                ),
            ],
            total=1,
        )
        assert data.total == 1
        assert len(data.items) == 1


class TestEvalRunCreate:
    def test_valid(self):
        data = EvalRunCreate(
            eval_set_id=uuid.uuid4(),
            config={"model": "qwen-plus", "top_k": 10},
        )
        assert data.config["model"] == "qwen-plus"

    def test_minimal(self):
        data = EvalRunCreate(eval_set_id=uuid.uuid4())
        assert data.config == {}


class TestEvalRunResponse:
    def test_valid(self):
        data = EvalRunResponse(
            id=uuid.uuid4(),
            org_id=uuid.uuid4(),
            eval_set_id=uuid.uuid4(),
            status="completed",
            metrics={"recall_at_10": 0.8},
            config={},
            created_at=datetime.now(timezone.utc),
        )
        assert data.status == "completed"
        assert data.metrics["recall_at_10"] == 0.8


class TestZeroResultQueryItem:
    def test_valid(self):
        data = ZeroResultQueryItem(
            query="unknown drug xyz",
            org_id=uuid.uuid4(),
            user_id=uuid.uuid4(),
            kb_ids=[],
            count=5,
            last_seen=datetime.now(timezone.utc),
        )
        assert data.count == 5


class TestZeroResultQueryResponse:
    def test_valid(self):
        data = ZeroResultQueryResponse(items=[], total=0)
        assert data.total == 0


class TestLowRatedAnswerItem:
    def test_valid(self):
        data = LowRatedAnswerItem(
            message_id=uuid.uuid4(),
            rating=1,
            reason_tags=["incorrect", "missing_source"],
            comment="Wrong answer",
            created_at=datetime.now(timezone.utc),
        )
        assert data.rating == 1
        assert len(data.reason_tags) == 2


class TestLowRatedAnswerResponse:
    def test_valid(self):
        data = LowRatedAnswerResponse(items=[], total=0)
        assert data.total == 0


class TestRAGAnalyticsSummary:
    def test_valid(self):
        data = RAGAnalyticsSummary(
            total_queries=1000,
            avg_latency_ms=350.5,
            zero_result_rate=0.05,
            avg_rating=3.8,
            low_rating_rate=0.02,
            avg_retrieved_count=8.5,
            avg_reranked_count=5.0,
        )
        assert data.total_queries == 1000
        assert data.avg_latency_ms == 350.5

    def test_edge_case_all_zeros(self):
        data = RAGAnalyticsSummary(
            total_queries=0,
            avg_latency_ms=0.0,
            zero_result_rate=0.0,
            avg_rating=0.0,
            low_rating_rate=0.0,
            avg_retrieved_count=0.0,
            avg_reranked_count=0.0,
        )
        assert data.total_queries == 0


class TestAuditLogResponse:
    def test_valid(self):
        data = AuditLogResponse(
            id=uuid.uuid4(),
            org_id=uuid.uuid4(),
            user_id=uuid.uuid4(),
            action="chat",
            resource_type="chat",
            details={"query": "test"},
            created_at=datetime.now(timezone.utc),
        )
        assert data.action == "chat"
        assert data.details["query"] == "test"


class TestAuditLogListResponse:
    def test_valid(self):
        data = AuditLogListResponse(items=[], total=0)
        assert data.total == 0
