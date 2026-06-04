"""Phase 3 API: Feedback, Evaluation, Analytics, Audit Logs."""
import uuid
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.database import get_db
from app.models.audit import AnswerFeedback, EvaluationSet, EvaluationRun, EvaluationQuestion
from app.models.conversation import ConversationMessage
from app.schemas.analytics import (
    FeedbackCreate,
    FeedbackResponse,
    EvalSetCreate,
    EvalSetResponse,
    EvalSetListResponse,
    EvalRunCreate,
    EvalRunResponse,
    ZeroResultQueryResponse,
    ZeroResultQueryItem,
    LowRatedAnswerResponse,
    LowRatedAnswerItem,
    RAGAnalyticsSummary,
    AuditLogListResponse,
    AuditLogResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["analytics"])


# --- Feedback ---

@router.post("/answers/{message_id}/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    message_id: uuid.UUID,
    data: FeedbackCreate,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Submit feedback for an answer."""
    org_id = str(user["org_id"])
    logger.info(
        "Submit feedback request org_id=%s user_id=%s message_id=%s rating=%s reason_count=%s",
        org_id,
        user["user_id"],
        message_id,
        data.rating,
        len(data.reason_tags or []),
    )
    message_result = await db.execute(
        select(ConversationMessage).where(
            ConversationMessage.id == message_id,
            ConversationMessage.org_id == user["org_id"],
            ConversationMessage.user_id == user["user_id"],
            ConversationMessage.role == "assistant",
        )
    )
    message = message_result.scalar_one_or_none()
    trace_id = message.trace_id if message and message.trace_id else str(message_id)

    feedback = AnswerFeedback(
        id=uuid.uuid4(),
        org_id=org_id,
        message_id=message_id,
        trace_id=trace_id,
        rating=data.rating,
        reason_tags=data.reason_tags,
        comment=data.comment,
        created_by=user["user_id"],
        created_at=datetime.now(timezone.utc),
    )
    db.add(feedback)
    await db.commit()

    # Write audit log
    from app.services.audit import write_audit_log
    await write_audit_log(
        org_id=org_id,
        user_id=str(user["user_id"]),
        action="submit_feedback",
        resource_type="feedback",
        resource_id=str(message_id),
        details={"rating": data.rating, "reason_tags": data.reason_tags},
    )

    logger.info("Submit feedback succeeded org_id=%s user_id=%s message_id=%s feedback_id=%s", org_id, user["user_id"], message_id, feedback.id)
    return feedback


# --- Evaluation Sets ---

@router.post("/evaluation-sets", response_model=EvalSetResponse)
async def create_eval_set(
    data: EvalSetCreate,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Create an evaluation set with questions."""
    org_id = str(user["org_id"])
    logger.info(
        "Create evaluation set request org_id=%s user_id=%s name=%s question_count=%s",
        org_id,
        user["user_id"],
        data.name,
        len(data.questions),
    )

    eval_set = EvaluationSet(
        id=uuid.uuid4(),
        org_id=org_id,
        name=data.name,
        scenario=data.scenario,
        description=data.description,
        dataset_path=data.dataset_path or "",
        status="active",
        created_at=datetime.now(timezone.utc),
    )
    db.add(eval_set)

    # Create questions
    for q in data.questions:
        question = EvaluationQuestion(
            id=uuid.uuid4(),
            org_id=org_id,
            eval_set_id=eval_set.id,
            question=q.question,
            expected_kb_ids=[str(kb) for kb in q.expected_kb_ids],
            expected_doc_ids=[str(did) for did in q.expected_doc_ids],
            expected_answer=q.expected_answer,
            category=q.category,
            difficulty=q.difficulty,
        )
        db.add(question)

    await db.commit()
    logger.info("Create evaluation set succeeded org_id=%s user_id=%s eval_set_id=%s", org_id, user["user_id"], eval_set.id)
    return eval_set


@router.get("/evaluation-sets", response_model=EvalSetListResponse)
async def list_eval_sets(
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    """List evaluation sets for the org."""
    org_id = str(user["org_id"])
    logger.info("List evaluation sets request org_id=%s user_id=%s limit=%s offset=%s", org_id, user["user_id"], limit, offset)

    stmt = select(EvaluationSet).where(
        EvaluationSet.org_id == org_id,
        EvaluationSet.status == "active",
    )
    count_stmt = select(func.count()).select_from(stmt.subquery())
    total = (await db.execute(count_stmt)).scalar() or 0

    stmt = stmt.order_by(EvaluationSet.created_at.desc()).limit(limit).offset(offset)
    result = await db.execute(stmt)
    items = list(result.scalars().all())

    # Get question counts
    for item in items:
        q_stmt = select(func.count()).where(EvaluationQuestion.eval_set_id == item.id)
        item.question_count = (await db.execute(q_stmt)).scalar() or 0

    logger.info("List evaluation sets complete org_id=%s user_id=%s total=%s returned=%s", org_id, user["user_id"], total, len(items))
    return EvalSetListResponse(items=items, total=total)


@router.get("/evaluation-sets/{eval_set_id}", response_model=EvalSetResponse)
async def get_eval_set(
    eval_set_id: uuid.UUID,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get an evaluation set by ID."""
    org_id = str(user["org_id"])
    logger.info("Get evaluation set request org_id=%s user_id=%s eval_set_id=%s", org_id, user["user_id"], eval_set_id)
    eval_set = await db.get(EvaluationSet, eval_set_id)
    if not eval_set or str(eval_set.org_id) != org_id:
        logger.warning("Get evaluation set failed org_id=%s user_id=%s eval_set_id=%s reason=not_found", org_id, user["user_id"], eval_set_id)
        raise HTTPException(status_code=404, detail="Evaluation set not found")

    q_stmt = select(func.count()).where(EvaluationQuestion.eval_set_id == eval_set_id)
    eval_set.question_count = (await db.execute(q_stmt)).scalar() or 0
    logger.info("Get evaluation set succeeded org_id=%s user_id=%s eval_set_id=%s question_count=%s", org_id, user["user_id"], eval_set_id, eval_set.question_count)
    return eval_set


@router.get("/evaluation-sets/{eval_set_id}/questions")
async def list_eval_questions(
    eval_set_id: uuid.UUID,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List questions in an evaluation set."""
    org_id = str(user["org_id"])
    logger.info("List evaluation questions request org_id=%s user_id=%s eval_set_id=%s", org_id, user["user_id"], eval_set_id)

    # Verify eval set belongs to org
    eval_set = await db.get(EvaluationSet, eval_set_id)
    if not eval_set or str(eval_set.org_id) != org_id:
        logger.warning("List evaluation questions failed org_id=%s user_id=%s eval_set_id=%s reason=eval_set_not_found", org_id, user["user_id"], eval_set_id)
        raise HTTPException(status_code=404, detail="Evaluation set not found")

    stmt = (
        select(EvaluationQuestion)
        .where(EvaluationQuestion.eval_set_id == eval_set_id)
        .order_by(EvaluationQuestion.created_at.asc())
    )
    result = await db.execute(stmt)
    questions = list(result.scalars().all())
    logger.info("List evaluation questions complete org_id=%s user_id=%s eval_set_id=%s returned=%s", org_id, user["user_id"], eval_set_id, len(questions))
    return questions


# --- Evaluation Runs ---

@router.post("/evaluations/run", response_model=EvalRunResponse)
async def run_evaluation(
    data: EvalRunCreate,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Run an evaluation against an evaluation set."""
    org_id = str(user["org_id"])
    logger.info("Run evaluation request org_id=%s user_id=%s eval_set_id=%s", org_id, user["user_id"], data.eval_set_id)

    # Verify eval set belongs to org
    eval_set = await db.get(EvaluationSet, data.eval_set_id)
    if not eval_set or str(eval_set.org_id) != org_id:
        logger.warning("Run evaluation failed org_id=%s user_id=%s eval_set_id=%s reason=eval_set_not_found", org_id, user["user_id"], data.eval_set_id)
        raise HTTPException(status_code=404, detail="Evaluation set not found")

    # Create evaluation run
    run = EvaluationRun(
        id=uuid.uuid4(),
        org_id=org_id,
        eval_set_id=data.eval_set_id,
        status="pending",
        config=data.config,
        metrics={},
        created_at=datetime.now(timezone.utc),
    )
    db.add(run)
    await db.commit()

    # Kick off Celery evaluation task
    from app.workers.celery_app import celery_app
    celery_app.send_task(
        "run_evaluation",
        args=[org_id, str(run.id), str(data.eval_set_id), data.config],
    )
    logger.info("Run evaluation task queued org_id=%s user_id=%s eval_set_id=%s run_id=%s", org_id, user["user_id"], data.eval_set_id, run.id)

    # Audit log
    from app.services.audit import write_audit_log
    await write_audit_log(
        org_id=org_id,
        user_id=str(user["user_id"]),
        action="run_evaluation",
        resource_type="evaluation",
        resource_id=str(run.id),
        details={"eval_set_id": str(data.eval_set_id), "config": data.config},
    )

    logger.info("Run evaluation request accepted org_id=%s user_id=%s run_id=%s", org_id, user["user_id"], run.id)
    return run


@router.get("/evaluations/{run_id}", response_model=EvalRunResponse)
async def get_evaluation_run(
    run_id: uuid.UUID,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get evaluation run status and results."""
    org_id = str(user["org_id"])
    logger.info("Get evaluation run request org_id=%s user_id=%s run_id=%s", org_id, user["user_id"], run_id)
    run = await db.get(EvaluationRun, run_id)
    if not run or str(run.org_id) != org_id:
        logger.warning("Get evaluation run failed org_id=%s user_id=%s run_id=%s reason=not_found", org_id, user["user_id"], run_id)
        raise HTTPException(status_code=404, detail="Evaluation run not found")
    logger.info("Get evaluation run succeeded org_id=%s user_id=%s run_id=%s status=%s", org_id, user["user_id"], run_id, run.status)
    return run


# --- Analytics ---

@router.get("/analytics/zero-result-queries", response_model=ZeroResultQueryResponse)
async def get_zero_result_queries(
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    limit: int = Query(50, ge=1, le=200),
):
    """Get zero-result queries for the org."""
    org_id = str(user["org_id"])
    logger.info("Get zero-result queries request org_id=%s user_id=%s limit=%s", org_id, user["user_id"], limit)
    from app.services.clickhouse import clickhouse_client
    rows = await clickhouse_client.get_zero_result_queries(org_id, limit)

    items = []
    for row in rows:
        try:
            kb_ids = [uuid.UUID(kb.strip().strip("'")) for kb in (row.get("kb_ids") or "").split(",") if kb.strip()]
        except Exception:
            kb_ids = []
        items.append(ZeroResultQueryItem(
            query=row.get("query_text", ""),
            org_id=uuid.UUID(row.get("org_id", org_id)),
            user_id=uuid.UUID(row.get("user_id", "00000000-0000-0000-0000-000000000000")),
            kb_ids=kb_ids,
            count=row.get("cnt", 0),
            last_seen=row.get("last_seen", ""),
        ))

    logger.info("Get zero-result queries complete org_id=%s user_id=%s returned=%s", org_id, user["user_id"], len(items))
    return ZeroResultQueryResponse(items=items, total=len(items))


@router.get("/analytics/low-rated-answers", response_model=LowRatedAnswerResponse)
async def get_low_rated_answers(
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    limit: int = Query(50, ge=1, le=200),
):
    """Get low-rated answers for the org."""
    org_id = str(user["org_id"])
    logger.info("Get low-rated answers request org_id=%s user_id=%s limit=%s", org_id, user["user_id"], limit)
    from app.services.clickhouse import clickhouse_client
    rows = await clickhouse_client.get_low_rated_answers(org_id, limit)

    items = []
    for row in rows:
        # Also fetch feedback from PostgreSQL by trace_id
        trace_id = row.get("trace_id", "")
        feedback_stmt = select(AnswerFeedback).where(
            AnswerFeedback.org_id == org_id,
            AnswerFeedback.trace_id == trace_id,
        ).limit(1)
        result = await db.execute(feedback_stmt)
        feedback = result.scalars().first()

        items.append(LowRatedAnswerItem(
            message_id=feedback.message_id if feedback else uuid.UUID("00000000-0000-0000-0000-000000000000"),
            rating=row.get("rating", 0),
            reason_tags=feedback.reason_tags if feedback else [],
            comment=feedback.comment if feedback else None,
            created_at=feedback.created_at if feedback else None,
        ))

    logger.info("Get low-rated answers complete org_id=%s user_id=%s returned=%s", org_id, user["user_id"], len(items))
    return LowRatedAnswerResponse(items=items, total=len(items))


@router.get("/analytics/summary", response_model=RAGAnalyticsSummary)
async def get_analytics_summary(
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get RAG analytics summary for the org."""
    org_id = str(user["org_id"])
    logger.info("Get analytics summary request org_id=%s user_id=%s", org_id, user["user_id"])
    from app.services.clickhouse import clickhouse_client
    summary = await clickhouse_client.get_analytics_summary(org_id)
    logger.info("Get analytics summary complete org_id=%s user_id=%s keys=%s", org_id, user["user_id"], list(summary.keys()))
    return RAGAnalyticsSummary(**summary)


# --- Audit Logs ---

@router.get("/audit-logs", response_model=AuditLogListResponse)
async def get_audit_logs(
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    action: str | None = Query(None),
    resource_type: str | None = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """Get audit logs for the org."""
    org_id = str(user["org_id"])
    logger.info(
        "Get audit logs request org_id=%s user_id=%s action=%s resource_type=%s limit=%s offset=%s",
        org_id,
        user["user_id"],
        action,
        resource_type,
        limit,
        offset,
    )
    from app.services.audit import query_audit_logs
    items, total = await query_audit_logs(
        org_id=org_id,
        action=action,
        resource_type=resource_type,
        limit=limit,
        offset=offset,
    )
    logger.info("Get audit logs complete org_id=%s user_id=%s total=%s returned=%s", org_id, user["user_id"], total, len(items))
    return AuditLogListResponse(
        items=[AuditLogResponse.model_validate(i) for i in items],
        total=total,
    )
