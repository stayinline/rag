import logging
from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.rag_trace import RAGTraceDetail
from app.services.rag_trace import RAGTrace

logger = logging.getLogger(__name__)


def _json_safe(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def trace_steps_to_dicts(trace: RAGTrace) -> list[dict]:
    return [
        {
            "step": step.step,
            "duration_ms": round(float(step.duration_ms or 0), 2),
            "started_at": step.started_at.isoformat(),
            "details": _json_safe(step.details or {}),
        }
        for step in trace.steps
    ]


async def save_rag_trace_detail(
    db: AsyncSession,
    *,
    trace: RAGTrace | None,
    org_id: str,
    user_id: str,
    conversation_id: UUID,
    message_id: UUID,
) -> None:
    if trace is None:
        logger.warning("RAG trace detail save skipped reason=missing_trace message_id=%s", message_id)
        return

    detail = RAGTraceDetail(
        org_id=UUID(str(org_id)),
        user_id=UUID(str(user_id)),
        trace_id=trace.trace_id,
        conversation_id=conversation_id,
        message_id=message_id,
        query=trace.query,
        answer_preview=trace.answer_preview or "",
        answer_length=int(trace.answer_length or 0),
        kb_ids=[str(kb_id) for kb_id in trace.kb_ids],
        steps=trace_steps_to_dicts(trace),
        total_latency_ms=int(trace.total_latency_ms or 0),
        model=trace.model or None,
        prompt_version=trace.prompt_version or None,
    )
    db.add(detail)
    logger.info(
        "RAG trace detail queued trace_id=%s message_id=%s steps=%s",
        trace.trace_id,
        message_id,
        len(trace.steps),
    )


async def list_rag_trace_details(
    db: AsyncSession,
    *,
    org_id: str,
    limit: int,
    offset: int,
) -> tuple[list[RAGTraceDetail], int]:
    stmt = select(RAGTraceDetail).where(RAGTraceDetail.org_id == UUID(str(org_id)))
    count_stmt = select(func.count()).select_from(stmt.subquery())
    total = (await db.execute(count_stmt)).scalar() or 0
    result = await db.execute(
        stmt.order_by(RAGTraceDetail.created_at.desc()).limit(limit).offset(offset)
    )
    return list(result.scalars().all()), int(total)


async def get_rag_trace_detail(
    db: AsyncSession,
    *,
    org_id: str,
    trace_id: str,
) -> RAGTraceDetail | None:
    result = await db.execute(
        select(RAGTraceDetail).where(
            RAGTraceDetail.org_id == UUID(str(org_id)),
            RAGTraceDetail.trace_id == trace_id,
        )
    )
    return result.scalar_one_or_none()
