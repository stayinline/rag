"""Audit logging service."""
import logging

from sqlalchemy import select, func

from app.database import async_session
from app.models.audit import AuditLog

logger = logging.getLogger(__name__)


async def write_audit_log(
    org_id: str,
    user_id: str,
    action: str,
    resource_type: str | None = None,
    resource_id: str | None = None,
    details: dict | None = None,
    ip_address: str | None = None,
    user_agent: str | None = None,
    status_code: int | None = None,
):
    """Write an audit log entry."""
    import uuid

    entry = AuditLog(
        id=uuid.uuid4(),
        org_id=org_id,
        user_id=user_id,
        action=action,
        resource_type=resource_type,
        resource_id=resource_id,
        details=details or {},
        ip_address=ip_address,
        user_agent=user_agent,
        status_code=status_code,
    )
    async with async_session() as session:
        session.add(entry)
        await session.commit()


async def query_audit_logs(
    org_id: str,
    action: str | None = None,
    resource_type: str | None = None,
    user_id: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> tuple[list[AuditLog], int]:
    """Query audit logs with filters."""
    async with async_session() as session:
        stmt = select(AuditLog).where(AuditLog.org_id == org_id)

        if action:
            stmt = stmt.where(AuditLog.action == action)
        if resource_type:
            stmt = stmt.where(AuditLog.resource_type == resource_type)
        if user_id:
            stmt = stmt.where(AuditLog.user_id == user_id)

        # Get total count
        count_stmt = select(func.count()).select_from(stmt.subquery())
        total_result = await session.execute(count_stmt)
        total = total_result.scalar() or 0

        stmt = stmt.order_by(AuditLog.created_at.desc()).limit(limit).offset(offset)
        result = await session.execute(stmt)
        items = list(result.scalars().all())
        return items, total
