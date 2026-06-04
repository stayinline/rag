import logging
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select, update as sql_update
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.database import get_db
from app.models.conversation import Conversation, ConversationMessage
from app.schemas.conversation import ConversationDetailResponse, ConversationListResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.get("", response_model=ConversationListResponse)
async def list_conversations(
    limit: int = 50,
    offset: int = 0,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    safe_limit = min(max(limit, 1), 100)
    safe_offset = max(offset, 0)
    logger.info(
        "List conversations request org_id=%s user_id=%s limit=%s offset=%s",
        user["org_id"],
        user["user_id"],
        safe_limit,
        safe_offset,
    )
    base_stmt = select(Conversation).where(
        Conversation.org_id == user["org_id"],
        Conversation.user_id == user["user_id"],
        Conversation.deleted_at.is_(None),
    )
    count_stmt = select(func.count()).select_from(base_stmt.subquery())
    total = (await db.execute(count_stmt)).scalar() or 0

    result = await db.execute(
        base_stmt
        .order_by(Conversation.last_message_at.desc().nullslast(), Conversation.created_at.desc())
        .limit(safe_limit)
        .offset(safe_offset)
    )
    items = result.scalars().all()
    logger.info(
        "List conversations complete org_id=%s user_id=%s total=%s returned=%s",
        user["org_id"],
        user["user_id"],
        total,
        len(items),
    )
    return {"items": items, "total": total}


@router.get("/{conversation_id}", response_model=ConversationDetailResponse)
async def get_conversation(
    conversation_id: uuid.UUID,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    logger.info(
        "Get conversation request org_id=%s user_id=%s conversation_id=%s",
        user["org_id"],
        user["user_id"],
        conversation_id,
    )
    conversation = await _get_owned_conversation(conversation_id, user, db)

    result = await db.execute(
        select(ConversationMessage)
        .where(
            ConversationMessage.org_id == user["org_id"],
            ConversationMessage.conversation_id == conversation.id,
            ConversationMessage.user_id == user["user_id"],
        )
        .order_by(ConversationMessage.sequence.asc(), ConversationMessage.created_at.asc())
    )
    messages = result.scalars().all()
    logger.info(
        "Get conversation complete org_id=%s user_id=%s conversation_id=%s messages=%s",
        user["org_id"],
        user["user_id"],
        conversation_id,
        len(messages),
    )
    return {
        "id": conversation.id,
        "title": conversation.title,
        "kb_ids": conversation.kb_ids or [],
        "message_count": conversation.message_count,
        "last_message_at": conversation.last_message_at,
        "created_at": conversation.created_at,
        "updated_at": conversation.updated_at,
        "messages": messages,
    }


@router.delete("/{conversation_id}", status_code=204)
async def delete_conversation(
    conversation_id: uuid.UUID,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    logger.info(
        "Delete conversation request org_id=%s user_id=%s conversation_id=%s",
        user["org_id"],
        user["user_id"],
        conversation_id,
    )
    await _get_owned_conversation(conversation_id, user, db)

    await db.execute(
        sql_update(Conversation)
        .where(
            Conversation.id == conversation_id,
            Conversation.org_id == user["org_id"],
            Conversation.user_id == user["user_id"],
        )
        .values(deleted_at=datetime.now(timezone.utc))
    )
    await db.commit()
    logger.info(
        "Delete conversation succeeded org_id=%s user_id=%s conversation_id=%s",
        user["org_id"],
        user["user_id"],
        conversation_id,
    )


async def _get_owned_conversation(
    conversation_id: uuid.UUID,
    user: dict,
    db: AsyncSession,
) -> Conversation:
    result = await db.execute(
        select(Conversation).where(
            Conversation.id == conversation_id,
            Conversation.org_id == user["org_id"],
            Conversation.user_id == user["user_id"],
            Conversation.deleted_at.is_(None),
        )
    )
    conversation = result.scalar_one_or_none()
    if not conversation:
        logger.warning(
            "Conversation lookup failed org_id=%s user_id=%s conversation_id=%s reason=not_found",
            user["org_id"],
            user["user_id"],
            conversation_id,
        )
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation
