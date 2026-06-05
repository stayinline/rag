import json
import logging
import time
from datetime import datetime, timezone
from uuid import UUID, uuid4

import anyio
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.config import settings
from app.database import get_db
from app.models.conversation import Conversation, ConversationMessage
from app.schemas.chat import ChatRequest, ChatStreamChunk
from app.services.rag import assemble_context_and_generate

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("")
async def create_chat(
    data: ChatRequest,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    start = time.monotonic()
    kb_ids = [str(kb) for kb in data.kb_ids] if data.kb_ids else []
    logger.info(
        "Chat request org_id=%s user_id=%s kb_count=%s stream=%s conversation_id=%s query_length=%s",
        user["org_id"],
        user["user_id"],
        len(kb_ids),
        data.stream,
        data.conversation_id,
        len(data.query or ""),
    )
    conversation, history_messages = await _load_conversation_context(db, user, data)
    history_message_count = len(history_messages)
    llm_messages = _build_llm_history_messages(history_messages)

    if data.stream:

        async def event_stream():
            chunk_count = 0
            last_trace_id = None
            full_answer = ""
            sources = []
            try:
                iterator = assemble_context_and_generate(
                    query=data.query,
                    org_id=str(user["org_id"]),
                    kb_ids=kb_ids,
                    user_id=str(user["user_id"]),
                    messages=llm_messages,
                )
                while True:
                    item = await anyio.to_thread.run_sync(_next_or_none, iterator)
                    if item is None:
                        break
                    chunk_count += 1
                    last_trace_id = item.get("trace_id") or last_trace_id
                    delta = item.get("delta", "")
                    full_answer += delta
                    done = item.get("done", False)
                    if done:
                        sources = item.get("sources", [])
                        conversation_id, message_id = await _persist_chat_turn(
                            db=db,
                            user=user,
                            data=data,
                            answer=full_answer,
                            trace_id=last_trace_id,
                            sources=sources,
                            conversation=conversation,
                            next_sequence=history_message_count,
                        )
                    else:
                        conversation_id, message_id = data.conversation_id, None
                    chunk = ChatStreamChunk(
                        delta=delta,
                        done=done,
                        trace_id=item.get("trace_id"),
                        message_id=message_id,
                        sources=sources if done else [],
                        conversation_id=conversation_id,
                    )
                    if chunk.done:
                        logger.info(
                            "Chat stream complete org_id=%s user_id=%s trace_id=%s chunks=%s "
                            "sources=%s duration_ms=%.2f",
                            user["org_id"],
                            user["user_id"],
                            chunk.trace_id,
                            chunk_count,
                            len(chunk.sources),
                            (time.monotonic() - start) * 1000,
                        )
                    yield f"data: {json.dumps(chunk.model_dump(mode='json'), ensure_ascii=False)}\n\n"
            except Exception:
                logger.exception(
                    "Chat stream failed org_id=%s user_id=%s trace_id=%s chunks=%s duration_ms=%.2f",
                    user["org_id"],
                    user["user_id"],
                    last_trace_id,
                    chunk_count,
                    (time.monotonic() - start) * 1000,
                )
                raise

        return StreamingResponse(event_stream(), media_type="text/event-stream")
    else:
        # Non-streaming: accumulate full response
        full_answer = ""
        sources = []
        trace_id = None
        for item in assemble_context_and_generate(
            query=data.query,
            org_id=str(user["org_id"]),
            kb_ids=kb_ids,
            user_id=str(user["user_id"]),
            messages=llm_messages,
        ):
            full_answer += item.get("delta", "")
            if item.get("done"):
                sources = item.get("sources", [])
                trace_id = item.get("trace_id")

        logger.info(
            "Chat request complete org_id=%s user_id=%s trace_id=%s answer_length=%s sources=%s duration_ms=%.2f",
            user["org_id"],
            user["user_id"],
            trace_id,
            len(full_answer),
            len(sources),
            (time.monotonic() - start) * 1000,
        )
        conversation_id, message_id = await _persist_chat_turn(
            db=db,
            user=user,
            data=data,
            answer=full_answer,
            trace_id=trace_id,
            sources=sources,
            conversation=conversation,
            next_sequence=history_message_count,
        )

        return {
            "answer": full_answer,
            "trace_id": trace_id or "",
            "message_id": message_id,
            "conversation_id": str(conversation_id),
            "sources": sources,
            "model": settings.llm_model,
            "prompt_version": "v1",
        }


async def _load_conversation_context(
    db: AsyncSession,
    user: dict,
    data: ChatRequest,
) -> tuple[Conversation | None, list[ConversationMessage]]:
    if not data.conversation_id:
        return None, []

    result = await db.execute(
        select(Conversation).where(
            Conversation.id == data.conversation_id,
            Conversation.org_id == user["org_id"],
            Conversation.user_id == user["user_id"],
            Conversation.deleted_at.is_(None),
        )
    )
    conversation = result.scalar_one_or_none()
    if not conversation:
        logger.warning(
            "Chat conversation lookup failed org_id=%s user_id=%s conversation_id=%s reason=not_found",
            user["org_id"],
            user["user_id"],
            data.conversation_id,
        )
        raise HTTPException(status_code=404, detail="Conversation not found")

    messages_result = await db.execute(
        select(ConversationMessage)
        .where(
            ConversationMessage.org_id == user["org_id"],
            ConversationMessage.conversation_id == conversation.id,
            ConversationMessage.user_id == user["user_id"],
        )
        .order_by(ConversationMessage.sequence.asc(), ConversationMessage.created_at.asc())
    )
    messages = list(messages_result.scalars().all())
    logger.info(
        "Chat conversation context loaded org_id=%s user_id=%s conversation_id=%s messages=%s",
        user["org_id"],
        user["user_id"],
        conversation.id,
        len(messages),
    )
    return conversation, messages


def _build_llm_history_messages(history_messages: list[ConversationMessage]) -> list[dict]:
    if not history_messages:
        return []

    max_context_rounds = max(int(getattr(settings, "max_context_rounds", 2)), 0)
    summary_after_rounds = max(int(getattr(settings, "summary_after_rounds", 2)), 0)
    recent_message_count = max_context_rounds * 2

    sorted_messages = sorted(
        history_messages,
        key=lambda message: (
            getattr(message, "sequence", 0) or 0,
            getattr(message, "created_at", datetime.min.replace(tzinfo=timezone.utc)),
        ),
    )
    if summary_after_rounds <= 0 or len(sorted_messages) <= summary_after_rounds * 2:
        older_messages = []
        recent_messages = sorted_messages
    else:
        older_messages = sorted_messages[:-recent_message_count] if recent_message_count else sorted_messages
        recent_messages = sorted_messages[-recent_message_count:] if recent_message_count else []

    llm_messages = []
    if older_messages:
        llm_messages.append({"role": "system", "content": _summarize_history_for_prompt(older_messages)})

    for message in recent_messages:
        role = getattr(message, "role", "")
        if role not in {"user", "assistant"}:
            continue
        content = str(getattr(message, "content", "") or "").strip()
        if content:
            llm_messages.append({"role": role, "content": content})

    return llm_messages


def _summarize_history_for_prompt(messages: list[ConversationMessage]) -> str:
    lines = []
    for message in messages:
        role = getattr(message, "role", "")
        if role not in {"user", "assistant"}:
            continue
        content = str(getattr(message, "content", "") or "").strip()
        if not content:
            continue
        label = "用户" if role == "user" else "助手"
        if len(content) > 500:
            content = f"{content[:500]}..."
        lines.append(f"{label}: {content}")

    summary = "\n".join(lines)
    max_summary_chars = 4000
    if len(summary) > max_summary_chars:
        summary = summary[-max_summary_chars:]

    return (
        "以下是本轮之前较早对话的压缩摘要。回答当前问题时必须结合这些历史事实、约束、用户偏好和指代关系；"
        "如果摘要与最新原文消息冲突，以最新原文消息为准。\n"
        f"{summary}"
    )


async def _persist_chat_turn(
    db: AsyncSession,
    user: dict,
    data: ChatRequest,
    answer: str,
    trace_id: str | None,
    sources: list,
    conversation: Conversation | None = None,
    next_sequence: int | None = None,
) -> tuple[UUID, UUID]:
    now = datetime.now(timezone.utc)
    kb_ids = [str(kb_id) for kb_id in data.kb_ids] if data.kb_ids else []

    if conversation is None and data.conversation_id:
        result = await db.execute(
            select(Conversation).where(
                Conversation.id == data.conversation_id,
                Conversation.org_id == user["org_id"],
                Conversation.user_id == user["user_id"],
                Conversation.deleted_at.is_(None),
            )
        )
        conversation = result.scalar_one_or_none()
        if not conversation:
            logger.warning(
                "Chat conversation lookup failed org_id=%s user_id=%s conversation_id=%s reason=not_found",
                user["org_id"],
                user["user_id"],
                data.conversation_id,
            )
            raise HTTPException(status_code=404, detail="Conversation not found")
    elif conversation is None:
        conversation = Conversation(
            id=uuid4(),
            org_id=user["org_id"],
            user_id=user["user_id"],
            title=_make_conversation_title(data.query),
            kb_ids=kb_ids,
            message_count=0,
            last_message_at=now,
        )
        db.add(conversation)
        await db.flush()

    next_sequence_value = next_sequence if next_sequence is not None else (conversation.message_count or 0)
    if data.conversation_id and next_sequence is None:
        count_result = await db.execute(
            select(func.count())
            .select_from(ConversationMessage)
            .where(
                ConversationMessage.org_id == user["org_id"],
                ConversationMessage.conversation_id == conversation.id,
                ConversationMessage.user_id == user["user_id"],
            )
        )
        next_sequence_value = count_result.scalar() or next_sequence_value

    user_message = ConversationMessage(
        id=uuid4(),
        org_id=user["org_id"],
        conversation_id=conversation.id,
        user_id=user["user_id"],
        role="user",
        content=data.query,
        sequence=next_sequence_value + 1,
        kb_ids=kb_ids,
    )
    assistant_message = ConversationMessage(
        id=uuid4(),
        org_id=user["org_id"],
        conversation_id=conversation.id,
        user_id=user["user_id"],
        role="assistant",
        content=answer,
        sequence=next_sequence_value + 2,
        trace_id=trace_id,
        sources=sources,
        kb_ids=kb_ids,
        model=settings.llm_model,
        prompt_version="v1",
    )
    db.add(user_message)
    db.add(assistant_message)

    existing_kb_ids = conversation.kb_ids or []
    if kb_ids and not existing_kb_ids:
        conversation.kb_ids = kb_ids
    conversation.message_count = next_sequence_value + 2
    conversation.last_message_at = now

    await db.commit()
    await db.refresh(conversation)
    await db.refresh(assistant_message)
    logger.info(
        "Chat turn persisted org_id=%s user_id=%s conversation_id=%s user_message_id=%s assistant_message_id=%s",
        user["org_id"],
        user["user_id"],
        conversation.id,
        user_message.id,
        assistant_message.id,
    )
    return conversation.id, assistant_message.id


def _make_conversation_title(query: str) -> str:
    title = " ".join((query or "").strip().split())
    if len(title) > 40:
        return f"{title[:40]}..."
    return title or "新对话"


def _next_or_none(iterator):
    try:
        return next(iterator)
    except StopIteration:
        return None
