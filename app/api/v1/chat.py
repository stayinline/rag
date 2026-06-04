import json
import logging
import time

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.api.deps import get_current_user
from app.config import settings
from app.schemas.chat import ChatRequest, ChatStreamChunk
from app.services.rag import assemble_context_and_generate

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("")
async def create_chat(
    data: ChatRequest,
    user: dict = Depends(get_current_user),
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

    if data.stream:

        def event_stream():
            chunk_count = 0
            last_trace_id = None
            try:
                for item in assemble_context_and_generate(
                    query=data.query,
                    org_id=str(user["org_id"]),
                    kb_ids=kb_ids,
                    user_id=str(user["user_id"]),
                ):
                    chunk_count += 1
                    last_trace_id = item.get("trace_id") or last_trace_id
                    chunk = ChatStreamChunk(
                        delta=item.get("delta", ""),
                        done=item.get("done", False),
                        trace_id=item.get("trace_id"),
                        sources=item.get("sources", []),
                        conversation_id=data.conversation_id,
                    )
                    if chunk.done:
                        logger.info(
                            "Chat stream complete org_id=%s user_id=%s trace_id=%s chunks=%s sources=%s duration_ms=%.2f",
                            user["org_id"],
                            user["user_id"],
                            chunk.trace_id,
                            chunk_count,
                            len(chunk.sources),
                            (time.monotonic() - start) * 1000,
                        )
                    yield f"data: {json.dumps(chunk.model_dump(), ensure_ascii=False)}\n\n"
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
        return {
            "answer": full_answer,
            "trace_id": trace_id or "",
            "conversation_id": str(data.conversation_id) if data.conversation_id else None,
            "sources": sources,
            "model": settings.llm_model,
            "prompt_version": "v1",
        }
