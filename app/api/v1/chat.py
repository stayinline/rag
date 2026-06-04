import json

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.api.deps import get_current_user
from app.config import settings
from app.schemas.chat import ChatRequest, ChatStreamChunk
from app.services.rag import assemble_context_and_generate

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("")
async def create_chat(
    data: ChatRequest,
    user: dict = Depends(get_current_user),
):
    kb_ids = [str(kb) for kb in data.kb_ids] if data.kb_ids else []

    if data.stream:

        def event_stream():
            for item in assemble_context_and_generate(
                query=data.query,
                org_id=str(user["org_id"]),
                kb_ids=kb_ids,
            ):
                chunk = ChatStreamChunk(
                    delta=item.get("delta", ""),
                    done=item.get("done", False),
                    trace_id=item.get("trace_id"),
                    sources=item.get("sources", []),
                    conversation_id=data.conversation_id,
                )
                yield f"data: {json.dumps(chunk.model_dump(), ensure_ascii=False)}\n\n"

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
        ):
            full_answer += item.get("delta", "")
            if item.get("done"):
                sources = item.get("sources", [])
                trace_id = item.get("trace_id")

        return {
            "answer": full_answer,
            "trace_id": trace_id or "",
            "conversation_id": str(data.conversation_id) if data.conversation_id else None,
            "sources": sources,
            "model": settings.llm_model,
            "prompt_version": "v1",
        }
