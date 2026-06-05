import logging
import time

from fastapi import APIRouter, Depends

from app.api.deps import get_current_user
from app.logging_config import format_log_text
from app.schemas.search import SearchRequest
from app.services.rag import retrieve_sources

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/search", tags=["search"])


@router.post("")
async def search(
    data: SearchRequest,
    user: dict = Depends(get_current_user),
):
    start = time.monotonic()
    kb_ids = [str(kb) for kb in data.kb_ids]
    logger.info(
        "Search request received org_id=%s user_id=%s kb_ids=%s top_k=%s query=%r query_length=%s filters=%s",
        user["org_id"],
        user["user_id"],
        kb_ids,
        data.top_k,
        format_log_text(data.query, 500),
        len(data.query or ""),
        data.filters,
    )
    sources = retrieve_sources(
        query=data.query,
        org_id=str(user["org_id"]),
        kb_ids=kb_ids,
        top_k=data.top_k,
        expand_query=True,
    )

    results = []
    for s in sources:
        results.append({
            "chunk_id": s.chunk_id,
            "document_id": s.document_id,
            "document_title": s.document_title,
            "section_path": s.section_path,
            "page_start": s.page_start,
            "page_end": s.page_end,
            "content_preview": s.content_preview[:300],
            "vector_score": s.vector_score,
            "bm25_score": s.bm25_score,
            "combined_score": s.score,
        })

    logger.info(
        "Search complete org_id=%s user_id=%s results=%s duration_ms=%.2f top_result=%s",
        user["org_id"],
        user["user_id"],
        len(results),
        (time.monotonic() - start) * 1000,
        _summarize_search_result(results[0]) if results else None,
    )
    return {
        "query": data.query,
        "total": len(results),
        "results": results,
    }


def _summarize_search_result(result: dict) -> dict:
    return {
        "chunk_id": result.get("chunk_id"),
        "document_id": result.get("document_id"),
        "title": result.get("document_title"),
        "score": result.get("combined_score"),
        "page_start": result.get("page_start"),
    }
