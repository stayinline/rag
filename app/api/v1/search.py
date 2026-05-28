from fastapi import APIRouter, Depends

from app.api.deps import get_current_user
from app.schemas.search import SearchRequest
from app.services.rag import hybrid_search

router = APIRouter(prefix="/search", tags=["search"])


@router.post("")
async def search(
    data: SearchRequest,
    user: dict = Depends(get_current_user),
):
    kb_ids = [str(kb) for kb in data.kb_ids]
    sources = hybrid_search(
        query=data.query,
        org_id=str(user["org_id"]),
        kb_ids=kb_ids,
        top_k=data.top_k,
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
            "vector_score": s.score,
            "bm25_score": s.score,
            "combined_score": s.score,
        })

    return {
        "query": data.query,
        "total": len(results),
        "results": results,
    }
