from urllib.parse import urlparse
import uuid

import weaviate
from weaviate.classes.query import Filter

from app.config import settings
from app.services.llm import generate_stream
from app.services.weaviate_client import COLLECTION_NAME, get_client
from app.services.query_rewriter import rewrite_query
from app.services.reranker import get_reranker


class RAGSource:
    def __init__(self, chunk_id: str, document_id: str, document_title: str,
                 section_path: str | None, page_start: int | None, page_end: int | None,
                 score: float, content_preview: str,
                 document_type: str = "general", section_type: str | None = None):
        self.chunk_id = chunk_id
        self.document_id = document_id
        self.document_title = document_title
        self.section_path = section_path
        self.page_start = page_start
        self.page_end = page_end
        self.score = score
        self.content_preview = content_preview[:300]
        self.document_type = document_type
        self.section_type = section_type

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "document_title": self.document_title,
            "section_path": self.section_path,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "score": self.score,
            "content_preview": self.content_preview,
            "document_type": self.document_type,
            "section_type": self.section_type,
        }


def _build_where_filter(org_id: str, kb_ids: list[str]) -> Filter:
    """Build Weaviate where filter with mandatory tenant and status constraints."""
    conditions = [
        Filter.by_property("org_id").equal(org_id),
        Filter.by_property("status").equal("ready"),
    ]
    if kb_ids:
        conditions.append(Filter.by_property("kb_id").contains_any(kb_ids))
    return Filter.all_of(conditions)


def _weaviate_to_source(obj, score: float) -> RAGSource:
    """Convert a Weaviate result object to RAGSource."""
    props = obj.properties
    return RAGSource(
        chunk_id=str(obj.uuid),
        document_id=props.get("document_id", ""),
        document_title=props.get("title", ""),
        section_path=props.get("section_path"),
        page_start=props.get("page_start"),
        page_end=props.get("page_end"),
        score=score,
        content_preview=props.get("content", ""),
        document_type=props.get("document_type", "general"),
        section_type=None,
    )


def hybrid_search(
    query: str,
    org_id: str,
    kb_ids: list[str],
    top_k: int = settings.rag_top_k,
    expand_query: bool = False,
) -> list[RAGSource]:
    """Hybrid search with permission filters and optional query expansion."""
    queries = [query]
    if expand_query and getattr(settings, "query_expansion", True):
        rewrite_result = rewrite_query(query)
        queries = rewrite_result.expanded

    client = get_client()
    client.connect()
    try:
        collection = client.collections.get(COLLECTION_NAME)
        where = _build_where_filter(org_id, kb_ids)

        all_results = []
        seen_ids = set()

        for q in queries:
            response = collection.query.hybrid(
                query=q,
                filters=where,
                limit=top_k,
                alpha=0.5,
                return_metadata={"score"},
            )
            for obj in response.objects:
                obj_uuid = str(obj.uuid)
                if obj_uuid not in seen_ids:
                    seen_ids.add(obj_uuid)
                    metadata = obj.metadata or {}
                    score = metadata.get("score", 0.0) if metadata else 0.0
                    all_results.append(_weaviate_to_source(obj, score))

        return all_results[:top_k]

    finally:
        client.close()


def rerank_sources(query: str, sources: list[RAGSource], top_n: int | None = None) -> list[RAGSource]:
    """Rerank sources using the configured reranker."""
    if not sources:
        return sources

    top_n = top_n or getattr(settings, "reranker_top_n", 10)
    reranker = get_reranker()

    documents = [s.content_preview for s in sources]
    results = reranker.rerank(query, documents)

    reranked = []
    for r in results[:top_n]:
        source = sources[r.index]
        source.score = r.score
        reranked.append(source)

    return reranked


def build_context(sources: list[RAGSource]) -> tuple[str, list[dict]]:
    """Build context string from sources and return citation info."""
    context_parts = []
    citations = []

    for idx, source in enumerate(sources, 1):
        context_parts.append(
            f"[{idx}] {source.document_title}"
            + (f" - {source.section_path}" if source.section_path else "")
            + (f" [{source.document_type}]" if source.document_type != "general" else "")
            + f"\n{source.content_preview}\n"
        )
        citations.append(source.to_dict())

    return "\n---\n".join(context_parts), citations


def assemble_context_and_generate(
    query: str,
    org_id: str,
    kb_ids: list[str],
    max_chunks: int = settings.rag_max_chunks,
):
    """Full RAG pipeline: query rewrite -> search -> rerank -> context -> stream generate.
    Yields (delta, is_done, sources) dicts.
    """
    trace_id = str(uuid.uuid4())

    # Step 1: Hybrid search with optional query expansion
    sources = hybrid_search(query, org_id, kb_ids, top_k=settings.rag_top_k, expand_query=True)

    # Step 2: Rerank
    sources = rerank_sources(query, sources)

    # Step 3: Truncate to max_chunks
    sources = sources[:max_chunks]

    # Step 4: Build context
    context, citations = build_context(sources)

    if not context.strip():
        yield {
            "delta": "未找到相关的参考资料，无法回答此问题。",
            "done": True,
            "trace_id": trace_id,
            "sources": [],
        }
        return

    # Step 5: Generate answer (streaming)
    response = generate_stream(query=query, context=context)

    accumulated = ""
    for chunk in response:
        if hasattr(chunk, "output") and chunk.output:
            delta = chunk.output.choices[0].get("message", {}).get("content", "")
            if delta:
                accumulated += delta
                yield {
                    "delta": delta,
                    "done": False,
                    "trace_id": trace_id,
                    "sources": [],
                }

    yield {
        "delta": "",
        "done": True,
        "trace_id": trace_id,
        "sources": [s.to_dict() for s in sources],
    }
