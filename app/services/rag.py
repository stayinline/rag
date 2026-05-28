import uuid

from app.config import settings
from app.services.llm import generate_stream
from app.services.weaviate_client import COLLECTION_NAME, get_client


class RAGSource:
    def __init__(self, chunk_id: str, document_id: str, document_title: str,
                 section_path: str | None, page_start: int | None, page_end: int | None,
                 score: float, content_preview: str):
        self.chunk_id = chunk_id
        self.document_id = document_id
        self.document_title = document_title
        self.section_path = section_path
        self.page_start = page_start
        self.page_end = page_end
        self.score = score
        self.content_preview = content_preview[:300]

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
        }


def hybrid_search(
    query: str,
    org_id: str,
    kb_ids: list[str],
    top_k: int = settings.rag_top_k,
) -> list[RAGSource]:
    """Hybrid search with permission filters."""
    client = get_client()
    client.connect()
    try:
        collection = client.collections.get(COLLECTION_NAME)

        # Build mandatory filters
        from weaviate.classes.query import Filter

        where = Filter.all_of([
            Filter.by_property("org_id").equal(org_id),
            Filter.by_property("status").equal("ready"),
        ])
        if kb_ids:
            where = Filter.all_of([
                Filter.by_property("org_id").equal(org_id),
                Filter.by_property("status").equal("ready"),
                Filter.by_property("kb_id").contains_any(kb_ids),
            ])

        # Hybrid search with BM25 + vector
        response = collection.query.hybrid(
            query=query,
            filters=where,
            limit=top_k,
            alpha=0.5,
            return_metadata={"score", "explain_score"},
        )

        sources = []
        for obj in response.objects:
            props = obj.properties
            metadata = obj.metadata or {}
            score = metadata.get("score", 0.0) if metadata else 0.0

            sources.append(RAGSource(
                chunk_id=str(obj.uuid),
                document_id=props.get("document_id", ""),
                document_title=props.get("title", ""),
                section_path=props.get("section_path"),
                page_start=props.get("page_start"),
                page_end=props.get("page_end"),
                score=score,
                content_preview=props.get("content", ""),
            ))
        return sources

    finally:
        client.close()


def build_context(sources: list[RAGSource]) -> tuple[str, list[dict]]:
    """Build context string from sources and return citation info."""
    context_parts = []
    citations = []

    for idx, source in enumerate(sources, 1):
        context_parts.append(
            f"[{idx}] {source.document_title}"
            + (f" - {source.section_path}" if source.section_path else "")
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
    """Full RAG pipeline: search -> context -> stream generate.
    Yields (delta, is_done, sources) tuples.
    """
    trace_id = str(uuid.uuid4())

    # Step 1: Hybrid search with permission filters
    sources = hybrid_search(query, org_id, kb_ids, top_k=settings.rag_top_k)
    sources = sources[:max_chunks]

    # Step 2: Build context
    context, citations = build_context(sources)

    if not context.strip():
        yield {
            "delta": "未找到相关的参考资料，无法回答此问题。",
            "done": True,
            "trace_id": trace_id,
            "sources": [],
        }
        return

    # Step 3: Generate answer (streaming)
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
