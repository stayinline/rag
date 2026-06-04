import time
import uuid
import logging

from app.config import settings
from app.services.embedding import embed_text
from app.services.llm import generate_stream
from app.services.weaviate_client import COLLECTION_NAME, get_client
from weaviate.classes.query import Filter, MetadataQuery
from app.services.query_rewriter import rewrite_query
from app.services.reranker import get_reranker

logger = logging.getLogger(__name__)


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


def _build_where_filter(org_id: str, kb_ids: list[str], security_levels: list[str] | None = None) -> Filter:
    """Build Weaviate where filter with mandatory tenant, status, and security constraints."""
    conditions = [
        Filter.by_property("org_id").equal(org_id),
        Filter.by_property("status").equal("ready"),
    ]
    if kb_ids:
        conditions.append(Filter.by_property("kb_id").contains_any(kb_ids))
    if security_levels:
        conditions.append(Filter.by_property("security_level").contains_any(security_levels))
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


def _metadata_score(metadata) -> float:
    if not metadata:
        return 0.0
    if isinstance(metadata, dict):
        return metadata.get("score") or 0.0
    return getattr(metadata, "score", None) or 0.0


def hybrid_search(
    query: str,
    org_id: str,
    kb_ids: list[str],
    top_k: int = settings.rag_top_k,
    expand_query: bool = False,
    trace: object = None,
) -> list[RAGSource]:
    """Hybrid search with permission filters and optional query expansion."""
    t0 = time.monotonic()
    logger.info(
        "Hybrid search start org_id=%s kb_count=%s top_k=%s expand_query=%s query_length=%s",
        org_id,
        len(kb_ids),
        top_k,
        expand_query,
        len(query or ""),
    )
    queries = [query]
    if expand_query and getattr(settings, "query_expansion", True):
        rewrite_result = rewrite_query(query)
        queries = rewrite_result.expanded
        logger.info(
            "Hybrid search query rewrite complete org_id=%s expanded_count=%s entity_counts=%s",
            org_id,
            len(queries),
            {k: len(v) for k, v in rewrite_result.entities.items()},
        )
        if trace:
            trace.add_step("rewrite", details={"rewrite_count": len(queries)})

    client = get_client()
    logger.debug("Hybrid search connecting to Weaviate collection=%s", COLLECTION_NAME)
    client.connect()
    try:
        collection = client.collections.get(COLLECTION_NAME)
        where = _build_where_filter(org_id, kb_ids)

        all_results = []
        seen_ids = set()

        for q in queries:
            query_start = time.monotonic()
            query_vector = embed_text(q)
            logger.debug(
                "Hybrid search embedding ready org_id=%s query_length=%s vector_dims=%s",
                org_id,
                len(q or ""),
                len(query_vector),
            )
            response = collection.query.hybrid(
                query=q,
                vector=query_vector,
                filters=where,
                limit=top_k,
                alpha=0.5,
                return_metadata=MetadataQuery(score=True),
            )
            logger.info(
                "Hybrid search Weaviate query complete org_id=%s query_length=%s returned=%s duration_ms=%.2f",
                org_id,
                len(q or ""),
                len(response.objects),
                (time.monotonic() - query_start) * 1000,
            )
            for obj in response.objects:
                obj_uuid = str(obj.uuid)
                if obj_uuid not in seen_ids:
                    seen_ids.add(obj_uuid)
                    score = _metadata_score(obj.metadata)
                    all_results.append(_weaviate_to_source(obj, score))

        results = all_results[:top_k]
        duration_ms = (time.monotonic() - t0) * 1000
        if trace:
            trace.add_step("search", duration_ms=duration_ms, details={"retrieved_count": len(results)})
        logger.info(
            "Hybrid search complete org_id=%s kb_count=%s expanded_count=%s unique_results=%s returned=%s duration_ms=%.2f",
            org_id,
            len(kb_ids),
            len(queries),
            len(all_results),
            len(results),
            duration_ms,
        )
        return results

    finally:
        client.close()
        logger.debug("Hybrid search Weaviate client closed org_id=%s", org_id)


def rerank_sources(query: str, sources: list[RAGSource], top_n: int | None = None, trace: object = None) -> list[RAGSource]:
    """Rerank sources using the configured reranker."""
    t0 = time.monotonic()
    if not sources:
        logger.info("Rerank skipped reason=no_sources query_length=%s", len(query or ""))
        return sources

    top_n = top_n or getattr(settings, "reranker_top_n", 10)
    reranker = get_reranker()
    logger.info(
        "Rerank start reranker=%s source_count=%s top_n=%s query_length=%s",
        reranker.__class__.__name__,
        len(sources),
        top_n,
        len(query or ""),
    )

    documents = [s.content_preview for s in sources]
    results = reranker.rerank(query, documents)

    reranked = []
    for r in results[:top_n]:
        source = sources[r.index]
        source.score = r.score
        reranked.append(source)

    duration_ms = (time.monotonic() - t0) * 1000
    if trace:
        trace.add_step("rerank", duration_ms=duration_ms, details={"reranked_count": len(reranked)})
    logger.info(
        "Rerank complete reranker=%s input_count=%s returned=%s duration_ms=%.2f",
        reranker.__class__.__name__,
        len(sources),
        len(reranked),
        duration_ms,
    )
    return reranked


def build_context(sources: list[RAGSource]) -> tuple[str, list[dict]]:
    """Build context string from sources and return citation info."""
    t0 = time.monotonic()
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

    context = "\n---\n".join(context_parts)
    logger.info(
        "Build context complete source_count=%s citation_count=%s context_length=%s duration_ms=%.2f",
        len(sources),
        len(citations),
        len(context),
        (time.monotonic() - t0) * 1000,
    )
    return context, citations


def assemble_context_and_generate(
    query: str,
    org_id: str,
    kb_ids: list[str],
    max_chunks: int = settings.rag_max_chunks,
    user_id: str = "",
):
    """Full RAG pipeline: query rewrite -> search -> rerank -> context -> stream generate.
    Yields (delta, is_done, sources) dicts.
    """
    from app.services.rag_trace import trace_collector

    trace_id = str(uuid.uuid4())
    t_start = time.monotonic()
    logger.info(
        "RAG pipeline start trace_id=%s org_id=%s user_id=%s kb_count=%s max_chunks=%s query_length=%s",
        trace_id,
        org_id,
        user_id or "anonymous",
        len(kb_ids),
        max_chunks,
        len(query or ""),
    )

    # Start trace
    trace = trace_collector.start_trace(trace_id, org_id, user_id or "anonymous", query, kb_ids)

    # Step 1: Hybrid search with optional query expansion
    sources = hybrid_search(query, org_id, kb_ids, top_k=settings.rag_top_k, expand_query=True, trace=trace)

    # Step 2: Rerank
    sources = rerank_sources(query, sources, trace=trace)

    # Step 3: Truncate to max_chunks
    pre_truncate_count = len(sources)
    sources = sources[:max_chunks]
    logger.info(
        "RAG source selection complete trace_id=%s retrieved_after_rerank=%s selected=%s max_chunks=%s",
        trace_id,
        pre_truncate_count,
        len(sources),
        max_chunks,
    )

    # Step 4: Build context
    context, citations = build_context(sources)
    if trace:
        trace.add_step("context", details={"source_count": len(sources)})

    if not context.strip():
        trace.total_latency_ms = (time.monotonic() - t_start) * 1000
        logger.info(
            "RAG pipeline complete trace_id=%s org_id=%s reason=no_context duration_ms=%.2f",
            trace_id,
            org_id,
            trace.total_latency_ms,
        )
        _write_trace_to_clickhouse(trace)
        yield {
            "delta": "未找到相关的参考资料，无法回答此问题。",
            "done": True,
            "trace_id": trace_id,
            "sources": [],
        }
        return

    # Step 5: Generate answer (streaming)
    t_gen_start = time.monotonic()
    logger.info(
        "RAG generation start trace_id=%s org_id=%s context_length=%s source_count=%s model=%s",
        trace_id,
        org_id,
        len(context),
        len(sources),
        settings.llm_model,
    )
    response = generate_stream(query=query, context=context)

    accumulated = ""
    chunk_count = 0
    for chunk in response:
        if hasattr(chunk, "output") and chunk.output:
            delta = chunk.output.choices[0].get("message", {}).get("content", "")
            if delta:
                chunk_count += 1
                accumulated += delta
                yield {
                    "delta": delta,
                    "done": False,
                    "trace_id": trace_id,
                    "sources": [],
                }

    gen_duration_ms = (time.monotonic() - t_gen_start) * 1000
    if trace:
        trace.add_step("generation", duration_ms=gen_duration_ms)
    logger.info(
        "RAG generation complete trace_id=%s chunks=%s answer_length=%s duration_ms=%.2f",
        trace_id,
        chunk_count,
        len(accumulated),
        gen_duration_ms,
    )

    trace.total_latency_ms = (time.monotonic() - t_start) * 1000
    logger.info(
        "RAG pipeline complete trace_id=%s org_id=%s user_id=%s sources=%s answer_length=%s duration_ms=%.2f",
        trace_id,
        org_id,
        user_id or "anonymous",
        len(sources),
        len(accumulated),
        trace.total_latency_ms,
    )
    _write_trace_to_clickhouse(trace)

    yield {
        "delta": "",
        "done": True,
        "trace_id": trace_id,
        "sources": [s.to_dict() for s in sources],
    }


def _write_trace_to_clickhouse(trace):
    """Write trace to ClickHouse asynchronously."""
    if not getattr(settings, "enable_trace_logging", True):
        logger.debug("Trace logging disabled trace_id=%s", getattr(trace, "trace_id", None))
        return
    try:
        import asyncio

        from app.services.clickhouse import clickhouse_client

        event = trace.to_clickhouse_event()
        logger.debug("Queue ClickHouse trace write trace_id=%s", trace.trace_id)
        coro = clickhouse_client.write_trace_event(event)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(coro)
        else:
            loop.create_task(coro)
    except Exception as e:
        logger.warning("Failed to write trace to ClickHouse: %s", e)
