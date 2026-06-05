import time
import uuid
import logging
import threading
from typing import Any

from app.config import settings
from app.logging_config import format_log_text
from app.services.embedding import embed_text
from app.services.llm import generate_stream
from app.services.weaviate_client import COLLECTION_NAME, get_client
from weaviate.classes.query import Filter, MetadataQuery
from app.services.query_rewriter import rewrite_query
from app.services.reranker import get_reranker

logger = logging.getLogger(__name__)


class _TTLCache:
    def __init__(self) -> None:
        self._items: dict[tuple[Any, ...], tuple[float, Any]] = {}
        self._lock = threading.RLock()

    def get(self, key: tuple[Any, ...], ttl: int) -> Any | None:
        if ttl <= 0:
            return None
        now = time.monotonic()
        with self._lock:
            item = self._items.get(key)
            if item is None:
                return None
            expires_at, value = item
            if expires_at <= now:
                self._items.pop(key, None)
                return None
            return value

    def set(self, key: tuple[Any, ...], value: Any, ttl: int) -> None:
        if ttl <= 0:
            return
        with self._lock:
            self._items[key] = (time.monotonic() + ttl, value)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()


_query_embedding_cache = _TTLCache()
_retrieval_cache = _TTLCache()


class RAGSource:
    def __init__(self, chunk_id: str, document_id: str, document_title: str,
                 section_path: str | None, page_start: int | None, page_end: int | None,
                 score: float, content_preview: str,
                 document_type: str = "general", section_type: str | None = None):
        content = content_preview or ""
        self.chunk_id = chunk_id
        self.document_id = document_id
        self.document_title = document_title
        self.section_path = section_path
        self.page_start = page_start
        self.page_end = page_end
        self.score = score
        self.content = content
        self.content_preview = content[:300]
        self.document_type = document_type
        self.section_type = section_type

    def clone(self) -> "RAGSource":
        return RAGSource(
            chunk_id=self.chunk_id,
            document_id=self.document_id,
            document_title=self.document_title,
            section_path=self.section_path,
            page_start=self.page_start,
            page_end=self.page_end,
            score=self.score,
            content_preview=self.content,
            document_type=self.document_type,
            section_type=self.section_type,
        )

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


def _clear_rag_caches() -> None:
    _query_embedding_cache.clear()
    _retrieval_cache.clear()


def _clone_sources(sources: list[RAGSource]) -> list[RAGSource]:
    return [source.clone() for source in sources]


def _query_cache_key(query: str) -> tuple[Any, ...]:
    return ("embedding", getattr(settings, "embedding_model", ""), query or "")


def _retrieval_cache_key(
    query: str,
    org_id: str,
    kb_ids: list[str],
    top_k: int,
    expand_query: bool,
) -> tuple[Any, ...]:
    return (
        "retrieval",
        org_id,
        tuple(sorted(str(kb_id) for kb_id in kb_ids)),
        top_k,
        expand_query,
        bool(getattr(settings, "query_expansion", True)),
        query or "",
    )


def _embed_query(query: str) -> list[float]:
    cache_key = _query_cache_key(query)
    cached = _query_embedding_cache.get(cache_key, getattr(settings, "query_cache_ttl", 0))
    if cached is not None:
        logger.debug("Query embedding cache hit query_length=%s", len(query or ""))
        return list(cached)

    vector = embed_text(query)
    _query_embedding_cache.set(cache_key, tuple(vector), getattr(settings, "query_cache_ttl", 0))
    return vector


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


def _score_for_log(score: Any) -> float:
    try:
        return round(float(score or 0), 4)
    except (TypeError, ValueError):
        return 0.0


def _summarize_source(source: RAGSource, preview_chars: int = 120) -> dict:
    return {
        "chunk_id": source.chunk_id,
        "document_id": source.document_id,
        "title": format_log_text(source.document_title, 80),
        "section_path": format_log_text(source.section_path, 120),
        "page_start": source.page_start,
        "page_end": source.page_end,
        "score": _score_for_log(source.score),
        "content_preview": format_log_text(source.content, preview_chars),
    }


def _summarize_sources(sources: list[RAGSource], limit: int = 5, preview_chars: int = 120) -> list[dict]:
    return [_summarize_source(source, preview_chars) for source in sources[:limit]]


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
    retrieval_cache_key = _retrieval_cache_key(query, org_id, kb_ids, top_k, expand_query)
    cached_sources = _retrieval_cache.get(retrieval_cache_key, getattr(settings, "retrieval_cache_ttl", 0))
    if cached_sources is not None:
        results = _clone_sources(cached_sources)
        if trace:
            trace.add_step("search", details={"retrieved_count": len(results), "cache_hit": True})
        logger.info(
            "Hybrid search retrieval cache hit org_id=%s kb_ids=%s top_k=%s expand_query=%s returned=%s "
            "top_results=%s",
            org_id,
            kb_ids,
            top_k,
            expand_query,
            len(results),
            _summarize_sources(results),
        )
        return results

    logger.info(
        "Hybrid search start org_id=%s kb_ids=%s top_k=%s expand_query=%s query=%r query_length=%s",
        org_id,
        kb_ids,
        top_k,
        expand_query,
        format_log_text(query, 500),
        len(query or ""),
    )
    queries = [query]
    if expand_query and getattr(settings, "query_expansion", True):
        rewrite_result = rewrite_query(query)
        queries = rewrite_result.expanded
        logger.info(
            "Hybrid search query rewrite complete org_id=%s expanded_count=%s entity_counts=%s queries=%s",
            org_id,
            len(queries),
            {k: len(v) for k, v in rewrite_result.entities.items()},
            [format_log_text(item, 180) for item in queries[:5]],
        )
        if trace:
            trace.add_step("rewrite", details={"rewrite_count": len(queries)})

    client = get_client()
    logger.info(
        "Hybrid search using Weaviate collection=%s org_id=%s filter_org_id=%s filter_status=%s filter_kb_ids=%s",
        COLLECTION_NAME,
        org_id,
        org_id,
        "ready",
        kb_ids,
    )
    collection = client.collections.get(COLLECTION_NAME)
    where = _build_where_filter(org_id, kb_ids)

    all_results = []
    seen_ids = set()

    for q in queries:
        query_start = time.monotonic()
        query_vector = _embed_query(q)
        logger.debug(
            "Hybrid search embedding ready org_id=%s query=%r query_length=%s vector_dims=%s",
            org_id,
            format_log_text(q, 300),
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
            "Hybrid search Weaviate query complete org_id=%s query=%r query_length=%s returned=%s "
            "duration_ms=%.2f top_results=%s",
            org_id,
            format_log_text(q, 300),
            len(q or ""),
            len(response.objects),
            (time.monotonic() - query_start) * 1000,
            _summarize_sources(
                [_weaviate_to_source(obj, _metadata_score(obj.metadata)) for obj in response.objects[:5]]
            ),
        )
        for obj in response.objects:
            obj_uuid = str(obj.uuid)
            if obj_uuid not in seen_ids:
                seen_ids.add(obj_uuid)
                score = _metadata_score(obj.metadata)
                all_results.append(_weaviate_to_source(obj, score))

    results = all_results[:top_k]
    _retrieval_cache.set(
        retrieval_cache_key,
        _clone_sources(results),
        getattr(settings, "retrieval_cache_ttl", 0),
    )
    duration_ms = (time.monotonic() - t0) * 1000
    if trace:
        trace.add_step("search", duration_ms=duration_ms, details={"retrieved_count": len(results)})
    logger.info(
        "Hybrid search complete org_id=%s kb_ids=%s expanded_count=%s unique_results=%s returned=%s "
        "duration_ms=%.2f top_results=%s",
        org_id,
        kb_ids,
        len(queries),
        len(all_results),
        len(results),
        duration_ms,
        _summarize_sources(results),
    )
    return results


def rerank_sources(
    query: str,
    sources: list[RAGSource],
    top_n: int | None = None,
    trace: object = None,
) -> list[RAGSource]:
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

    documents = [s.content for s in sources]
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
        "Rerank complete reranker=%s input_count=%s returned=%s duration_ms=%.2f top_results=%s",
        reranker.__class__.__name__,
        len(sources),
        len(reranked),
        duration_ms,
        _summarize_sources(reranked),
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
            + f"\n{source.content}\n"
        )
        citations.append(source.to_dict())

    context = "\n---\n".join(context_parts)
    logger.info(
        "Build context complete source_count=%s citation_count=%s context_length=%s duration_ms=%.2f citations=%s",
        len(sources),
        len(citations),
        len(context),
        (time.monotonic() - t0) * 1000,
        [
            {
                "chunk_id": item.get("chunk_id"),
                "document_id": item.get("document_id"),
                "title": format_log_text(item.get("document_title"), 80),
                "score": _score_for_log(item.get("score")),
                "page_start": item.get("page_start"),
            }
            for item in citations[:5]
        ],
    )
    return context, citations


def _build_history_aware_query(query: str, messages: list[dict]) -> str:
    if not messages:
        return query

    history_parts = [
        str(message.get("content", "")).strip()
        for message in messages
        if message.get("role") in {"system", "user", "assistant"} and str(message.get("content", "")).strip()
    ]
    if not history_parts:
        return query

    history_text = "\n".join(history_parts)
    max_history_chars = 2000
    if len(history_text) > max_history_chars:
        history_text = history_text[-max_history_chars:]

    return f"{history_text}\n当前问题：{query}"


def assemble_context_and_generate(
    query: str,
    org_id: str,
    kb_ids: list[str],
    max_chunks: int = settings.rag_max_chunks,
    user_id: str = "",
    messages: list[dict] | None = None,
):
    """Full RAG pipeline: query rewrite -> search -> rerank -> context -> stream generate.
    Yields (delta, is_done, sources) dicts.
    """
    from app.services.rag_trace import trace_collector

    trace_id = str(uuid.uuid4())
    t_start = time.monotonic()
    logger.info(
        "RAG pipeline start trace_id=%s org_id=%s user_id=%s kb_ids=%s max_chunks=%s query=%r query_length=%s "
        "history_messages=%s",
        trace_id,
        org_id,
        user_id or "anonymous",
        kb_ids,
        max_chunks,
        format_log_text(query, 500),
        len(query or ""),
        len(messages or []),
    )

    # Start trace
    trace = trace_collector.start_trace(trace_id, org_id, user_id or "anonymous", query, kb_ids)

    # Step 1: Hybrid search with optional query expansion
    retrieval_query = _build_history_aware_query(query, messages or [])
    logger.info(
        "RAG retrieval query prepared trace_id=%s uses_history=%s retrieval_query=%r retrieval_query_length=%s",
        trace_id,
        retrieval_query != query,
        format_log_text(retrieval_query, 700),
        len(retrieval_query or ""),
    )
    sources = hybrid_search(retrieval_query, org_id, kb_ids, top_k=settings.rag_top_k, expand_query=True, trace=trace)
    logger.info(
        "RAG retrieval complete trace_id=%s source_count=%s top_results=%s",
        trace_id,
        len(sources),
        _summarize_sources(sources),
    )

    # Step 2: Rerank
    sources = rerank_sources(retrieval_query, sources, trace=trace)
    logger.info(
        "RAG rerank complete trace_id=%s source_count=%s top_results=%s",
        trace_id,
        len(sources),
        _summarize_sources(sources),
    )

    # Step 3: Truncate to max_chunks
    pre_truncate_count = len(sources)
    sources = sources[:max_chunks]
    logger.info(
        "RAG source selection complete trace_id=%s retrieved_after_rerank=%s selected=%s max_chunks=%s "
        "selected_sources=%s",
        trace_id,
        pre_truncate_count,
        len(sources),
        max_chunks,
        _summarize_sources(sources),
    )

    # Step 4: Build context
    context, citations = build_context(sources)
    if trace:
        trace.add_step("context", details={"source_count": len(sources)})

    if not context.strip():
        trace.total_latency_ms = (time.monotonic() - t_start) * 1000
        logger.info(
            "RAG pipeline complete trace_id=%s org_id=%s reason=no_context source_count=%s query=%r duration_ms=%.2f",
            trace_id,
            org_id,
            len(sources),
            format_log_text(query, 500),
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
        "RAG generation start trace_id=%s org_id=%s context_length=%s source_count=%s model=%s source_ids=%s",
        trace_id,
        org_id,
        len(context),
        len(sources),
        settings.llm_model,
        [source.chunk_id for source in sources],
    )
    response = generate_stream(query=query, context=context, messages=messages)

    accumulated = ""
    chunk_count = 0
    for chunk in response:
        if hasattr(chunk, "output") and chunk.output:
            delta = chunk.output.choices[0].get("message", {}).get("content", "")
            if delta:
                chunk_count += 1
                accumulated += delta
                logger.debug(
                    "RAG generation chunk trace_id=%s chunk_index=%s delta_length=%s answer_length=%s",
                    trace_id,
                    chunk_count,
                    len(delta),
                    len(accumulated),
                )
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
