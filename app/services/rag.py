import time
import uuid
import logging
import threading
from typing import Any

import tiktoken
from app.config import settings
from app.logging_config import format_log_text
from app.services.chunker import count_tokens
from app.services.citation_validator import validate_answer_citations
from app.services.contextual_compression import compress_sources_for_query
from app.services.embedding import embed_text
from app.services.llm import generate_stream
from app.services.planner import QueryPlan, build_query_plan
from app.services.weaviate_client import COLLECTION_NAME, get_client
from weaviate.classes.query import Filter, MetadataQuery
from app.services.query_rewriter import ConversationalQueryRewriteResult, rewrite_conversational_query, rewrite_query
from app.services.reranker import get_reranker

logger = logging.getLogger(__name__)
_TOKEN_ENCODER = tiktoken.get_encoding("cl100k_base")


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
                 document_type: str = "general", section_type: str | None = None,
                 chunk_index: int | None = None, parent_chunk_id: str | None = None,
                 kb_id: str = "", child_chunk_ids: list[str] | None = None,
                 rank_before: int = 0, rank_after: int = 0,
                 vector_score: float = 0.0, bm25_score: float = 0.0,
                 rerank_score: float = 0.0, combined_score: float | None = None,
                 metadata_score: float = 0.0, feedback_score: float = 0.0,
                 hybrid_score: float = 0.0):
        content = content_preview or ""
        self.chunk_id = chunk_id
        self.document_id = document_id
        self.kb_id = kb_id
        self.document_title = document_title
        self.section_path = section_path
        self.page_start = page_start
        self.page_end = page_end
        self.score = score
        self.content = content
        self.content_preview = content[:300]
        self.document_type = document_type
        self.section_type = section_type
        self.chunk_index = chunk_index
        self.parent_chunk_id = parent_chunk_id
        self.child_chunk_ids = list(child_chunk_ids or [])
        self.rank_before = rank_before
        self.rank_after = rank_after
        self.vector_score = vector_score
        self.bm25_score = bm25_score
        self.rerank_score = rerank_score
        self.combined_score = score if combined_score is None else combined_score
        self.metadata_score = metadata_score
        self.feedback_score = feedback_score
        self.hybrid_score = hybrid_score

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
            chunk_index=self.chunk_index,
            parent_chunk_id=self.parent_chunk_id,
            kb_id=self.kb_id,
            child_chunk_ids=self.child_chunk_ids,
            rank_before=self.rank_before,
            rank_after=self.rank_after,
            vector_score=self.vector_score,
            bm25_score=self.bm25_score,
            rerank_score=self.rerank_score,
            combined_score=self.combined_score,
            metadata_score=self.metadata_score,
            feedback_score=self.feedback_score,
            hybrid_score=self.hybrid_score,
        )

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "kb_id": self.kb_id,
            "document_title": self.document_title,
            "section_path": self.section_path,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "score": self.score,
            "content_preview": self.content_preview,
            "document_type": self.document_type,
            "section_type": self.section_type,
            "chunk_index": self.chunk_index,
            "parent_chunk_id": self.parent_chunk_id,
            "child_chunk_ids": self.child_chunk_ids,
            "rank_before": self.rank_before,
            "rank_after": self.rank_after,
            "vector_score": self.vector_score,
            "bm25_score": self.bm25_score,
            "combined_score": self.combined_score,
            "metadata_score": self.metadata_score,
            "feedback_score": self.feedback_score,
            "hybrid_score": self.hybrid_score,
            "rerank_score": self.rerank_score,
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


def _effective_context_budget_tokens() -> int:
    context_window = int(getattr(settings, "llm_context_window_tokens", 8192) or 8192)
    output_budget = int(getattr(settings, "llm_max_output_tokens", 2048) or 2048)
    safety_margin = int(getattr(settings, "rag_context_safety_margin_tokens", 512) or 512)
    budget = context_window - output_budget - safety_margin
    return max(budget, int(getattr(settings, "rag_chunk_size", 600) or 600))


def _embed_query(query: str) -> tuple[list[float], bool, float]:
    t0 = time.monotonic()
    cache_key = _query_cache_key(query)
    cached = _query_embedding_cache.get(cache_key, getattr(settings, "query_cache_ttl", 0))
    if cached is not None:
        logger.debug("Query embedding cache hit query_length=%s", len(query or ""))
        return list(cached), True, (time.monotonic() - t0) * 1000

    vector = embed_text(query)
    _query_embedding_cache.set(cache_key, tuple(vector), getattr(settings, "query_cache_ttl", 0))
    return vector, False, (time.monotonic() - t0) * 1000


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


def _build_relaxed_where_filter(org_id: str) -> Filter:
    """Build Weaviate filter without optional KB constraints for zero-result fallback."""
    return _build_where_filter(org_id, [])


def _weaviate_to_source(
    obj,
    score: float,
    *,
    vector_score: float = 0.0,
    bm25_score: float = 0.0,
    combined_score: float | None = None,
    metadata_score: float = 0.0,
    hybrid_score: float = 0.0,
) -> RAGSource:
    """Convert a Weaviate result object to RAGSource."""
    props = obj.properties
    combined = score if combined_score is None else combined_score
    return RAGSource(
        chunk_id=str(obj.uuid),
        document_id=props.get("document_id", ""),
        kb_id=props.get("kb_id", ""),
        document_title=props.get("title", ""),
        section_path=props.get("section_path"),
        page_start=props.get("page_start"),
        page_end=props.get("page_end"),
        score=score,
        content_preview=props.get("content", ""),
        document_type=props.get("document_type", "general"),
        section_type=props.get("section_type"),
        chunk_index=props.get("chunk_index"),
        parent_chunk_id=props.get("parent_chunk_id"),
        child_chunk_ids=props.get("child_chunk_ids") or [],
        vector_score=vector_score,
        bm25_score=bm25_score,
        rerank_score=score,
        combined_score=combined,
        metadata_score=metadata_score,
        hybrid_score=hybrid_score,
    )


def _metadata_score(metadata) -> float:
    if not metadata:
        return 0.0
    if isinstance(metadata, dict):
        return metadata.get("score") or 0.0
    return getattr(metadata, "score", None) or 0.0


def _metadata_distance(metadata) -> float | None:
    if not metadata:
        return None
    value = metadata.get("distance") if isinstance(metadata, dict) else getattr(metadata, "distance", None)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _vector_score_from_metadata(metadata) -> float:
    distance = _metadata_distance(metadata)
    if distance is None:
        return _metadata_score(metadata)
    return max(0.0, 1.0 - distance)


def _score_map(objects) -> dict[str, float]:
    return {str(obj.uuid): _metadata_score(obj.metadata) for obj in objects}


def _vector_score_map(objects) -> dict[str, float]:
    return {str(obj.uuid): _vector_score_from_metadata(obj.metadata) for obj in objects}


def _score_for_log(score: Any) -> float:
    try:
        return round(float(score or 0), 4)
    except (TypeError, ValueError):
        return 0.0


def _safe_score(score: Any) -> float:
    try:
        return float(score or 0)
    except (TypeError, ValueError):
        return 0.0


def _is_uuid_like(value: Any) -> bool:
    if not value:
        return False
    try:
        uuid.UUID(str(value))
    except (TypeError, ValueError):
        return False
    return True


def _summarize_source(source: RAGSource, preview_chars: int = 120) -> dict:
    return {
        "chunk_id": source.chunk_id,
        "document_id": source.document_id,
        "kb_id": source.kb_id,
        "title": format_log_text(source.document_title, 80),
        "section_path": format_log_text(source.section_path, 120),
        "page_start": source.page_start,
        "page_end": source.page_end,
        "score": _score_for_log(source.score),
        "vector_score": _score_for_log(source.vector_score),
        "bm25_score": _score_for_log(source.bm25_score),
        "combined_score": _score_for_log(source.combined_score),
        "metadata_score": _score_for_log(source.metadata_score),
        "feedback_score": _score_for_log(source.feedback_score),
        "hybrid_score": _score_for_log(source.hybrid_score),
        "content_preview": format_log_text(source.content, preview_chars),
    }


def _summarize_sources(sources: list[RAGSource], limit: int = 5, preview_chars: int = 120) -> list[dict]:
    return [_summarize_source(source, preview_chars) for source in sources[:limit]]


def _response_objects(response) -> list:
    objects = getattr(response, "objects", None)
    if objects is None:
        return []
    if isinstance(objects, list):
        return objects
    if not isinstance(objects, tuple):
        module_name = type(objects).__module__
        if module_name.startswith("unittest.mock"):
            return []
    try:
        return list(objects)
    except TypeError:
        return []


def _fallback_top_k(top_k: int) -> int:
    multiplier = int(getattr(settings, "rag_fallback_top_k_multiplier", 2) or 2)
    return max(top_k + 1, top_k * max(multiplier, 1))


def _query_hybrid_objects(collection, *, query: str, query_vector: list[float], where: Filter, limit: int) -> list:
    response = collection.query.hybrid(
        query=query,
        vector=query_vector,
        filters=where,
        limit=limit,
        alpha=0.5,
        return_metadata=MetadataQuery(score=True),
    )
    return _response_objects(response)


def _query_dense_objects(collection, *, query_vector: list[float], where: Filter, limit: int) -> list:
    response = collection.query.near_vector(
        near_vector=query_vector,
        filters=where,
        limit=limit,
        return_metadata=MetadataQuery(score=True, distance=True),
    )
    return _response_objects(response)


def _query_bm25_objects(collection, *, query: str, where: Filter, limit: int) -> list:
    response = collection.query.bm25(
        query=query,
        filters=where,
        limit=limit,
        return_metadata=MetadataQuery(score=True),
    )
    return _response_objects(response)


def _query_metadata_objects(collection, *, where: Filter, limit: int) -> list:
    response = collection.query.fetch_objects(
        filters=where,
        limit=limit,
        return_metadata=MetadataQuery(score=True),
    )
    return _response_objects(response)


def _component_score_maps(
    collection,
    *,
    query: str,
    query_vector: list[float],
    where: Filter,
    limit: int,
) -> tuple[dict[str, float], dict[str, float]]:
    vector_scores: dict[str, float] = {}
    bm25_scores: dict[str, float] = {}
    try:
        vector_scores = _vector_score_map(
            _query_dense_objects(collection, query_vector=query_vector, where=where, limit=limit)
        )
    except Exception as exc:
        logger.warning("Dense component score lookup failed: %s", exc, exc_info=True)
    try:
        bm25_scores = _score_map(_query_bm25_objects(collection, query=query, where=where, limit=limit))
    except Exception as exc:
        logger.warning("BM25 component score lookup failed: %s", exc, exc_info=True)
    return vector_scores, bm25_scores


def _metadata_filter_terms(query: str) -> list[str]:
    rewrite_result = rewrite_query(query, expand_synonyms=False)
    terms: list[str] = []
    for values in rewrite_result.entities.values():
        for value in values:
            term = str(value).split("->")[-1].strip()
            if term and term not in terms:
                terms.append(term)
    return terms


def _metadata_where_filter(base_where: Filter, terms: list[str]) -> Filter | None:
    if not terms:
        return None
    return Filter.all_of([
        base_where,
        Filter.any_of([
            Filter.by_property("domain_tags").contains_any(terms),
            Filter.by_property("entities").contains_any(terms),
        ]),
    ])


def _normalized_score_map(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    max_score = max(abs(score) for score in scores.values())
    if max_score <= 0:
        return {key: 0.0 for key in scores}
    return {key: max(score, 0.0) / max_score for key, score in scores.items()}


def _combined_retrieval_score(
    *,
    vector_score: float,
    bm25_score: float,
    metadata_score: float,
    hybrid_score: float,
) -> float:
    return (
        float(getattr(settings, "retrieval_vector_weight", 0.4) or 0.0) * vector_score
        + float(getattr(settings, "retrieval_bm25_weight", 0.3) or 0.0) * bm25_score
        + float(getattr(settings, "retrieval_metadata_weight", 0.2) or 0.0) * metadata_score
        + float(getattr(settings, "retrieval_hybrid_weight", 0.1) or 0.0) * hybrid_score
    )


def _fused_score_for_uuid(
    obj_uuid: str,
    *,
    hybrid_scores: dict[str, float],
    vector_scores: dict[str, float],
    bm25_scores: dict[str, float],
    metadata_scores: dict[str, float],
    normalized_hybrid: dict[str, float],
    normalized_vector: dict[str, float],
    normalized_bm25: dict[str, float],
    normalized_metadata: dict[str, float],
) -> float:
    if hybrid_scores and not vector_scores and not bm25_scores and not metadata_scores:
        return hybrid_scores.get(obj_uuid, 0.0)
    return _combined_retrieval_score(
        vector_score=normalized_vector.get(obj_uuid, 0.0),
        bm25_score=normalized_bm25.get(obj_uuid, 0.0),
        metadata_score=normalized_metadata.get(obj_uuid, 0.0),
        hybrid_score=normalized_hybrid.get(obj_uuid, 0.0),
    )


def _append_multi_index_sources(
    *,
    objects: list,
    hybrid_scores: dict[str, float],
    vector_scores: dict[str, float],
    bm25_scores: dict[str, float],
    metadata_scores: dict[str, float],
    seen_ids: set[str],
    all_results: list[RAGSource],
) -> int:
    normalized_vector = _normalized_score_map(vector_scores)
    normalized_bm25 = _normalized_score_map(bm25_scores)
    normalized_metadata = _normalized_score_map(metadata_scores)
    normalized_hybrid = _normalized_score_map(hybrid_scores)

    added_count = 0
    for obj in objects:
        obj_uuid = str(obj.uuid)
        if obj_uuid in seen_ids:
            continue
        seen_ids.add(obj_uuid)
        combined_score = _fused_score_for_uuid(
            obj_uuid,
            hybrid_scores=hybrid_scores,
            vector_scores=vector_scores,
            bm25_scores=bm25_scores,
            metadata_scores=metadata_scores,
            normalized_hybrid=normalized_hybrid,
            normalized_vector=normalized_vector,
            normalized_bm25=normalized_bm25,
            normalized_metadata=normalized_metadata,
        )
        all_results.append(
            _weaviate_to_source(
                obj,
                combined_score,
                vector_score=vector_scores.get(obj_uuid, 0.0),
                bm25_score=bm25_scores.get(obj_uuid, 0.0),
                combined_score=combined_score,
                metadata_score=metadata_scores.get(obj_uuid, 0.0),
                hybrid_score=hybrid_scores.get(obj_uuid, 0.0),
            )
        )
        added_count += 1
    all_results.sort(key=lambda source: source.combined_score, reverse=True)
    return added_count


def _append_hybrid_sources(
    *,
    objects: list,
    vector_scores: dict[str, float],
    bm25_scores: dict[str, float],
    seen_ids: set[str],
    all_results: list[RAGSource],
) -> int:
    added_count = 0
    for obj in objects:
        obj_uuid = str(obj.uuid)
        if obj_uuid in seen_ids:
            continue
        seen_ids.add(obj_uuid)
        combined_score = _metadata_score(obj.metadata)
        all_results.append(
            _weaviate_to_source(
                obj,
                combined_score,
                vector_score=vector_scores.get(obj_uuid, 0.0),
                bm25_score=bm25_scores.get(obj_uuid, 0.0),
                combined_score=combined_score,
                hybrid_score=combined_score,
            )
        )
        added_count += 1
    return added_count


def _append_dense_sources(*, objects: list, seen_ids: set[str], all_results: list[RAGSource]) -> int:
    added_count = 0
    for obj in objects:
        obj_uuid = str(obj.uuid)
        if obj_uuid in seen_ids:
            continue
        seen_ids.add(obj_uuid)
        vector_score = _vector_score_from_metadata(obj.metadata)
        all_results.append(
            _weaviate_to_source(
                obj,
                vector_score,
                vector_score=vector_score,
                bm25_score=0.0,
                combined_score=vector_score,
                hybrid_score=0.0,
            )
        )
        added_count += 1
    return added_count


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
            trace.add_step(
                "embedding",
                details={
                    "skipped": True,
                    "reason": "retrieval_cache_hit",
                    "query_length": len(query or ""),
                    "model": getattr(settings, "embedding_model", ""),
                },
            )
            trace.add_step(
                "vector_search",
                details={
                    "skipped": True,
                    "reason": "retrieval_cache_hit",
                    "top_k": top_k,
                    "returned": len(results),
                },
            )
            trace.add_step(
                "search",
                details={
                    "retrieved_count": len(results),
                    "cache_hit": True,
                    "top_results": _summarize_sources(results, preview_chars=180),
                },
            )
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
    relaxed_where = _build_relaxed_where_filter(org_id)

    all_results: list[RAGSource] = []
    seen_ids: set[str] = set()
    fallback_attempts: list[dict[str, Any]] = []

    for q in queries:
        query_vector, embedding_cache_hit, embedding_duration_ms = _embed_query(q)
        logger.debug(
            "Hybrid search embedding ready org_id=%s query=%r query_length=%s vector_dims=%s",
            org_id,
            format_log_text(q, 300),
            len(q or ""),
            len(query_vector),
        )
        if trace:
            trace.add_step(
                "embedding",
                duration_ms=embedding_duration_ms,
                details={
                    "query": format_log_text(q, 500),
                    "query_length": len(q or ""),
                    "vector_dims": len(query_vector),
                    "model": getattr(settings, "embedding_model", ""),
                    "cache_hit": embedding_cache_hit,
                },
            )

        query_start = time.monotonic()
        objects = _query_hybrid_objects(
            collection,
            query=q,
            query_vector=query_vector,
            where=where,
            limit=top_k,
        )
        vector_search_duration_ms = (time.monotonic() - query_start) * 1000
        dense_objects = _query_dense_objects(collection, query_vector=query_vector, where=where, limit=top_k)
        bm25_objects = _query_bm25_objects(collection, query=q, where=where, limit=top_k)
        metadata_terms = _metadata_filter_terms(q) if getattr(settings, "metadata_retrieval", True) else []
        metadata_where = _metadata_where_filter(where, metadata_terms)
        metadata_objects = (
            _query_metadata_objects(collection, where=metadata_where, limit=top_k)
            if metadata_where is not None
            else []
        )
        hybrid_scores = _score_map(objects)
        vector_scores = _vector_score_map(dense_objects)
        bm25_scores = _score_map(bm25_objects)
        metadata_weight = float(getattr(settings, "metadata_score_weight", 0.2) or 0.2)
        metadata_scores = {str(obj.uuid): metadata_weight for obj in metadata_objects}
        objects_by_id = {
            str(obj.uuid): obj
            for obj in [*objects, *dense_objects, *bm25_objects, *metadata_objects]
        }
        fused_objects = list(objects_by_id.values())
        normalized_vector = _normalized_score_map(vector_scores)
        normalized_bm25 = _normalized_score_map(bm25_scores)
        normalized_metadata = _normalized_score_map(metadata_scores)
        normalized_hybrid = _normalized_score_map(hybrid_scores)
        top_results = [
            _weaviate_to_source(
                obj,
                _fused_score_for_uuid(
                    str(obj.uuid),
                    hybrid_scores=hybrid_scores,
                    vector_scores=vector_scores,
                    bm25_scores=bm25_scores,
                    metadata_scores=metadata_scores,
                    normalized_hybrid=normalized_hybrid,
                    normalized_vector=normalized_vector,
                    normalized_bm25=normalized_bm25,
                    normalized_metadata=normalized_metadata,
                ),
                vector_score=vector_scores.get(str(obj.uuid), 0.0),
                bm25_score=bm25_scores.get(str(obj.uuid), 0.0),
                combined_score=_fused_score_for_uuid(
                    str(obj.uuid),
                    hybrid_scores=hybrid_scores,
                    vector_scores=vector_scores,
                    bm25_scores=bm25_scores,
                    metadata_scores=metadata_scores,
                    normalized_hybrid=normalized_hybrid,
                    normalized_vector=normalized_vector,
                    normalized_bm25=normalized_bm25,
                    normalized_metadata=normalized_metadata,
                ),
                metadata_score=metadata_scores.get(str(obj.uuid), 0.0),
                hybrid_score=hybrid_scores.get(str(obj.uuid), 0.0),
            )
            for obj in fused_objects[:5]
        ]
        logger.info(
            "Hybrid search Weaviate query complete org_id=%s query=%r query_length=%s returned=%s "
            "dense_returned=%s bm25_returned=%s metadata_returned=%s duration_ms=%.2f top_results=%s",
            org_id,
            format_log_text(q, 300),
            len(q or ""),
            len(objects),
            len(dense_objects),
            len(bm25_objects),
            len(metadata_objects),
            vector_search_duration_ms,
            _summarize_sources(top_results),
        )
        if trace:
            trace.add_step(
                "vector_search",
                duration_ms=vector_search_duration_ms,
                details={
                    "query": format_log_text(q, 500),
                    "collection": COLLECTION_NAME,
                    "top_k": top_k,
                    "alpha": 0.5,
                    "returned": len(fused_objects),
                    "mode": "multi_index",
                    "hybrid_returned": len(objects),
                    "dense_returned": len(dense_objects),
                    "bm25_returned": len(bm25_objects),
                    "metadata_returned": len(metadata_objects),
                    "metadata_terms": metadata_terms,
                    "filter": {
                        "org_id": org_id,
                        "status": "ready",
                        "kb_ids": kb_ids,
                    },
                    "top_results": _summarize_sources(top_results, preview_chars=180),
                },
            )
        _append_multi_index_sources(
            objects=fused_objects,
            hybrid_scores=hybrid_scores,
            vector_scores=vector_scores,
            bm25_scores=bm25_scores,
            metadata_scores=metadata_scores,
            seen_ids=seen_ids,
            all_results=all_results,
        )

    if not all_results:
        fallback_limit = _fallback_top_k(top_k)
        for q in queries:
            query_vector, embedding_cache_hit, embedding_duration_ms = _embed_query(q)
            if trace:
                trace.add_step(
                    "embedding",
                    duration_ms=embedding_duration_ms,
                    details={
                        "query": format_log_text(q, 500),
                        "query_length": len(q or ""),
                        "vector_dims": len(query_vector),
                        "model": getattr(settings, "embedding_model", ""),
                        "cache_hit": embedding_cache_hit,
                        "fallback": True,
                    },
                )

            fallback_specs = []
            if kb_ids:
                fallback_specs.append(("relaxed_filter_hybrid", "hybrid", relaxed_where, top_k))
            fallback_specs.extend([
                ("dense_only", "dense", where, top_k),
                ("dense_only_expanded_top_k", "dense", where, fallback_limit),
            ])
            for attempt_name, mode, attempt_where, limit in fallback_specs:
                query_start = time.monotonic()
                if mode == "hybrid":
                    objects = _query_hybrid_objects(
                        collection,
                        query=q,
                        query_vector=query_vector,
                        where=attempt_where,
                        limit=limit,
                    )
                    vector_scores, bm25_scores = _component_score_maps(
                        collection,
                        query=q,
                        query_vector=query_vector,
                        where=attempt_where,
                        limit=max(limit, len(objects)),
                    ) if objects else ({}, {})
                    added_count = _append_hybrid_sources(
                        objects=objects,
                        vector_scores=vector_scores,
                        bm25_scores=bm25_scores,
                        seen_ids=seen_ids,
                        all_results=all_results,
                    )
                else:
                    objects = _query_dense_objects(
                        collection,
                        query_vector=query_vector,
                        where=attempt_where,
                        limit=limit,
                    )
                    added_count = _append_dense_sources(
                        objects=objects,
                        seen_ids=seen_ids,
                        all_results=all_results,
                    )
                attempt_duration_ms = (time.monotonic() - query_start) * 1000
                filter_kb_ids = [] if attempt_where is relaxed_where else kb_ids
                attempt_details = {
                    "attempt": attempt_name,
                    "mode": mode,
                    "top_k": limit,
                    "returned": len(objects),
                    "added_count": added_count,
                    "filter": {
                        "org_id": org_id,
                        "status": "ready",
                        "kb_ids": filter_kb_ids,
                    },
                    "duration_ms": round(attempt_duration_ms, 2),
                    "top_results": _summarize_sources(all_results, preview_chars=180),
                }
                fallback_attempts.append(attempt_details)
                logger.info(
                    "Hybrid search fallback attempt complete org_id=%s query=%r attempt=%s mode=%s "
                    "top_k=%s returned=%s added=%s duration_ms=%.2f",
                    org_id,
                    format_log_text(q, 300),
                    attempt_name,
                    mode,
                    limit,
                    len(objects),
                    added_count,
                    attempt_duration_ms,
                )
                if trace:
                    trace.add_step("search_fallback", details=attempt_details)
                if all_results:
                    break
            if all_results:
                break

    results = all_results[:top_k]
    _retrieval_cache.set(
        retrieval_cache_key,
        _clone_sources(results),
        getattr(settings, "retrieval_cache_ttl", 0),
    )
    duration_ms = (time.monotonic() - t0) * 1000
    if trace:
        trace.add_step(
            "search",
            duration_ms=duration_ms,
            details={
                "retrieved_count": len(results),
                "expanded_count": len(queries),
                "fallback_attempts": fallback_attempts,
                "top_results": _summarize_sources(results, preview_chars=180),
            },
        )
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


def retrieve_sources(
    query: str,
    org_id: str,
    kb_ids: list[str],
    top_k: int = settings.rag_top_k,
    expand_query: bool = True,
    top_n: int | None = None,
    trace: object = None,
    feedback_weights: object = None,
) -> list[RAGSource]:
    """Run the shared retrieval path: optional expansion -> hybrid search -> rerank."""
    retrieval_top_k = max(top_k, int(getattr(settings, "rag_top_k", top_k) or top_k))
    plan = build_query_plan(query)
    if trace:
        trace.add_step("planner", details=plan.to_dict())
    sources = _retrieve_plan_sources(
        plan=plan,
        query=query,
        org_id=org_id,
        kb_ids=kb_ids,
        top_k=retrieval_top_k,
        expand_query=expand_query,
        trace=trace,
    )
    for idx, source in enumerate(sources, start=1):
        source.rank_before = idx
    reranked = rerank_sources(query, sources, top_n=top_n or top_k, trace=trace)
    if feedback_weights is not None:
        try:
            from app.services.feedback_learning import apply_feedback_weights

            reranked = apply_feedback_weights(reranked, feedback_weights)
            if trace:
                trace.add_step(
                    "feedback_rerank",
                    details={
                        "sample_count": getattr(feedback_weights, "sample_count", 0),
                        "source_count": len(reranked),
                        "top_results": _summarize_sources(reranked, preview_chars=180),
                    },
                )
        except Exception as exc:
            logger.warning("Feedback rerank failed; preserving reranker order: %s", exc, exc_info=True)
    for idx, source in enumerate(reranked, start=1):
        source.rank_after = idx
        source.rerank_score = source.score
    expanded = expand_parent_child_context(reranked, trace=trace)
    _write_retrieval_hits_to_clickhouse(trace, expanded)
    return expanded


def _retrieve_plan_sources(
    *,
    plan: QueryPlan,
    query: str,
    org_id: str,
    kb_ids: list[str],
    top_k: int,
    expand_query: bool,
    trace: object = None,
) -> list[RAGSource]:
    plan_queries = plan.queries or [query]
    if len(plan_queries) == 1:
        return hybrid_search(
            query=plan_queries[0],
            org_id=org_id,
            kb_ids=kb_ids,
            top_k=top_k,
            expand_query=expand_query,
            trace=trace,
        )

    merged: dict[str, RAGSource] = {}
    per_query_top_k = max(top_k, int((top_k + len(plan_queries) - 1) / len(plan_queries)))
    for idx, item in enumerate(plan_queries, start=1):
        sub_sources = hybrid_search(
            query=item,
            org_id=org_id,
            kb_ids=kb_ids,
            top_k=per_query_top_k,
            expand_query=expand_query,
            trace=trace,
        )
        for source in sub_sources:
            existing = merged.get(source.chunk_id)
            if existing is None or source.combined_score > existing.combined_score:
                source.plan_query = item
                source.plan_query_index = idx
                merged[source.chunk_id] = source

    sources = sorted(merged.values(), key=lambda source: source.combined_score, reverse=True)[:top_k]
    if trace:
        trace.add_step(
            "planner_merge",
            details={
                "query_count": len(plan_queries),
                "merged_count": len(merged),
                "returned": len(sources),
                "top_results": _summarize_sources(sources, preview_chars=180),
            },
        )
    return sources


def expand_parent_child_context(sources: list[RAGSource], trace: object = None) -> list[RAGSource]:
    """Attach adjacent chunk previews from PostgreSQL chunk metadata to each source."""
    window = int(getattr(settings, "rag_parent_context_window", 1) or 0)
    if window <= 0 or not sources:
        return sources

    try:
        import asyncio
        from sqlalchemy import select

        from app.database import async_session
        from app.models.chunk import DocumentChunk

        async def _load_neighbors() -> dict[str, list[DocumentChunk]]:
            result: dict[str, list[DocumentChunk]] = {}
            async with async_session() as session:
                for source in sources:
                    if source.chunk_index is None:
                        continue
                    try:
                        document_id = uuid.UUID(str(source.document_id))
                    except ValueError:
                        continue
                    stmt = (
                        select(DocumentChunk)
                        .where(
                            DocumentChunk.document_id == document_id,
                            DocumentChunk.chunk_index >= max(int(source.chunk_index) - window, 0),
                            DocumentChunk.chunk_index <= int(source.chunk_index) + window,
                        )
                        .order_by(DocumentChunk.chunk_index.asc())
                    )
                    rows = (await session.execute(stmt)).scalars().all()
                    result[source.chunk_id] = list(rows)
            return result

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            neighbors_by_chunk = asyncio.run(_load_neighbors())
        else:
            neighbors_by_chunk = _run_async_in_thread(_load_neighbors)

        expanded_count = 0
        for source in sources:
            neighbors = neighbors_by_chunk.get(source.chunk_id, [])
            if len(neighbors) <= 1:
                continue
            parts = []
            seen = set()
            for row in neighbors:
                text = str(getattr(row, "content_preview", "") or "").strip()
                if not text or text in seen:
                    continue
                seen.add(text)
                marker = "当前块" if str(getattr(row, "weaviate_id", "")) == source.chunk_id else f"相邻块 {row.chunk_index}"
                parts.append(f"{marker}: {text}")
            if parts:
                source.content = "\n".join(parts)
                expanded_count += 1
        if trace:
            trace.add_step(
                "parent_child_context",
                details={
                    "window": window,
                    "input_count": len(sources),
                    "expanded_count": expanded_count,
                },
            )
        return sources
    except Exception as exc:
        logger.warning("Parent-child context expansion failed: %s", exc, exc_info=True)
        return sources


def _run_async_in_thread(async_factory):
    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(lambda: asyncio.run(async_factory())).result()


def _write_retrieval_hits_to_clickhouse(trace: object, sources: list[RAGSource]) -> None:
    if not trace or not getattr(settings, "enable_trace_logging", True):
        return

    trace_id = str(getattr(trace, "trace_id", "") or "")
    org_id = str(getattr(trace, "org_id", "") or "")
    if not trace_id or not _is_uuid_like(org_id):
        return

    query_hash = ""
    get_query_hash = getattr(trace, "get_query_hash", None)
    if callable(get_query_hash):
        query_hash = str(get_query_hash())

    events = []
    for idx, source in enumerate(sources, start=1):
        chunk_id = str(source.chunk_id or "")
        document_id = str(source.document_id or "")
        if not (_is_uuid_like(chunk_id) and _is_uuid_like(document_id)):
            continue
        rank_after = source.rank_after or idx
        events.append(
            _make_retrieval_hit_event(
                trace_id=trace_id,
                org_id=org_id,
                query_hash=query_hash,
                source=source,
                rank_after=rank_after,
            )
        )
    if not events:
        return

    try:
        import asyncio

        from app.services.clickhouse import clickhouse_client

        def async_factory():
            return clickhouse_client.write_retrieval_hits(events)

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            ok = asyncio.run(async_factory())
        else:
            ok = _run_async_in_thread(async_factory)
        if trace:
            trace.add_step(
                "retrieval_hit_events",
                details={
                    "attempted_count": len(events),
                    "written": bool(ok),
                },
            )
    except Exception as exc:
        logger.warning("Failed to write retrieval hit events to ClickHouse: %s", exc)


def _make_retrieval_hit_event(
    *,
    trace_id: str,
    org_id: str,
    query_hash: str,
    source: RAGSource,
    rank_after: int,
):
    from app.services.clickhouse import RetrievalHitEvent

    return RetrievalHitEvent(
        trace_id=trace_id,
        org_id=org_id,
        query_hash=query_hash,
        chunk_id=str(source.chunk_id),
        document_id=str(source.document_id),
        rank_before=max(int(source.rank_before or 0), 0),
        rank_after=max(int(rank_after or 0), 0),
        vector_score=_safe_score(source.vector_score),
        bm25_score=_safe_score(source.bm25_score),
        rerank_score=_safe_score(source.rerank_score or source.score),
    )


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
        if trace:
            trace.add_step(
                "rerank",
                details={
                    "skipped": True,
                    "reason": "no_sources",
                    "input_count": 0,
                    "reranked_count": 0,
                },
            )
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
    try:
        results = reranker.rerank(query, documents)
    except Exception as exc:
        logger.warning(
            "Rerank failed; preserving hybrid order reranker=%s input_count=%s error=%s",
            reranker.__class__.__name__,
            len(sources),
            exc,
            exc_info=True,
        )
        if trace:
            trace.add_step(
                "rerank",
                duration_ms=(time.monotonic() - t0) * 1000,
                details={
                    "reranker": reranker.__class__.__name__,
                    "input_count": len(sources),
                    "reranked_count": min(len(sources), top_n),
                    "fallback": True,
                    "error": str(exc)[:500],
                },
            )
        return sources[:top_n]

    reranked = []
    for r in results[:top_n]:
        if r.index < 0 or r.index >= len(sources):
            logger.warning("Rerank result index out of range index=%s source_count=%s", r.index, len(sources))
            continue
        source = sources[r.index]
        source.score = r.score
        reranked.append(source)

    duration_ms = (time.monotonic() - t0) * 1000
    if trace:
        trace.add_step(
            "rerank",
            duration_ms=duration_ms,
            details={
                "reranker": reranker.__class__.__name__,
                "input_count": len(sources),
                "reranked_count": len(reranked),
                "top_results": _summarize_sources(reranked, preview_chars=180),
            },
        )
    logger.info(
        "Rerank complete reranker=%s input_count=%s returned=%s duration_ms=%.2f top_results=%s",
        reranker.__class__.__name__,
        len(sources),
        len(reranked),
        duration_ms,
        _summarize_sources(reranked),
    )
    return reranked


def build_context(
    sources: list[RAGSource],
    max_tokens: int | None = None,
    trace: object = None,
) -> tuple[str, list[dict]]:
    """Build context string from sources and return citation info."""
    t0 = time.monotonic()
    max_tokens = max_tokens or _effective_context_budget_tokens()
    context_parts = []
    citations = []
    used_tokens = 0
    skipped_count = 0

    for source in sources:
        next_idx = len(citations) + 1
        part = (
            f"[{next_idx}] {source.document_title}"
            + (f" - {source.section_path}" if source.section_path else "")
            + (f" [{source.document_type}]" if source.document_type != "general" else "")
            + f"\n{source.content}\n"
        )
        part_tokens = count_tokens(part)
        if citations and used_tokens + part_tokens > max_tokens:
            skipped_count += 1
            continue
        if not citations and part_tokens > max_tokens:
            part = _trim_context_part(
                prefix=(
                    f"[{next_idx}] {source.document_title}"
                    + (f" - {source.section_path}" if source.section_path else "")
                    + (f" [{source.document_type}]" if source.document_type != "general" else "")
                    + "\n"
                ),
                content=source.content,
                max_tokens=max_tokens,
            )
            part_tokens = count_tokens(part)
        context_parts.append(part)
        used_tokens += part_tokens
        citations.append(source.to_dict())

    context = "\n---\n".join(context_parts)
    if trace:
        trace.add_step(
            "context_budget",
            details={
                "max_tokens": max_tokens,
                "used_tokens": count_tokens(context),
                "input_sources": len(sources),
                "selected_sources": len(citations),
                "skipped_sources": skipped_count,
            },
        )
    logger.info(
        "Build context complete source_count=%s citation_count=%s context_length=%s context_tokens=%s "
        "max_tokens=%s skipped_sources=%s duration_ms=%.2f citations=%s",
        len(sources),
        len(citations),
        len(context),
        count_tokens(context),
        max_tokens,
        skipped_count,
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


def _trim_context_part(prefix: str, content: str, max_tokens: int) -> str:
    suffix = "\n"
    allowed_content_tokens = max(max_tokens - count_tokens(prefix) - count_tokens(suffix), 0)
    if allowed_content_tokens <= 0:
        return prefix.rstrip()
    encoded = _TOKEN_ENCODER.encode(content or "")
    trimmed_content = _TOKEN_ENCODER.decode(encoded[:allowed_content_tokens])
    return f"{prefix}{trimmed_content}{suffix}"


def _build_history_aware_query(query: str, messages: list[dict]) -> str:
    return _rewrite_history_aware_query(query, messages).rewritten


def _rewrite_history_aware_query(query: str, messages: list[dict]) -> ConversationalQueryRewriteResult:
    return rewrite_conversational_query(query, messages)


def assemble_context_and_generate(
    query: str,
    org_id: str,
    kb_ids: list[str],
    max_chunks: int = settings.rag_max_chunks,
    user_id: str = "",
    messages: list[dict] | None = None,
    feedback_weights: object = None,
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

    # Step 1: Shared retrieval path with optional query expansion and rerank
    rewrite_result = _rewrite_history_aware_query(query, messages or [])
    retrieval_query = rewrite_result.rewritten
    logger.info(
        "RAG retrieval query prepared trace_id=%s uses_history=%s query_rewrite_reason=%s uses_llm=%s "
        "retrieval_query=%r retrieval_query_length=%s",
        trace_id,
        retrieval_query != query,
        rewrite_result.reason,
        rewrite_result.used_llm,
        format_log_text(retrieval_query, 700),
        len(retrieval_query or ""),
    )
    if trace and retrieval_query != query:
        trace.add_step(
            "query_rewrite",
            details={
                "used_llm": rewrite_result.used_llm,
                "reason": rewrite_result.reason,
                "original_length": len(query or ""),
                "rewritten_length": len(retrieval_query or ""),
                "rewritten_preview": format_log_text(retrieval_query, 500),
            },
        )
    sources = retrieve_sources(
        retrieval_query,
        org_id,
        kb_ids,
        top_k=max(max_chunks, getattr(settings, "reranker_top_n", max_chunks)),
        expand_query=True,
        trace=trace,
        feedback_weights=feedback_weights,
    )
    logger.info(
        "RAG retrieval complete trace_id=%s source_count=%s top_results=%s",
        trace_id,
        len(sources),
        _summarize_sources(sources),
    )

    # Step 2: Keep an upper bound before context token budgeting.
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

    # Step 3: Compress retrieved context before prompt assembly.
    sources, compression_stats = compress_sources_for_query(retrieval_query, sources)
    if trace:
        trace.add_step(
            "contextual_compression",
            details={
                "input_count": compression_stats.input_count,
                "compressed_count": compression_stats.compressed_count,
                "original_chars": compression_stats.original_chars,
                "compressed_chars": compression_stats.compressed_chars,
            },
        )

    # Step 4: Build context
    context, citations = build_context(sources, trace=trace)
    if trace:
        trace.add_step(
            "context",
            details={
                "source_count": len(sources),
                "citation_count": len(citations),
                "context_length": len(context),
                "sources": _summarize_sources(sources, preview_chars=180),
            },
        )

    if not context.strip():
        trace.total_latency_ms = (time.monotonic() - t_start) * 1000
        trace.answer_preview = "No relevant context found; answer generation skipped."
        trace.answer_length = len(trace.answer_preview)
        trace.add_step(
            "llm_generation",
            details={
                "skipped": True,
                "reason": "no_context",
                "model": settings.llm_model,
            },
        )
        trace.add_step(
            "answer",
            details={
                "answer_length": trace.answer_length,
                "answer_preview": trace.answer_preview,
                "reason": "no_context",
            },
        )
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
    citation_validation = validate_answer_citations(accumulated, citations)
    final_answer = citation_validation.answer
    if trace:
        trace.model = settings.llm_model
        trace.prompt_version = "v1"
        trace.answer_preview = format_log_text(final_answer, 1000)
        trace.answer_length = len(final_answer)
        trace.add_step(
            "llm_generation",
            duration_ms=gen_duration_ms,
            details={
                "model": settings.llm_model,
                "prompt_version": "v1",
                "context_length": len(context),
                "source_count": len(sources),
                "chunk_count": chunk_count,
                "answer_length": len(final_answer),
            },
        )
        trace.add_step(
            "citation",
            details={
                "citation_count": citation_validation.citation_count,
                "used_citation_numbers": citation_validation.used_citation_numbers,
                "invalid_citation_numbers": citation_validation.invalid_citation_numbers,
                "low_confidence_citation_numbers": citation_validation.low_confidence_citation_numbers,
                "is_valid": citation_validation.is_valid,
            },
        )
        trace.add_step(
            "answer",
            details={
                "answer_length": len(final_answer),
                "answer_preview": trace.answer_preview,
                "source_count": len(sources),
            },
        )
    logger.info(
        "RAG generation complete trace_id=%s chunks=%s answer_length=%s citation_valid=%s duration_ms=%.2f",
        trace_id,
        chunk_count,
        len(final_answer),
        citation_validation.is_valid,
        gen_duration_ms,
    )

    trace.total_latency_ms = (time.monotonic() - t_start) * 1000
    logger.info(
        "RAG pipeline complete trace_id=%s org_id=%s user_id=%s sources=%s answer_length=%s duration_ms=%.2f",
        trace_id,
        org_id,
        user_id or "anonymous",
        len(sources),
        len(final_answer),
        trace.total_latency_ms,
    )
    _write_trace_to_clickhouse(trace)

    yield {
        "delta": final_answer[len(accumulated):],
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
