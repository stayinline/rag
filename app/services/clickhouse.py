"""ClickHouse analytics service for RAG events."""
import logging
import math
from dataclasses import dataclass, field

from app.config import settings

logger = logging.getLogger(__name__)


EMPTY_ANALYTICS_SUMMARY = {
    "total_queries": 0,
    "avg_latency_ms": 0.0,
    "zero_result_rate": 0.0,
    "avg_rating": 0.0,
    "low_rating_rate": 0.0,
    "avg_retrieved_count": 0.0,
    "avg_reranked_count": 0.0,
}


@dataclass
class RAGTraceEvent:
    """RAG trace event for ClickHouse."""
    trace_id: str
    org_id: str
    user_id: str
    scenario: str = "qa"
    query_hash: str = ""
    query_text: str = ""
    kb_ids: list[str] = field(default_factory=list)
    rewrite_count: int = 0
    retrieved_count: int = 0
    reranked_count: int = 0
    source_count: int = 0
    latency_ms: int = 0
    first_token_ms: int = 0
    model: str = ""
    prompt_version: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost_units: float = 0.0
    safety_flags: list[str] = field(default_factory=list)
    rating: int | None = None


@dataclass
class RetrievalHitEvent:
    """Retrieval hit event for ClickHouse."""
    trace_id: str
    org_id: str
    query_hash: str = ""
    chunk_id: str = ""
    document_id: str = ""
    rank_before: int = 0
    rank_after: int = 0
    vector_score: float = 0.0
    bm25_score: float = 0.0
    rerank_score: float = 0.0
    clicked: bool = False
    cited: bool = False


class ClickHouseClient:
    """Async ClickHouse client for RAG analytics."""

    def __init__(
        self,
        url: str | None = None,
        user: str | None = None,
        password: str | None = None,
        database: str | None = None,
    ):
        self.url = url or getattr(settings, "clickhouse_url", "http://localhost:8123")
        self.user = user if user is not None else getattr(settings, "clickhouse_user", "default")
        self.password = password if password is not None else getattr(settings, "clickhouse_password", "")
        self.database = database if database is not None else getattr(settings, "clickhouse_database", "default")

    def _params(self, query: str) -> dict[str, str]:
        params = {"query": query}
        if self.database:
            params["database"] = self.database
        return params

    @property
    def _auth(self) -> tuple[str, str] | None:
        if not self.user:
            return None
        return (self.user, self.password)

    async def write_trace_event(self, event: RAGTraceEvent) -> bool:
        """Write a RAG trace event to ClickHouse."""
        try:
            import httpx
            logger.debug("ClickHouse write trace start trace_id=%s org_id=%s", event.trace_id, event.org_id)
            query = """
                INSERT INTO rag_trace_events
                (event_time, trace_id, org_id, user_id, scenario, query_hash, query_text,
                 kb_ids, rewrite_count, retrieved_count, reranked_count, source_count,
                 latency_ms, first_token_ms, model, prompt_version,
                 input_tokens, output_tokens, cost_units, safety_flags, rating)
                VALUES
            """
            kb_ids_str = ", ".join(f"'{kb}'" for kb in event.kb_ids) if event.kb_ids else ""
            safety_str = ", ".join(f"'{s}'" for s in event.safety_flags) if event.safety_flags else ""
            values = (
                f"now(), '{event.trace_id}', '{event.org_id}', '{event.user_id}', "
                f"'{event.scenario}', '{event.query_hash}', '{_escape(event.query_text)}', "
                f"[{kb_ids_str}], {event.rewrite_count}, {event.retrieved_count}, "
                f"{event.reranked_count}, {event.source_count}, {event.latency_ms}, "
                f"{event.first_token_ms}, '{event.model}', '{event.prompt_version}', "
                f"{event.input_tokens}, {event.output_tokens}, {event.cost_units}, "
                f"[{safety_str}], {event.rating if event.rating else 'NULL'}"
            )
            full_query = f"{query} ({values})"

            async with httpx.AsyncClient(timeout=10, auth=self._auth) as client:
                resp = await client.post(
                    self.url,
                    params={**self._params(full_query), "default_format": "Values"},
                )
                ok = resp.status_code == 200
                if ok:
                    logger.info("ClickHouse write trace complete trace_id=%s org_id=%s", event.trace_id, event.org_id)
                else:
                    logger.warning(
                        "ClickHouse write trace returned non-200 trace_id=%s org_id=%s status_code=%s body_preview=%s",
                        event.trace_id,
                        event.org_id,
                        resp.status_code,
                        resp.text[:300],
                    )
                return ok
        except Exception as e:
            logger.warning("Failed to write trace event to ClickHouse: %s", e)
            return False

    async def write_retrieval_hit(self, event: RetrievalHitEvent) -> bool:
        """Write a retrieval hit event to ClickHouse."""
        try:
            import httpx
            logger.debug("ClickHouse write retrieval hit start trace_id=%s chunk_id=%s", event.trace_id, event.chunk_id)
            values = (
                f"now(), '{event.trace_id}', '{event.org_id}', '{event.query_hash}', "
                f"'{event.chunk_id}', '{event.document_id}', {event.rank_before}, "
                f"{event.rank_after}, {event.vector_score}, {event.bm25_score}, "
                f"{event.rerank_score}, {event.clicked}, {event.cited}"
            )
            query = (
                "INSERT INTO retrieval_hit_events "
                "(event_time, trace_id, org_id, query_hash, chunk_id, document_id, "
                "rank_before, rank_after, vector_score, bm25_score, rerank_score, clicked, cited) "
                f"VALUES ({values})"
            )

            async with httpx.AsyncClient(timeout=10, auth=self._auth) as client:
                resp = await client.post(
                    self.url,
                    params=self._params(query),
                )
                ok = resp.status_code == 200
                if ok:
                    logger.debug("ClickHouse write retrieval hit complete trace_id=%s chunk_id=%s", event.trace_id, event.chunk_id)
                else:
                    logger.warning(
                        "ClickHouse write retrieval hit returned non-200 trace_id=%s chunk_id=%s status_code=%s body_preview=%s",
                        event.trace_id,
                        event.chunk_id,
                        resp.status_code,
                        resp.text[:300],
                    )
                return ok
        except Exception as e:
            logger.warning("Failed to write retrieval hit to ClickHouse: %s", e)
            return False

    async def update_trace_rating(self, org_id: str, trace_id: str, rating: int) -> bool:
        """Update the feedback rating for a stored trace event."""
        try:
            import httpx
            logger.debug("ClickHouse update trace rating start trace_id=%s rating=%s", trace_id, rating)
            query = (
                "ALTER TABLE rag_trace_events "
                f"UPDATE rating = {int(rating)} "
                f"WHERE org_id = '{org_id}' AND trace_id = '{_escape(trace_id)}'"
            )

            async with httpx.AsyncClient(timeout=10, auth=self._auth) as client:
                resp = await client.post(
                    self.url,
                    params={**self._params(query), "mutations_sync": "1"},
                )
                ok = resp.status_code == 200
                if ok:
                    logger.info("ClickHouse update trace rating complete trace_id=%s rating=%s", trace_id, rating)
                else:
                    logger.warning(
                        "ClickHouse update trace rating returned non-200 trace_id=%s status_code=%s body_preview=%s",
                        trace_id,
                        resp.status_code,
                        resp.text[:300],
                    )
                return ok
        except Exception as e:
            logger.warning("Failed to update trace rating in ClickHouse: %s", e)
            return False

    async def get_zero_result_queries(
        self, org_id: str, limit: int = 50
    ) -> list[dict]:
        """Get zero-result queries for an org."""
        try:
            import httpx
            logger.info("ClickHouse query zero-result start org_id=%s limit=%s", org_id, limit)
            query = f"""
                SELECT query_text, org_id, user_id, kb_ids, count() as cnt, max(event_time) as last_seen
                FROM rag_trace_events
                WHERE org_id = '{org_id}' AND retrieved_count = 0
                GROUP BY query_text, org_id, user_id, kb_ids
                ORDER BY cnt DESC, last_seen DESC
                LIMIT {limit}
                FORMAT JSON
            """
            async with httpx.AsyncClient(timeout=10, auth=self._auth) as client:
                resp = await client.post(self.url, params=self._params(query))
                if resp.status_code == 200:
                    data = resp.json()
                    rows = data.get("data", [])
                    logger.info("ClickHouse query zero-result complete org_id=%s returned=%s", org_id, len(rows))
                    return rows
                logger.warning("ClickHouse query zero-result returned non-200 org_id=%s status_code=%s body_preview=%s", org_id, resp.status_code, resp.text[:300])
            return []
        except Exception as e:
            logger.warning("Failed to query zero-result queries: %s", e)
            return []

    async def get_low_rated_answers(
        self, org_id: str, limit: int = 50
    ) -> list[dict]:
        """Get low-rated answers for an org."""
        try:
            import httpx
            logger.info("ClickHouse query low-rated answers start org_id=%s limit=%s", org_id, limit)
            query = f"""
                SELECT trace_id, query_text, rating
                FROM rag_trace_events
                WHERE org_id = '{org_id}' AND rating IS NOT NULL AND rating <= 2
                ORDER BY event_time DESC
                LIMIT {limit}
                FORMAT JSON
            """
            async with httpx.AsyncClient(timeout=10, auth=self._auth) as client:
                resp = await client.post(self.url, params=self._params(query))
                if resp.status_code == 200:
                    data = resp.json()
                    rows = data.get("data", [])
                    logger.info("ClickHouse query low-rated answers complete org_id=%s returned=%s", org_id, len(rows))
                    return rows
                logger.warning("ClickHouse query low-rated answers returned non-200 org_id=%s status_code=%s body_preview=%s", org_id, resp.status_code, resp.text[:300])
            return []
        except Exception as e:
            logger.warning("Failed to query low-rated answers: %s", e)
            return []

    async def get_analytics_summary(self, org_id: str) -> dict:
        """Get RAG analytics summary for an org."""
        try:
            import httpx
            logger.info("ClickHouse analytics summary start org_id=%s", org_id)
            query = f"""
                SELECT
                    count() as total_queries,
                    avg(latency_ms) as avg_latency_ms,
                    avg(rating) as avg_rating,
                    avg(retrieved_count) as avg_retrieved_count,
                    avg(reranked_count) as avg_reranked_count,
                    sumIf(1, retrieved_count = 0) as zero_results,
                    sumIf(1, rating IS NOT NULL AND rating <= 2) as low_ratings
                FROM rag_trace_events
                WHERE org_id = '{org_id}'
                FORMAT JSON
            """
            async with httpx.AsyncClient(timeout=10, auth=self._auth) as client:
                resp = await client.post(self.url, params=self._params(query))
                if resp.status_code == 200:
                    data = resp.json()
                    rows = data.get("data", [])
                    if rows:
                        row = rows[0]
                        total = _safe_int(row.get("total_queries"))
                        zero_results = _safe_int(row.get("zero_results"))
                        low_ratings = _safe_int(row.get("low_ratings"))
                        summary = {
                            "total_queries": total,
                            "avg_latency_ms": round(_safe_float(row.get("avg_latency_ms")), 1),
                            "avg_rating": round(_safe_float(row.get("avg_rating")), 2),
                            "avg_retrieved_count": round(_safe_float(row.get("avg_retrieved_count")), 1),
                            "avg_reranked_count": round(_safe_float(row.get("avg_reranked_count")), 1),
                            "zero_result_rate": round(zero_results / max(total, 1), 4),
                            "low_rating_rate": round(low_ratings / max(total, 1), 4),
                        }
                        logger.info("ClickHouse analytics summary complete org_id=%s total_queries=%s", org_id, total)
                        return summary
                else:
                    logger.warning("ClickHouse analytics summary returned non-200 org_id=%s status_code=%s body_preview=%s", org_id, resp.status_code, resp.text[:300])
            return dict(EMPTY_ANALYTICS_SUMMARY)
        except Exception as e:
            logger.warning("Failed to get analytics summary: %s", e)
            return dict(EMPTY_ANALYTICS_SUMMARY)


def _escape(s: str) -> str:
    """Escape string for SQL."""
    return s.replace("'", r"\'").replace("\\", r"\\")


def _safe_float(value) -> float:
    if value in (None, ""):
        return 0.0
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


def _safe_int(value) -> int:
    return int(_safe_float(value))


# Global client instance
clickhouse_client = ClickHouseClient()
