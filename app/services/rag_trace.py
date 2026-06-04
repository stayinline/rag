"""RAG trace service for capturing the full RAG pipeline execution."""
import hashlib
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class TraceStep:
    """A single step in the RAG trace."""
    step: str  # auth, rewrite, search, rerank, context, generation, citation
    duration_ms: float = 0
    details: dict = field(default_factory=dict)


@dataclass
class RAGTrace:
    """Complete trace of a RAG pipeline execution."""
    trace_id: str
    org_id: str
    user_id: str
    query: str
    kb_ids: list[str] = field(default_factory=list)
    steps: list[TraceStep] = field(default_factory=list)
    total_latency_ms: float = 0
    model: str = ""
    prompt_version: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    error: str | None = None

    def add_step(self, step: str, duration_ms: float = 0, details: dict | None = None):
        """Add a trace step."""
        logger.debug(
            "RAG trace add step trace_id=%s step=%s duration_ms=%.2f details=%s",
            self.trace_id,
            step,
            duration_ms,
            details or {},
        )
        self.steps.append(TraceStep(
            step=step,
            duration_ms=duration_ms,
            details=details or {},
        ))

    def get_query_hash(self) -> str:
        """Generate a hash of the query for deduplication."""
        return hashlib.md5(f"{self.org_id}:{self.query}".encode()).hexdigest()[:16]

    def get_retrieved_count(self) -> int:
        """Get total retrieved count from search step."""
        for s in self.steps:
            if s.step == "search":
                return s.details.get("retrieved_count", 0)
        return 0

    def get_reranked_count(self) -> int:
        """Get reranked count from rerank step."""
        for s in self.steps:
            if s.step == "rerank":
                return s.details.get("reranked_count", 0)
        return 0

    def get_rewrite_count(self) -> int:
        """Get rewrite count from rewrite step."""
        for s in self.steps:
            if s.step == "rewrite":
                return s.details.get("rewrite_count", 0)
        return 0

    def get_source_count(self) -> int:
        """Get source count from context step."""
        for s in self.steps:
            if s.step == "context":
                return s.details.get("source_count", 0)
        return 0

    def to_clickhouse_event(self):
        """Convert to ClickHouse RAGTraceEvent."""
        from app.services.clickhouse import RAGTraceEvent
        return RAGTraceEvent(
            trace_id=self.trace_id,
            org_id=self.org_id,
            user_id=self.user_id,
            query_hash=self.get_query_hash(),
            query_text=self.query[:2000],  # Truncate long queries
            kb_ids=self.kb_ids,
            rewrite_count=self.get_rewrite_count(),
            retrieved_count=self.get_retrieved_count(),
            reranked_count=self.get_reranked_count(),
            source_count=self.get_source_count(),
            latency_ms=int(self.total_latency_ms),
            model=self.model,
            prompt_version=self.prompt_version,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
        )


class TraceCollector:
    """Collects and stores RAG traces."""

    def __init__(self):
        self._traces: dict[str, RAGTrace] = {}

    def start_trace(self, trace_id: str, org_id: str, user_id: str, query: str, kb_ids: list[str]) -> RAGTrace:
        """Start a new trace."""
        logger.info(
            "RAG trace start trace_id=%s org_id=%s user_id=%s kb_count=%s query_length=%s",
            trace_id,
            org_id,
            user_id,
            len(kb_ids),
            len(query or ""),
        )
        trace = RAGTrace(
            trace_id=trace_id,
            org_id=org_id,
            user_id=user_id,
            query=query,
            kb_ids=kb_ids,
        )
        self._traces[str(trace_id)] = trace
        return trace

    def get_trace(self, trace_id: str) -> RAGTrace | None:
        """Get a trace by ID."""
        return self._traces.get(str(trace_id))

    def complete_trace(self, trace_id: str, duration_ms: float):
        """Mark a trace as complete."""
        trace = self._traces.get(str(trace_id))
        if trace:
            trace.total_latency_ms = duration_ms
            logger.info("RAG trace complete trace_id=%s duration_ms=%.2f", trace_id, duration_ms)
        else:
            logger.warning("RAG trace complete skipped trace_id=%s reason=not_found", trace_id)


# Global collector
trace_collector = TraceCollector()
