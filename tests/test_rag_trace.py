"""Tests for RAG trace service."""

from app.services.rag_trace import (
    TraceStep,
    RAGTrace,
    TraceCollector,
)
from app.services.rag_trace_store import trace_steps_to_dicts


class TestTraceStep:
    def test_defaults(self):
        step = TraceStep(step="search")
        assert step.duration_ms == 0
        assert step.details == {}

    def test_with_data(self):
        step = TraceStep(step="rewrite", duration_ms=50.0, details={"rewrite_count": 3})
        assert step.duration_ms == 50.0
        assert step.details["rewrite_count"] == 3


class TestRAGTrace:
    def test_creation(self):
        trace = RAGTrace(
            trace_id="t1",
            org_id="o1",
            user_id="u1",
            query="test query",
            kb_ids=["kb1"],
        )
        assert trace.trace_id == "t1"
        assert trace.total_latency_ms == 0
        assert len(trace.steps) == 0

    def test_add_step(self):
        trace = RAGTrace(trace_id="t1", org_id="o1", user_id="u1", query="q")
        trace.add_step("search", duration_ms=100.0, details={"retrieved_count": 5})
        assert len(trace.steps) == 1
        assert trace.steps[0].step == "search"
        assert trace.steps[0].duration_ms == 100.0

    def test_get_query_hash(self):
        trace = RAGTrace(trace_id="t1", org_id="o1", user_id="u1", query="test")
        hash1 = trace.get_query_hash()
        hash2 = trace.get_query_hash()
        assert hash1 == hash2  # Deterministic
        assert len(hash1) == 16

    def test_get_retrieved_count(self):
        trace = RAGTrace(trace_id="t1", org_id="o1", user_id="u1", query="q")
        trace.add_step("search", details={"retrieved_count": 10})
        assert trace.get_retrieved_count() == 10

    def test_get_retrieved_count_no_search_step(self):
        trace = RAGTrace(trace_id="t1", org_id="o1", user_id="u1", query="q")
        trace.add_step("rewrite", details={"rewrite_count": 2})
        assert trace.get_retrieved_count() == 0

    def test_get_reranked_count(self):
        trace = RAGTrace(trace_id="t1", org_id="o1", user_id="u1", query="q")
        trace.add_step("rerank", details={"reranked_count": 5})
        assert trace.get_reranked_count() == 5

    def test_get_rewrite_count(self):
        trace = RAGTrace(trace_id="t1", org_id="o1", user_id="u1", query="q")
        trace.add_step("rewrite", details={"rewrite_count": 3})
        assert trace.get_rewrite_count() == 3

    def test_get_source_count(self):
        trace = RAGTrace(trace_id="t1", org_id="o1", user_id="u1", query="q")
        trace.add_step("context", details={"source_count": 4})
        assert trace.get_source_count() == 4

    def test_to_clickhouse_event(self):
        trace = RAGTrace(
            trace_id="t1", org_id="o1", user_id="u1",
            query="test query", kb_ids=["kb1"],
            model="qwen-plus",
            prompt_version="v1",
            input_tokens=1000,
            output_tokens=500,
        )
        trace.add_step("rewrite", details={"rewrite_count": 2})
        trace.add_step("search", details={"retrieved_count": 10})
        trace.add_step("rerank", details={"reranked_count": 5})
        trace.add_step("context", details={"source_count": 4})

        event = trace.to_clickhouse_event()
        assert event.trace_id == "t1"
        assert event.org_id == "o1"
        assert event.rewrite_count == 2
        assert event.retrieved_count == 10
        assert event.reranked_count == 5
        assert event.source_count == 4
        assert event.model == "qwen-plus"


class TestTraceCollector:
    def test_start_trace(self):
        collector = TraceCollector()
        trace = collector.start_trace("t1", "o1", "u1", "query", ["kb1"])
        assert trace.trace_id == "t1"
        assert "t1" in collector._traces

    def test_get_trace(self):
        collector = TraceCollector()
        collector.start_trace("t1", "o1", "u1", "query", [])
        trace = collector.get_trace("t1")
        assert trace is not None
        assert trace.trace_id == "t1"

    def test_get_trace_not_found(self):
        collector = TraceCollector()
        trace = collector.get_trace("nonexistent")
        assert trace is None

    def test_complete_trace(self):
        collector = TraceCollector()
        collector.start_trace("t1", "o1", "u1", "query", [])
        collector.complete_trace("t1", duration_ms=500.0)
        trace = collector.get_trace("t1")
        assert trace.total_latency_ms == 500.0

    def test_trace_steps_order(self):
        collector = TraceCollector()
        trace = collector.start_trace("t1", "o1", "u1", "query", [])
        trace.add_step("rewrite")
        trace.add_step("search")
        trace.add_step("rerank")
        trace.add_step("context")
        trace.add_step("generation")

        steps = [s.step for s in trace.steps]
        assert steps == ["rewrite", "search", "rerank", "context", "generation"]

    def test_trace_steps_to_dicts_contains_details(self):
        trace = RAGTrace(trace_id="t1", org_id="o1", user_id="u1", query="q")
        trace.add_step("embedding", duration_ms=12.34, details={"vector_dims": 1536})
        trace.add_step("vector_search", duration_ms=45.6, details={"returned": 3})

        steps = trace_steps_to_dicts(trace)

        assert steps[0]["step"] == "embedding"
        assert steps[0]["duration_ms"] == 12.34
        assert steps[0]["details"]["vector_dims"] == 1536
        assert steps[1]["step"] == "vector_search"
