"""Tests for ClickHouse analytics service."""
import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from app.services.clickhouse import (
    EMPTY_ANALYTICS_SUMMARY,
    RAGTraceEvent,
    RetrievalHitEvent,
    ClickHouseClient,
    _escape,
)


class TestRAGTraceEvent:
    def test_defaults(self):
        event = RAGTraceEvent(trace_id="t1", org_id="o1", user_id="u1")
        assert event.scenario == "qa"
        assert event.kb_ids == []
        assert event.safety_flags == []
        assert event.rating is None

    def test_with_data(self):
        event = RAGTraceEvent(
            trace_id="t1", org_id="o1", user_id="u1",
            query_text="test query",
            kb_ids=["kb1", "kb2"],
            retrieved_count=10,
            reranked_count=5,
            latency_ms=500,
            safety_flags=["medical"],
        )
        assert event.query_text == "test query"
        assert len(event.kb_ids) == 2
        assert "medical" in event.safety_flags


class TestRetrievalHitEvent:
    def test_defaults(self):
        event = RetrievalHitEvent(trace_id="t1", org_id="o1")
        assert event.vector_score == 0.0
        assert event.clicked is False
        assert event.cited is False


def _make_async_context_mock():
    """Create a properly configured async context manager mock for httpx.AsyncClient."""
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    return mock_client


class TestClickHouseClient:
    def test_uses_connection_settings(self):
        client = ClickHouseClient(
            url="http://clickhouse:8123",
            user="default",
            password="secret",
            database="rag",
        )

        assert client.url == "http://clickhouse:8123"
        assert client._auth == ("default", "secret")
        assert client._params("SELECT 1") == {
            "query": "SELECT 1",
            "database": "rag",
        }

    @pytest.mark.asyncio
    async def test_write_trace_event_success(self):
        """Test successful trace event write."""
        mock_client = _make_async_context_mock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_client.post = AsyncMock(return_value=mock_resp)

        client = ClickHouseClient(
            url="http://localhost:8123",
            user="default",
            password="secret",
            database="rag",
        )
        event = RAGTraceEvent(
            trace_id="t1", org_id="o1", user_id="u1",
            query_text="test",
        )
        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await client.write_trace_event(event)
        assert result is True
        assert mock_client.post.call_args.kwargs["params"]["database"] == "rag"
        assert mock_client.post.call_args.kwargs["params"]["default_format"] == "Values"

    @pytest.mark.asyncio
    async def test_write_trace_event_failure(self):
        """Test trace event write failure returns False."""
        mock_client = _make_async_context_mock()
        mock_client.post.side_effect = Exception("fail")

        client = ClickHouseClient(url="http://localhost:8123")
        event = RAGTraceEvent(trace_id="t1", org_id="o1", user_id="u1")
        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await client.write_trace_event(event)
        assert result is False

    @pytest.mark.asyncio
    async def test_write_retrieval_hit(self):
        """Test retrieval hit write."""
        mock_client = _make_async_context_mock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_client.post = AsyncMock(return_value=mock_resp)

        client = ClickHouseClient(url="http://localhost:8123")
        event = RetrievalHitEvent(
            trace_id="t1", org_id="o1",
            chunk_id="c1", document_id="d1",
            vector_score=0.85,
            cited=True,
        )
        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await client.write_retrieval_hit(event)
        assert result is True

    @pytest.mark.asyncio
    async def test_get_zero_result_queries(self):
        """Test getting zero-result queries."""
        mock_client = _make_async_context_mock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "data": [
                {"query_text": "test", "org_id": "o1", "user_id": "u1", "cnt": 5},
            ]
        }
        mock_client.post = AsyncMock(return_value=mock_resp)

        client = ClickHouseClient(url="http://localhost:8123")
        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await client.get_zero_result_queries("o1")
        assert len(result) == 1
        assert result[0]["query_text"] == "test"

    @pytest.mark.asyncio
    async def test_get_zero_result_queries_error(self):
        """Test error handling for zero-result queries."""
        mock_client = _make_async_context_mock()
        mock_client.post.side_effect = Exception("fail")

        client = ClickHouseClient(url="http://localhost:8123")
        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await client.get_zero_result_queries("o1")
        assert result == []

    @pytest.mark.asyncio
    async def test_get_low_rated_answers(self):
        """Test getting low-rated answers."""
        mock_client = _make_async_context_mock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "data": [
                {"trace_id": "t1", "query_text": "test", "rating": 1},
            ]
        }
        mock_client.post = AsyncMock(return_value=mock_resp)

        client = ClickHouseClient(url="http://localhost:8123")
        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await client.get_low_rated_answers("o1")
        assert len(result) == 1
        assert result[0]["rating"] == 1

    @pytest.mark.asyncio
    async def test_get_analytics_summary(self):
        """Test getting analytics summary."""
        mock_client = _make_async_context_mock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "data": [{
                "total_queries": 100,
                "avg_latency_ms": 500.5,
                "avg_rating": 3.5,
                "avg_retrieved_count": 8.0,
                "avg_reranked_count": 5.0,
                "zero_results": 10,
                "low_ratings": 5,
            }]
        }
        mock_client.post = AsyncMock(return_value=mock_resp)

        client = ClickHouseClient(url="http://localhost:8123")
        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await client.get_analytics_summary("o1")
        assert result["total_queries"] == 100
        assert result["avg_latency_ms"] == 500.5

    @pytest.mark.asyncio
    async def test_get_analytics_summary_empty(self):
        """Test analytics summary with no data."""
        mock_client = _make_async_context_mock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": []}
        mock_client.post = AsyncMock(return_value=mock_resp)

        client = ClickHouseClient(url="http://localhost:8123")
        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await client.get_analytics_summary("o1")
        assert result == EMPTY_ANALYTICS_SUMMARY

    @pytest.mark.asyncio
    async def test_get_analytics_summary_error_returns_empty_summary(self):
        """Test analytics summary returns a valid zero summary on query failure."""
        mock_client = _make_async_context_mock()
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_resp.text = "Unknown table"
        mock_client.post = AsyncMock(return_value=mock_resp)

        client = ClickHouseClient(url="http://localhost:8123")
        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await client.get_analytics_summary("o1")
        assert result == EMPTY_ANALYTICS_SUMMARY

    @pytest.mark.asyncio
    async def test_update_trace_rating(self):
        """Test updating a trace rating."""
        mock_client = _make_async_context_mock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_client.post = AsyncMock(return_value=mock_resp)

        client = ClickHouseClient(url="http://localhost:8123")
        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await client.update_trace_rating("o1", "t1", 2)
        assert result is True
        assert "ALTER TABLE rag_trace_events" in mock_client.post.call_args.kwargs["params"]["query"]


class TestEscape:
    def test_no_special_chars(self):
        assert _escape("hello") == "hello"

    def test_single_quote(self):
        # replace("'", r"\'") then replace("\\", r"\\") => \ becomes \\
        assert _escape("it's") == r"it\\'s"

    def test_backslash(self):
        assert _escape("path\\to") == r"path\\to"

    def test_combined(self):
        # "it's a\\test" = literal: it's a\test
        # After escaping: it\\'s a\\test
        assert _escape("it's a\\test") == r"it\\'s a\\test"
