"""Tests for Search API endpoint."""
import uuid
from unittest.mock import patch

from app.services.rag import RAGSource


def test_search_with_results(test_client):
    doc_id = str(uuid.uuid4())
    source = RAGSource(
        chunk_id="c1",
        document_id=doc_id,
        document_title="Test Document",
        section_path="Methods",
        page_start=5,
        page_end=7,
        score=0.85,
        content_preview="The methods section describes...",
    )

    with patch("app.api.v1.search.retrieve_sources") as mock_search:
        mock_search.return_value = [source]

        resp = test_client.post(
            "/api/v1/search",
            json={"query": "test query", "top_k": 5},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["query"] == "test query"
    assert data["total"] == 1
    assert len(data["results"]) == 1
    assert data["results"][0]["document_title"] == "Test Document"
    assert data["results"][0]["section_path"] == "Methods"


def test_search_empty_results(test_client):
    with patch("app.api.v1.search.retrieve_sources") as mock_search:
        mock_search.return_value = []

        resp = test_client.post(
            "/api/v1/search",
            json={"query": "no results query"},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["query"] == "no results query"
    assert data["total"] == 0
    assert data["results"] == []


def test_search_with_kb_filter(test_client):
    kb_id = str(uuid.uuid4())
    with patch("app.api.v1.search.retrieve_sources") as mock_search:
        mock_search.return_value = []

        resp = test_client.post(
            "/api/v1/search",
            json={"query": "test", "kb_ids": [kb_id], "top_k": 10},
        )

    assert resp.status_code == 200
    call_args = mock_search.call_args
    assert kb_id in call_args[1]["kb_ids"]
    assert call_args[1]["expand_query"] is True


def test_search_unauthorized():
    """Test without auth override - returns 422 for missing header."""
    with patch("app.main.ensure_collection"):
        from app.main import app
        from fastapi.testclient import TestClient
        from app.api.deps import get_current_user

        saved = app.dependency_overrides.get(get_current_user)
        app.dependency_overrides.pop(get_current_user, None)

        try:
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.post(
                "/api/v1/search",
                json={"query": "test"},
            )
            assert resp.status_code in (401, 422)
        finally:
            if saved:
                app.dependency_overrides[get_current_user] = saved


def test_search_empty_query(test_client):
    resp = test_client.post(
        "/api/v1/search",
        json={"query": ""},
    )
    assert resp.status_code == 422
