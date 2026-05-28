"""Tests for Document API endpoints."""
import io
import os
import tempfile
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.conftest import make_mock_session


@pytest.fixture
def doc_test_client():
    """Create test client with DB override for document tests."""
    with patch("app.main.ensure_collection"):
        from app.main import app
        from app.api.deps import get_current_user, get_current_user_optional
        from app.database import get_db

        test_user = {
            "user_id": str(uuid.uuid4()),
            "org_id": str(uuid.uuid4()),
            "roles": ["editor"],
        }
        mock_sess = make_mock_session()

        async def _override_db():
            yield mock_sess

        app.dependency_overrides[get_current_user] = lambda: test_user
        app.dependency_overrides[get_current_user_optional] = lambda: test_user
        app.dependency_overrides[get_db] = _override_db

        from fastapi.testclient import TestClient
        client = TestClient(app)
        yield client, mock_sess

        app.dependency_overrides.pop(get_current_user, None)
        app.dependency_overrides.pop(get_current_user_optional, None)
        app.dependency_overrides.pop(get_db, None)


def test_upload_document(doc_test_client):
    client, mock_sess = doc_test_client
    kb_id = uuid.uuid4()

    # Set up KB lookup to return a valid KB
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = MagicMock()
    mock_sess.execute = AsyncMock(return_value=mock_result)

    # Populate document fields after add
    def populate_doc(doc_obj):
        doc_obj.id = uuid.uuid4()
        doc_obj.created_at = datetime.now(timezone.utc)
        doc_obj.updated_at = datetime.now(timezone.utc)
        doc_obj.status = "draft"
        doc_obj.security_level = "internal"
        doc_obj.current_version = 1
        doc_obj.source_type = "upload"

    mock_sess.add = MagicMock(side_effect=populate_doc)
    mock_sess.flush = AsyncMock()
    mock_sess.refresh = AsyncMock()
    mock_sess.commit = AsyncMock()

    with patch("app.api.v1.documents.parse_document_task") as mock_task, \
         patch("app.api.v1.documents.settings") as mock_settings:
        mock_task.delay = MagicMock()
        mock_settings.storage_path = tempfile.mkdtemp()

        resp = client.post(
            f"/api/v1/kbs/{kb_id}/documents",
            files={"file": ("test.txt", io.BytesIO(b"Test content"), "text/plain")},
        )

    assert resp.status_code == 201
    data = resp.json()
    assert "id" in data
    assert data["status"] == "draft"


def test_upload_document_kb_not_found(doc_test_client):
    client, mock_sess = doc_test_client
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    mock_sess.execute = AsyncMock(return_value=mock_result)

    with patch("app.api.v1.documents.settings") as mock_settings:
        mock_settings.storage_path = tempfile.mkdtemp()

        resp = client.post(
            "/api/v1/kbs/nonexistent-kb/documents",
            files={"file": ("test.txt", io.BytesIO(b"Test"), "text/plain")},
        )

    assert resp.status_code == 404


def test_get_document(doc_test_client):
    client, mock_sess = doc_test_client
    mock_doc = MagicMock()
    mock_doc.id = uuid.uuid4()
    mock_doc.org_id = uuid.uuid4()
    mock_doc.kb_id = uuid.uuid4()
    mock_doc.title = "Test Document"
    mock_doc.file_name = "test.pdf"
    mock_doc.file_type = "pdf"
    mock_doc.source_type = "upload"
    mock_doc.current_version = 1
    mock_doc.status = "ready"
    mock_doc.security_level = "internal"
    mock_doc.created_by = uuid.uuid4()
    mock_doc.created_at = "2024-01-01T00:00:00Z"
    mock_doc.updated_at = "2024-01-01T00:00:00Z"

    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = mock_doc
    mock_sess.execute = AsyncMock(return_value=mock_result)

    resp = client.get(f"/api/v1/documents/{mock_doc.id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["title"] == "Test Document"


def test_get_document_not_found(doc_test_client):
    client, mock_sess = doc_test_client
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    mock_sess.execute = AsyncMock(return_value=mock_result)

    resp = client.get(f"/api/v1/documents/{uuid.uuid4()}")
    assert resp.status_code == 404


def test_list_documents(doc_test_client):
    client, mock_sess = doc_test_client
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = []
    mock_result.scalar.return_value = 0
    mock_sess.execute = AsyncMock(return_value=mock_result)

    resp = client.get("/api/v1/kbs/test-kb/documents")
    assert resp.status_code == 200
    data = resp.json()
    assert data["items"] == []
    assert data["total"] == 0


def test_delete_document(doc_test_client):
    client, mock_sess = doc_test_client
    mock_doc = MagicMock()
    mock_doc.id = uuid.uuid4()
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = mock_doc
    mock_sess.execute = AsyncMock(return_value=mock_result)

    resp = client.delete(f"/api/v1/documents/{mock_doc.id}")
    assert resp.status_code == 204


def test_delete_document_not_found(doc_test_client):
    client, mock_sess = doc_test_client
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    mock_sess.execute = AsyncMock(return_value=mock_result)

    resp = client.delete(f"/api/v1/documents/{uuid.uuid4()}")
    assert resp.status_code == 404
