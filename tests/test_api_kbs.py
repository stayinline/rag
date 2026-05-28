"""Tests for KB API endpoints."""
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.conftest import make_mock_session


@pytest.fixture
def kb_test_client():
    """Create test client with DB override for KB tests."""
    with patch("app.main.ensure_collection"):
        from app.main import app
        from app.api.deps import get_current_user, get_current_user_optional
        from app.database import get_db

        test_user = {
            "user_id": str(uuid.uuid4()),
            "org_id": str(uuid.uuid4()),
            "roles": ["viewer"],
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


def _make_kb_mock():
    """Create a mock KB object with all required fields."""
    kb = MagicMock()
    kb.id = uuid.uuid4()
    kb.org_id = uuid.uuid4()
    kb.name = "Test KB"
    kb.description = "A test KB"
    kb.is_active = True
    kb.created_by = uuid.uuid4()
    kb.created_at = datetime.now(timezone.utc)
    kb.updated_at = datetime.now(timezone.utc)
    return kb


def test_list_kbs_empty(kb_test_client):
    client, mock_sess = kb_test_client
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = []
    mock_result.scalar.return_value = 0
    mock_sess.execute = AsyncMock(return_value=mock_result)

    resp = client.get("/api/v1/kbs")
    assert resp.status_code == 200
    data = resp.json()
    assert data["items"] == []
    assert data["total"] == 0


def test_create_kb(kb_test_client):
    client, mock_sess = kb_test_client

    added_kbs = []

    def capture_add(kb_obj):
        kb_obj.id = uuid.uuid4()
        kb_obj.created_at = datetime.now(timezone.utc)
        kb_obj.updated_at = datetime.now(timezone.utc)
        kb_obj.is_active = True
        added_kbs.append(kb_obj)

    mock_sess.add = MagicMock(side_effect=capture_add)
    mock_sess.flush = AsyncMock()
    mock_sess.refresh = AsyncMock()
    mock_sess.commit = AsyncMock()

    resp = client.post(
        "/api/v1/kbs",
        json={"name": "Test KB", "description": "A test KB"},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["name"] == "Test KB"


def test_get_kb_not_found(kb_test_client):
    client, mock_sess = kb_test_client
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    mock_sess.execute = AsyncMock(return_value=mock_result)

    resp = client.get(f"/api/v1/kbs/{uuid.uuid4()}")
    assert resp.status_code == 404


def test_update_kb(kb_test_client):
    client, mock_sess = kb_test_client
    mock_kb = _make_kb_mock()

    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = mock_kb
    mock_sess.execute = AsyncMock(return_value=mock_result)

    resp = client.patch(
        f"/api/v1/kbs/{mock_kb.id}",
        json={"name": "New Name"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "New Name"


def test_delete_kb(kb_test_client):
    client, mock_sess = kb_test_client
    mock_kb = _make_kb_mock()

    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = mock_kb
    mock_sess.execute = AsyncMock(return_value=mock_result)

    resp = client.delete(f"/api/v1/kbs/{mock_kb.id}")
    assert resp.status_code == 204


def test_health_check(test_client):
    resp = test_client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_openapi_docs_available(test_client):
    resp = test_client.get("/docs")
    assert resp.status_code == 200
