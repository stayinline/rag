"""Tests for Ingestion API endpoint."""
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.conftest import make_mock_session


@pytest.fixture
def ingestion_test_client():
    """Create test client with DB override for ingestion tests."""
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


def test_get_ingestion_job(ingestion_test_client):
    client, mock_sess = ingestion_test_client
    mock_job = MagicMock()
    mock_job.id = uuid.uuid4()
    mock_job.document_id = uuid.uuid4()
    mock_job.version_id = uuid.uuid4()
    mock_job.job_type = "parse"
    mock_job.status = "completed"
    mock_job.retry_count = 0
    mock_job.error_code = None
    mock_job.error_message = None
    mock_job.started_at = "2024-01-01T00:00:00Z"
    mock_job.finished_at = "2024-01-01T00:01:00Z"
    mock_job.created_at = "2024-01-01T00:00:00Z"

    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = mock_job
    mock_sess.execute = AsyncMock(return_value=mock_result)

    resp = client.get(f"/api/v1/ingestion-jobs/{mock_job.id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["job_type"] == "parse"
    assert data["status"] == "completed"


def test_get_ingestion_job_not_found(ingestion_test_client):
    client, mock_sess = ingestion_test_client
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    mock_sess.execute = AsyncMock(return_value=mock_result)

    resp = client.get(f"/api/v1/ingestion-jobs/{uuid.uuid4()}")
    assert resp.status_code == 404


def test_get_ingestion_job_unauthorized():
    """Test without auth override - returns 422 for missing header."""
    with patch("app.main.ensure_collection"):
        from app.main import app
        from fastapi.testclient import TestClient
        from app.api.deps import get_current_user

        saved = app.dependency_overrides.get(get_current_user)
        app.dependency_overrides.pop(get_current_user, None)

        try:
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.get(f"/api/v1/ingestion-jobs/{uuid.uuid4()}")
            assert resp.status_code in (401, 422)
        finally:
            if saved:
                app.dependency_overrides[get_current_user] = saved
