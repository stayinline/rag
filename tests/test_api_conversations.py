"""Tests for conversation history API endpoints."""
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from tests.conftest import make_mock_session


@pytest.fixture
def conversations_client():
    with patch("app.main.ensure_collection"):
        from app.api.deps import get_current_user
        from app.database import get_db
        from app.main import app

        test_user = {
            "user_id": str(uuid.uuid4()),
            "org_id": str(uuid.uuid4()),
            "roles": ["viewer"],
        }
        mock_sess = make_mock_session()

        async def _override_db():
            yield mock_sess

        app.dependency_overrides[get_current_user] = lambda: test_user
        app.dependency_overrides[get_db] = _override_db

        client = TestClient(app)
        client.mock_session = mock_sess
        yield client

        app.dependency_overrides.pop(get_current_user, None)
        app.dependency_overrides.pop(get_db, None)


def _make_conversation():
    conv = MagicMock()
    conv.id = uuid.uuid4()
    conv.title = "What is RAG?"
    conv.kb_ids = []
    conv.message_count = 2
    conv.last_message_at = datetime.now(timezone.utc)
    conv.created_at = datetime.now(timezone.utc)
    conv.updated_at = datetime.now(timezone.utc)
    return conv


def _make_message(conversation_id, role, content, sequence):
    message = MagicMock()
    message.id = uuid.uuid4()
    message.conversation_id = conversation_id
    message.role = role
    message.content = content
    message.sequence = sequence
    message.trace_id = str(uuid.uuid4()) if role == "assistant" else None
    message.sources = []
    message.kb_ids = []
    message.model = "qwen-plus" if role == "assistant" else None
    message.prompt_version = "v1" if role == "assistant" else None
    message.created_at = datetime.now(timezone.utc)
    return message


def test_list_conversations(conversations_client):
    client = conversations_client
    conv = _make_conversation()

    count_result = MagicMock()
    count_result.scalar.return_value = 1
    list_result = MagicMock()
    list_result.scalars.return_value.all.return_value = [conv]
    client.mock_session.execute = AsyncMock(side_effect=[count_result, list_result])

    resp = client.get("/api/v1/conversations")

    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 1
    assert data["items"][0]["title"] == "What is RAG?"


def test_get_conversation_messages(conversations_client):
    client = conversations_client
    conv = _make_conversation()
    user_msg = _make_message(conv.id, "user", "Question", 1)
    assistant_msg = _make_message(conv.id, "assistant", "Answer", 2)

    conv_result = MagicMock()
    conv_result.scalar_one_or_none.return_value = conv
    messages_result = MagicMock()
    messages_result.scalars.return_value.all.return_value = [user_msg, assistant_msg]
    client.mock_session.execute = AsyncMock(side_effect=[conv_result, messages_result])

    resp = client.get(f"/api/v1/conversations/{conv.id}")

    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == str(conv.id)
    assert [m["role"] for m in data["messages"]] == ["user", "assistant"]


def test_get_conversation_not_found(conversations_client):
    client = conversations_client
    result = MagicMock()
    result.scalar_one_or_none.return_value = None
    client.mock_session.execute = AsyncMock(return_value=result)

    resp = client.get(f"/api/v1/conversations/{uuid.uuid4()}")

    assert resp.status_code == 404


def test_delete_conversation(conversations_client):
    client = conversations_client
    conv = _make_conversation()
    result = MagicMock()
    result.scalar_one_or_none.return_value = conv
    client.mock_session.execute = AsyncMock(return_value=result)

    resp = client.delete(f"/api/v1/conversations/{conv.id}")

    assert resp.status_code == 204
    assert client.mock_session.commit.await_count == 1
