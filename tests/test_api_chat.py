"""Tests for Chat API endpoint."""
import uuid
from datetime import datetime, timezone
from unittest.mock import patch


def test_chat_stream(test_client):
    with patch("app.api.v1.chat.assemble_context_and_generate") as mock_gen:
        mock_gen.return_value = iter([
            {"delta": "RAG", "done": False, "trace_id": "t1", "sources": []},
            {"delta": " stands", "done": False, "trace_id": "t1", "sources": []},
            {"delta": " for Retrieval", "done": False, "trace_id": "t1", "sources": []},
            {"delta": "", "done": True, "trace_id": "t1", "sources": []},
        ])

        resp = test_client.post(
            "/api/v1/chat",
            json={"query": "What is RAG?", "stream": True},
        )

    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers.get("content-type", "")
    lines = resp.text.strip().split("\n")
    assert len(lines) >= 1
    assert lines[0].startswith("data: ")
    assert '"conversation_id":' in resp.text
    assert '"message_id":' in resp.text


def test_chat_non_stream(test_client):
    trace_id = str(uuid.uuid4())
    with patch("app.api.v1.chat.assemble_context_and_generate") as mock_gen, \
         patch("app.api.v1.chat.settings") as mock_settings:
        mock_settings.llm_model = "configured-chat-model"
        mock_gen.return_value = iter([
            {"delta": "RAG", "done": False, "trace_id": trace_id, "sources": []},
            {"delta": " is useful.", "done": False, "trace_id": trace_id, "sources": []},
            {"delta": "", "done": True, "trace_id": trace_id, "sources": [
                {"chunk_id": "c1", "document_id": "d1", "document_title": "Doc",
                 "section_path": None, "page_start": None, "page_end": None,
                 "score": 0.9, "content_preview": "Preview"}
            ]},
        ])

        resp = test_client.post(
            "/api/v1/chat",
            json={"query": "What is RAG?", "stream": False},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert "RAG is useful." in data["answer"]
    assert data["trace_id"] == trace_id
    assert data["message_id"]
    assert data["conversation_id"]
    assert len(data["sources"]) == 1
    assert data["model"] == "configured-chat-model"


def test_chat_with_kb_ids(test_client):
    kb_id = str(uuid.uuid4())
    with patch("app.api.v1.chat.assemble_context_and_generate") as mock_gen:
        mock_gen.return_value = iter([
            {"delta": "Answer", "done": False, "trace_id": "t1", "sources": []},
            {"delta": "", "done": True, "trace_id": "t1", "sources": []},
        ])

        resp = test_client.post(
            "/api/v1/chat",
            json={"query": "Test", "kb_ids": [kb_id], "stream": False},
        )

    assert resp.status_code == 200
    call_kwargs = mock_gen.call_args
    assert kb_id in call_kwargs[1]["kb_ids"]


def test_chat_empty_query(test_client):
    resp = test_client.post(
        "/api/v1/chat",
        json={"query": "", "stream": False},
    )
    assert resp.status_code == 422


def test_chat_unauthorized():
    """Test without auth - missing Authorization header returns 422 (validation error)."""
    with patch("app.main.ensure_collection"):
        from app.main import app
        from app.api.deps import get_current_user

        saved = app.dependency_overrides.get(get_current_user)
        app.dependency_overrides.pop(get_current_user, None)

        try:
            from fastapi.testclient import TestClient
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.post(
                "/api/v1/chat",
                json={"query": "Test", "stream": False},
            )
            # FastAPI returns 422 for missing required header, not 401
            assert resp.status_code in (401, 422)
        finally:
            if saved:
                app.dependency_overrides[get_current_user] = saved


def test_chat_with_conversation_id(test_client):
    conv_id = str(uuid.uuid4())
    conversation = _make_conversation(conv_id)
    messages_result = _mock_messages_result([])
    original_execute = test_client.mock_session.execute
    test_client.mock_session.execute.side_effect = [conversation, messages_result]
    with patch("app.api.v1.chat.assemble_context_and_generate") as mock_gen:
        mock_gen.return_value = iter([
            {"delta": "Response", "done": False, "trace_id": "t1", "sources": []},
            {"delta": "", "done": True, "trace_id": "t1", "sources": []},
        ])

        resp = test_client.post(
            "/api/v1/chat",
            json={"query": "Follow up", "conversation_id": conv_id, "stream": False},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["conversation_id"] == conv_id
    assert mock_gen.call_args.kwargs["messages"] == []
    test_client.mock_session.execute = original_execute


def test_chat_with_conversation_history_passes_summary_and_recent_messages(test_client):
    conv_id = str(uuid.uuid4())
    conversation = _make_conversation(conv_id)
    history = [
        _make_message(conv_id, "user", "第一轮用户问题：什么是免疫治疗？", 1),
        _make_message(conv_id, "assistant", "第一轮助手回答：免疫治疗说明。", 2),
        _make_message(conv_id, "user", "第二轮用户问题：它有哪些风险？", 3),
        _make_message(conv_id, "assistant", "第二轮助手回答：风险包括免疫相关不良反应。", 4),
        _make_message(conv_id, "user", "第三轮用户问题：适用人群是什么？", 5),
        _make_message(conv_id, "assistant", "第三轮助手回答：需结合分型和指南。", 6),
    ]
    messages_result = _mock_messages_result(history)
    original_execute = test_client.mock_session.execute
    test_client.mock_session.execute.side_effect = [conversation, messages_result]

    with patch("app.api.v1.chat.settings") as mock_settings, \
         patch("app.api.v1.chat.assemble_context_and_generate") as mock_gen:
        mock_settings.llm_model = "configured-chat-model"
        mock_settings.summary_after_rounds = 2
        mock_settings.max_context_rounds = 1
        mock_gen.return_value = iter([
            {"delta": "Answer", "done": False, "trace_id": "t1", "sources": []},
            {"delta": "", "done": True, "trace_id": "t1", "sources": []},
        ])

        resp = test_client.post(
            "/api/v1/chat",
            json={"query": "那它的禁忌是什么？", "conversation_id": conv_id, "stream": False},
        )

    assert resp.status_code == 200
    llm_messages = mock_gen.call_args.kwargs["messages"]
    assert llm_messages[0]["role"] == "system"
    assert "第一轮用户问题" in llm_messages[0]["content"]
    assert "第二轮助手回答" in llm_messages[0]["content"]
    assert llm_messages[1:] == [
        {"role": "user", "content": "第三轮用户问题：适用人群是什么？"},
        {"role": "assistant", "content": "第三轮助手回答：需结合分型和指南。"},
    ]
    test_client.mock_session.execute = original_execute


def test_chat_conversation_not_found(test_client):
    conv_id = str(uuid.uuid4())
    result = _mock_conversation_result(None)
    test_client.mock_session.execute.return_value = result
    with patch("app.api.v1.chat.assemble_context_and_generate") as mock_gen:
        mock_gen.return_value = iter([
            {"delta": "Response", "done": False, "trace_id": "t1", "sources": []},
            {"delta": "", "done": True, "trace_id": "t1", "sources": []},
        ])

        resp = test_client.post(
            "/api/v1/chat",
            json={"query": "Follow up", "conversation_id": conv_id, "stream": False},
        )

    assert resp.status_code == 404


def _mock_conversation_result(conversation):
    result = type("Result", (), {})()
    result.scalar_one_or_none = lambda: conversation
    return result


def _mock_count_result(count):
    result = type("Result", (), {})()
    result.scalar = lambda: count
    return result


def _mock_messages_result(messages):
    result = type("Result", (), {})()
    result.scalars = lambda: type("Scalars", (), {"all": lambda self: messages})()
    return result


def _make_message(conversation_id, role, content, sequence):
    message = type("ConversationMessage", (), {})()
    message.id = uuid.uuid4()
    message.conversation_id = uuid.UUID(str(conversation_id))
    message.role = role
    message.content = content
    message.sequence = sequence
    message.created_at = datetime.now(timezone.utc)
    return message


def _make_conversation(conv_id):
    conversation = type("ConversationResult", (), {})()
    conversation.scalar_one_or_none = lambda: _conversation_obj(conv_id)
    return conversation


def _conversation_obj(conv_id):
    obj = type("Conversation", (), {})()
    obj.id = uuid.UUID(conv_id)
    obj.kb_ids = []
    obj.message_count = 2
    obj.title = "Existing"
    obj.last_message_at = datetime.now(timezone.utc)
    obj.created_at = datetime.now(timezone.utc)
    obj.updated_at = datetime.now(timezone.utc)
    return obj
