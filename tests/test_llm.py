"""Tests for LLM service."""
from unittest.mock import MagicMock, patch

from app.services.llm import generate_stream, SYSTEM_PROMPT


def test_system_prompt_content():
    assert "知识助手" in SYSTEM_PROMPT
    assert "来源编号" in SYSTEM_PROMPT
    assert "中文" in SYSTEM_PROMPT


def test_generate_stream_calls_api():
    with patch("app.services.llm.Generation") as mock_gen, \
         patch("app.services.llm.dashscope"):
        from app.services.llm import settings

        settings.llm_model = "configured-chat-model"
        settings.dashscope_api_key = "configured-api-key"
        settings.llm_timeout = 123
        mock_resp = MagicMock()
        mock_resp.output = MagicMock()
        mock_resp.output.choices = [{"message": {"content": "answer"}}]
        mock_gen.call = MagicMock(return_value=iter([mock_resp]))

        generate_stream(
            query="What is RAG?",
            context="RAG stands for Retrieval Augmented Generation.",
        )

        mock_gen.call.assert_called_once()
        call_kwargs = mock_gen.call.call_args[1]
        assert call_kwargs["model"] == "configured-chat-model"
        assert call_kwargs["api_key"] == "configured-api-key"
        assert call_kwargs["request_timeout"] == 123
        assert call_kwargs["stream"] is True
        assert call_kwargs["incremental_output"] is True


def test_generate_stream_with_messages():
    with patch("app.services.llm.Generation") as mock_gen, \
         patch("app.services.llm.dashscope"):
        mock_resp = MagicMock()
        mock_resp.output = MagicMock()
        mock_resp.output.choices = [{"message": {"content": "follow up"}}]
        mock_gen.call = MagicMock(return_value=iter([mock_resp]))

        messages = [{"role": "assistant", "content": "previous answer"}]
        generate_stream(
            query="Tell me more",
            context="context",
            messages=messages,
        )

        call_kwargs = mock_gen.call.call_args[1]
        # Messages should include system, user query, and additional messages
        assert len(call_kwargs["messages"]) >= 2
