"""Tests for embedding service."""
from unittest.mock import MagicMock, patch

from app.services.embedding import embed_text, embed_texts


def test_embed_text(mock_dashscope):
    from app.services.embedding import settings

    settings.embedding_model = "configured-embedding-model"
    settings.dashscope_api_key = "configured-api-key"
    settings.llm_timeout = 123
    mock_embed = mock_dashscope["embed"]
    vector = embed_text("Hello world")
    assert isinstance(vector, list)
    assert len(vector) == 1536
    mock_embed.call.assert_called_once()
    call_kwargs = mock_embed.call.call_args[1]
    assert call_kwargs["model"] == "configured-embedding-model"
    assert call_kwargs["api_key"] == "configured-api-key"
    assert call_kwargs["request_timeout"] == 123


def test_embed_texts_batch(mock_dashscope):
    mock_embed = mock_dashscope["embed"]
    vectors = embed_texts(["text one", "text two"])
    assert len(vectors) == 2
    assert vectors[0] == [0.1] * 1536
    assert vectors[1] == [0.2] * 1536
    mock_embed.call.assert_called_once()


def test_embed_text_failure():
    with patch("app.services.embedding.TextEmbedding") as mock:
        resp = MagicMock()
        resp.status_code = 400
        resp.code = "INVALID_PARAMETER"
        resp.message = "Invalid input"
        mock.call = MagicMock(return_value=resp)
        with patch("app.services.embedding.dashscope"):
            try:
                embed_text("test")
                assert False, "Should have raised RuntimeError"
            except RuntimeError as e:
                assert "Embedding failed" in str(e)
