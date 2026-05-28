"""Tests for config module."""
from app.config import Settings


def test_default_settings():
    settings = Settings(
        secret_key="test",
        dashscope_api_key="test",
    )
    assert settings.app_name == "RAG Knowledge Base"
    assert settings.debug is False
    assert settings.llm_model == "qwen-plus"
    assert settings.embedding_model == "text-embedding-v3"
    assert settings.rag_max_chunks == 8
    assert settings.rag_chunk_size == 600
    assert settings.rag_top_k == 20


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "qwen-max")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    monkeypatch.setenv("SECRET_KEY", "test")
    monkeypatch.setenv("RAG_TOP_K", "50")

    settings = Settings()
    assert settings.llm_model == "qwen-max"
    assert settings.rag_top_k == 50
    assert settings.dashscope_api_key == "test-key"


def test_required_settings():
    # Settings should have defaults for all required fields
    settings = Settings(
        secret_key="test",
        dashscope_api_key="test",
    )
    assert settings.database_url is not None
    assert settings.redis_url is not None
    assert settings.weaviate_url is not None
    assert settings.storage_path is not None
