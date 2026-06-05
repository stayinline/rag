"""Tests for config module."""
from app.config import Settings


def test_default_settings():
    settings = Settings(
        secret_key="test",
        dashscope_api_key="test",
    )
    assert settings.app_name == "RAG Knowledge Base"
    assert settings.debug is True
    assert settings.llm_model == "qwen-plus-latest"
    assert settings.embedding_model == "text-embedding-v4"
    assert settings.rag_max_chunks == 8
    assert settings.rag_chunk_size == 600
    assert settings.rag_top_k == 20
    assert settings.llm_context_window_tokens == 8192
    assert settings.llm_max_output_tokens == 2048
    assert settings.rag_context_safety_margin_tokens == 512
    assert settings.rag_parent_context_window == 1
    assert settings.contextual_compression is True
    assert settings.contextual_compression_max_sentences == 3
    assert settings.llm_query_rewrite is False


def test_project_config_values_are_loaded():
    settings = Settings()
    assert settings.llm_model == "qwen-plus-latest"
    assert settings.embedding_model == "text-embedding-v4"
    assert settings.rerank_model_name == "qwen3.6-plus"
    assert settings.dashscope_api_key.startswith("sk-")
    assert settings.database_url == "postgresql+asyncpg://postgres:postgresql_admin@192.168.1.124:5432/rag"
    assert settings.redis_url == "redis://:redis_password@192.168.1.124:6379/0"
    assert settings.weaviate_url == "http://192.168.1.131:18080"
    assert settings.weaviate_grpc_port == 50051
    assert settings.clickhouse_url == "http://192.168.1.124:8123"
    assert settings.clickhouse_user == "default"
    assert settings.clickhouse_password
    assert settings.clickhouse_database == "rag"
    assert settings.debug is True
    assert settings.app_port == 8800


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "qwen-max")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    monkeypatch.setenv("SECRET_KEY", "test")
    monkeypatch.setenv("RAG_TOP_K", "50")
    monkeypatch.setenv("LLM_CONTEXT_WINDOW_TOKENS", "32768")
    monkeypatch.setenv("RAG_PARENT_CONTEXT_WINDOW", "2")
    monkeypatch.setenv("CONTEXTUAL_COMPRESSION", "false")
    monkeypatch.setenv("LLM_QUERY_REWRITE", "true")
    monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite:///./env.db")

    settings = Settings()
    assert settings.llm_model == "qwen-max"
    assert settings.rag_top_k == 50
    assert settings.llm_context_window_tokens == 32768
    assert settings.rag_parent_context_window == 2
    assert settings.contextual_compression is False
    assert settings.llm_query_rewrite is True
    assert settings.dashscope_api_key == "test-key"
    assert settings.database_url == "sqlite+aiosqlite:///./env.db"


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
