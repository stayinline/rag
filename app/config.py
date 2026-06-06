from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _load_root_config() -> dict[str, Any]:
    """Read simple constants from the project-level config.py without executing it."""
    config_path = Path(__file__).resolve().parents[1] / "config.py"
    if not config_path.exists():
        return {}

    values: dict[str, Any] = {}
    tree = ast.parse(config_path.read_text(encoding="utf-8"), filename=str(config_path))
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or not target.id.isupper():
            continue
        try:
            values[target.id] = ast.literal_eval(node.value)
        except (TypeError, ValueError, SyntaxError):
            continue
    _load_weaviate_client_config(tree, values)
    return values


def _load_weaviate_client_config(tree: ast.Module, values: dict[str, Any]) -> None:
    """Extract Weaviate connection values from a project-level client definition."""
    if "WEAVIATE_URL" in values and "WEAVIATE_GRPC_PORT" in values:
        return

    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for call in ast.walk(node.value):
            if not isinstance(call, ast.Call):
                continue
            func = call.func
            if not (
                isinstance(func, ast.Attribute)
                and func.attr == "from_url"
                and isinstance(func.value, ast.Name)
                and func.value.id == "ConnectionParams"
            ):
                continue

            if call.args and "WEAVIATE_URL" not in values:
                try:
                    values["WEAVIATE_URL"] = ast.literal_eval(call.args[0])
                except (TypeError, ValueError, SyntaxError):
                    pass

            for keyword in call.keywords:
                if keyword.arg == "grpc_port" and "WEAVIATE_GRPC_PORT" not in values:
                    try:
                        values["WEAVIATE_GRPC_PORT"] = ast.literal_eval(keyword.value)
                    except (TypeError, ValueError, SyntaxError):
                        pass
            return


_ROOT_CONFIG = _load_root_config()


def _root(name: str, default: Any) -> Any:
    return _ROOT_CONFIG.get(name, default)


def _field(default: Any, *aliases: str):
    return Field(default=default, validation_alias=AliasChoices(*aliases))


def _clickhouse_url() -> str:
    host = _root("CLICKHOUSE_HOST", None)
    port = _root("CLICKHOUSE_PORT", None)
    if host and port:
        return f"http://{host}:{port}"
    return "http://localhost:8123"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        populate_by_name=True,
        extra="ignore",
    )

    # Application
    app_name: str = _field("RAG Knowledge Base", "APP_NAME")
    debug: bool = _field(_root("DEBUG", False), "DEBUG")
    log_level: str = _field(_root("LOG_LEVEL", "INFO"), "LOG_LEVEL")
    secret_key: str = _field(_root("JWT_SECRET", "change-me-in-production"), "SECRET_KEY", "JWT_SECRET")
    api_prefix: str = _field("/api/v1", "API_PREFIX")

    # PostgreSQL
    database_url: str = _field(
        _root("POSTGRES_DSN", "postgresql+asyncpg://postgres:postgres@localhost:5432/rag_db"),
        "DATABASE_URL",
        "POSTGRES_DSN",
    )

    # Redis
    redis_url: str = _field(_root("REDIS_DSN", "redis://localhost:6379/0"), "REDIS_URL", "REDIS_DSN")

    # Weaviate
    weaviate_url: str = _field(_root("WEAVIATE_URL", "http://localhost:8080"), "WEAVIATE_URL")
    weaviate_grpc_port: int = _field(_root("WEAVIATE_GRPC_PORT", 50051), "WEAVIATE_GRPC_PORT")

    # Qwen / DashScope
    model_type: str = _field(_root("MODEL_TYPE", "qwen"), "MODEL_TYPE")
    dashscope_api_key: str = _field(_root("MODEL_API_KEY", ""), "DASHSCOPE_API_KEY", "MODEL_API_KEY")
    model_base_url: str = _field(_root("MODEL_BASE_URL", ""), "MODEL_BASE_URL")
    llm_model: str = _field(_root("CHAT_MODEL", "qwen-plus"), "LLM_MODEL", "CHAT_MODEL")
    llm_timeout: int = _field(_root("LLM_TIMEOUT", 90), "LLM_TIMEOUT")
    embedding_model: str = _field(
        _root("EMBEDDING_MODEL", _root("EMBEDDING_MODEL_NAME", "text-embedding-v3")),
        "EMBEDDING_MODEL",
        "EMBEDDING_MODEL_NAME",
    )
    rerank_model_name: str = _field(_root("RERANK_MODEL_NAME", ""), "RERANK_MODEL_NAME")
    summary_model: str = _field(_root("SUMMARY_MODEL", "qwen-turbo"), "SUMMARY_MODEL")
    llm_context_window_tokens: int = _field(
        _root("LLM_CONTEXT_WINDOW_TOKENS", 8192),
        "LLM_CONTEXT_WINDOW_TOKENS",
    )
    llm_max_output_tokens: int = _field(_root("LLM_MAX_OUTPUT_TOKENS", 2048), "LLM_MAX_OUTPUT_TOKENS")

    # File storage
    storage_path: str = _field("./data/files", "STORAGE_PATH")

    # JWT
    jwt_algorithm: str = _field(_root("JWT_ALGORITHM", "HS256"), "JWT_ALGORITHM")
    jwt_expire_minutes: int = _field(1440, "JWT_EXPIRE_MINUTES")

    # RAG
    rag_max_chunks: int = _field(8, "RAG_MAX_CHUNKS")
    rag_chunk_overlap: int = _field(80, "RAG_CHUNK_OVERLAP")
    rag_chunk_size: int = _field(600, "RAG_CHUNK_SIZE")
    rag_top_k: int = _field(20, "RAG_TOP_K")
    rag_context_safety_margin_tokens: int = _field(
        _root("RAG_CONTEXT_SAFETY_MARGIN_TOKENS", 512),
        "RAG_CONTEXT_SAFETY_MARGIN_TOKENS",
    )
    rag_parent_context_window: int = _field(_root("RAG_PARENT_CONTEXT_WINDOW", 1), "RAG_PARENT_CONTEXT_WINDOW")
    contextual_compression: bool = _field(_root("CONTEXTUAL_COMPRESSION", True), "CONTEXTUAL_COMPRESSION")
    contextual_compression_max_sentences: int = _field(
        _root("CONTEXTUAL_COMPRESSION_MAX_SENTENCES", 3),
        "CONTEXTUAL_COMPRESSION_MAX_SENTENCES",
    )

    # Celery
    celery_broker_url: str = _field(_root("REDIS_DSN", "redis://localhost:6379/1"), "CELERY_BROKER_URL")
    celery_result_backend: str = _field(_root("REDIS_DSN", "redis://localhost:6379/2"), "CELERY_RESULT_BACKEND")

    # Phase 2: SCI Paper & Domain Enhancement
    grobid_url: str = _field("http://localhost:8070", "GROBID_URL")
    reranker_type: str = _field("bm25", "RERANKER_TYPE")  # bm25, mock, or bge
    reranker_top_n: int = _field(10, "RERANKER_TOP_N")  # Number of candidates after reranking
    query_expansion: bool = _field(True, "QUERY_EXPANSION")  # Enable medical term expansion
    llm_query_rewrite: bool = _field(_root("LLM_QUERY_REWRITE", False), "LLM_QUERY_REWRITE")
    llm_query_rewrite_model: str = _field(_root("LLM_QUERY_REWRITE_MODEL", ""), "LLM_QUERY_REWRITE_MODEL")
    citation_min_score: float = _field(_root("CITATION_MIN_SCORE", 0.0), "CITATION_MIN_SCORE")
    metadata_retrieval: bool = _field(_root("METADATA_RETRIEVAL", True), "METADATA_RETRIEVAL")
    metadata_score_weight: float = _field(_root("METADATA_SCORE_WEIGHT", 0.2), "METADATA_SCORE_WEIGHT")
    retrieval_vector_weight: float = _field(_root("RETRIEVAL_VECTOR_WEIGHT", 0.4), "RETRIEVAL_VECTOR_WEIGHT")
    retrieval_bm25_weight: float = _field(_root("RETRIEVAL_BM25_WEIGHT", 0.3), "RETRIEVAL_BM25_WEIGHT")
    retrieval_metadata_weight: float = _field(_root("RETRIEVAL_METADATA_WEIGHT", 0.2), "RETRIEVAL_METADATA_WEIGHT")
    retrieval_hybrid_weight: float = _field(_root("RETRIEVAL_HYBRID_WEIGHT", 0.1), "RETRIEVAL_HYBRID_WEIGHT")
    feedback_learning: bool = _field(_root("FEEDBACK_LEARNING", True), "FEEDBACK_LEARNING")
    feedback_learning_window: int = _field(_root("FEEDBACK_LEARNING_WINDOW", 200), "FEEDBACK_LEARNING_WINDOW")
    feedback_learning_cache_ttl: int = _field(
        _root("FEEDBACK_LEARNING_CACHE_TTL", 300),
        "FEEDBACK_LEARNING_CACHE_TTL",
    )
    feedback_rerank_strength: float = _field(_root("FEEDBACK_RERANK_STRENGTH", 0.15), "FEEDBACK_RERANK_STRENGTH")

    # Phase 3: Analytics & Quality
    clickhouse_url: str = _field(_clickhouse_url(), "CLICKHOUSE_URL")
    clickhouse_host: str = _field(_root("CLICKHOUSE_HOST", "localhost"), "CLICKHOUSE_HOST")
    clickhouse_port: int = _field(_root("CLICKHOUSE_PORT", 8123), "CLICKHOUSE_PORT")
    clickhouse_user: str = _field(_root("CLICKHOUSE_USER", "default"), "CLICKHOUSE_USER")
    clickhouse_password: str = _field(_root("CLICKHOUSE_PASSWORD", ""), "CLICKHOUSE_PASSWORD")
    clickhouse_database: str = _field(_root("CLICKHOUSE_DATABASE", "default"), "CLICKHOUSE_DATABASE")
    enable_trace_logging: bool = _field(True, "ENABLE_TRACE_LOGGING")  # Enable RAG trace logging to ClickHouse

    # Project-level config extras
    rate_limit_requests: int = _field(_root("RATE_LIMIT_REQUESTS", 30), "RATE_LIMIT_REQUESTS")
    rate_limit_window: int = _field(_root("RATE_LIMIT_WINDOW", 60), "RATE_LIMIT_WINDOW")
    query_cache_ttl: int = _field(_root("QUERY_CACHE_TTL", 300), "QUERY_CACHE_TTL")
    retrieval_cache_ttl: int = _field(_root("RETRIEVAL_CACHE_TTL", 1800), "RETRIEVAL_CACHE_TTL")
    summary_after_rounds: int = _field(_root("SUMMARY_AFTER_ROUNDS", 2), "SUMMARY_AFTER_ROUNDS")
    max_context_rounds: int = _field(_root("MAX_CONTEXT_ROUNDS", 2), "MAX_CONTEXT_ROUNDS")
    sql_auto_fix_max_retries: int = _field(_root("SQL_AUTO_FIX_MAX_RETRIES", 3), "SQL_AUTO_FIX_MAX_RETRIES")
    planner_max_depth: int = _field(_root("PLANNER_MAX_DEPTH", 5), "PLANNER_MAX_DEPTH")
    planner_max_tool_calls: int = _field(_root("PLANNER_MAX_TOOL_CALLS", 10), "PLANNER_MAX_TOOL_CALLS")
    planner_mode: str = _field(_root("PLANNER_MODE", "langgraph"), "PLANNER_MODE")
    app_host: str = _field(_root("APP_HOST", "0.0.0.0"), "APP_HOST")
    app_port: int = _field(_root("APP_PORT", 8800), "APP_PORT")


settings = Settings()
