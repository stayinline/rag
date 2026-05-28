from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Application
    app_name: str = "RAG Knowledge Base"
    debug: bool = False
    secret_key: str = "change-me-in-production"
    api_prefix: str = "/api/v1"

    # PostgreSQL
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/rag_db"

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Weaviate
    weaviate_url: str = "http://localhost:8080"

    # Qwen / DashScope
    dashscope_api_key: str = ""
    llm_model: str = "qwen-plus"
    embedding_model: str = "text-embedding-v3"

    # File storage
    storage_path: str = "./data/files"

    # JWT
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 1440

    # RAG
    rag_max_chunks: int = 8
    rag_chunk_overlap: int = 80
    rag_chunk_size: int = 600
    rag_top_k: int = 20

    # Celery
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
