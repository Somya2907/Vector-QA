"""
config.py

Centralised configuration management using Pydantic Settings.
All environment variables are read here and exposed as a typed Settings object.
Call get_settings() anywhere in the application — it returns a cached singleton.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application-wide settings loaded from environment variables / .env file."""

    # LLM
    anthropic_api_key: str = Field("", env="ANTHROPIC_API_KEY")
    llm_model: str = Field("claude-sonnet-4-6", env="LLM_MODEL")
    llm_max_tokens: int = Field(1024, env="LLM_MAX_TOKENS")

    # Embeddings
    embedding_model: str = Field("voyage-3", env="EMBEDDING_MODEL")
    embedding_batch_size: int = Field(32, env="EMBEDDING_BATCH_SIZE")

    # Vector store
    vector_store_backend: str = Field("chromadb", env="VECTOR_STORE_BACKEND")
    vector_store_url: str = Field("", env="VECTOR_STORE_URL")
    vector_store_collection: str = Field("documents", env="VECTOR_STORE_COLLECTION")

    # Retrieval
    retrieval_top_k: int = Field(10, env="RETRIEVAL_TOP_K")

    # Chunking
    chunk_size: int = Field(512, env="CHUNK_SIZE")
    chunk_overlap: int = Field(64, env="CHUNK_OVERLAP")

    # API server
    api_host: str = Field("0.0.0.0", env="API_HOST")
    api_port: int = Field(8000, env="API_PORT")

    # Logging
    log_level: str = Field("INFO", env="LOG_LEVEL")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached application Settings singleton.

    Returns:
        Populated Settings instance (reads .env on first call).
    """
    return Settings()
