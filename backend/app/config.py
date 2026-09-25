from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+psycopg://isc:isc@localhost:5432/isc_kb"

    ollama_url: str = "http://localhost:11434"
    embed_model: str = "bge-m3"
    embed_dim: int = 1024
    embed_batch_size: int = 32

    max_upload_mb: int = 20
    job_max_attempts: int = 3
    worker_poll_seconds: float = 2.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
