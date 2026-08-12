from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    POSTGRES_USER: str = Field(default="postgres")
    POSTGRES_PASSWORD: str = Field(default="postgres")
    POSTGRES_DB: str = Field(default="scholar")
    POSTGRES_HOST: str = Field(default="localhost")
    POSTGRES_PORT: int = Field(default=5432)
    DATABASE_URL: str | None = Field(default=None)

    REDIS_URL: str = Field(default="redis://localhost:6379/0")

    QDRANT_URL: str = Field(default="http://localhost:6333")
    QDRANT_API_KEY: str | None = Field(default=None)
    COLLECTION_NAME: str = Field(default="documents")
    DENSE_VECTOR_SIZE: int = Field(default=384)

    OPENROUTER_API_KEY: str = Field(default="")
    OPENROUTER_BASE_URL: str = Field(default="https://openrouter.ai/api/v1")
    OPENROUTER_FAST_MODEL: str = Field(default="meta-llama/llama-3.1-8b-instruct")
    OPENROUTER_STRONG_MODEL: str = Field(default="anthropic/claude-sonnet-4")

    GEMINI_API_KEY: str = Field(default="")
    GEMINI_BASE_URL: str = Field(default="https://generativelanguage.googleapis.com/v1beta/openai/")
    GEMINI_FAST_MODEL: str = Field(default="gemini-2.5-flash")
    GEMINI_STRONG_MODEL: str = Field(default="gemini-3.5-flash")

    COHERE_API_KEY: str | None = Field(default=None)
    COHERE_RERANK_MODEL: str = Field(default="rerank-english-v3.0")
    RERANK_ENABLED: bool = Field(default=True)

    EMBEDDING_MODEL: str = Field(default="all-MiniLM-L6-v2")
    SPARSE_EMBEDDING_MODEL: str = Field(default="Qdrant/bm25")

    CORS_ORIGINS: str = Field(default="http://localhost:3000,http://localhost:8501")

    APP_NAME: str = Field(default="Scholar API")
    ENV: str = Field(default="development")
    LOG_LEVEL: str = Field(default="INFO")

    FAITHFULNESS_THRESHOLD: float = Field(default=0.7)
    RETRY_MAX_ATTEMPTS: int = Field(default=3)

    MAX_UPLOAD_MB: int = Field(default=50)

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]

    @property
    def max_upload_bytes(self) -> int:
        return self.MAX_UPLOAD_MB * 1024 * 1024

    @property
    def database_url(self) -> str:
        if self.DATABASE_URL:
            return self.DATABASE_URL
        return (
            f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True, extra="ignore")


settings = Settings()
