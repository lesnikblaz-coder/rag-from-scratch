from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        extra="ignore",
    )

    # vector database
    RAG_DB_USER: str
    RAG_DB_PASSWORD: str
    RAG_DB_HOST: str
    RAG_DB_PORT: str
    RAG_DB_NAME: str

    # API keys
    GEMINI_API_KEY: str

    # SECRET key
    API_SECRET: str

    # LLM MODELS
    EMBEDDING_MODEL: str = "gemini-embedding-001"
    INTERACTION_MODEL: str = "gemini-3.1-flash-lite"
    INTERACTION_MODEL_v3_8: str = "gemini-3.8-flash"

    # AGENT LIMITS
    MAX_AGENT_TOOL_ITERATIONS: int = 5
    MAX_AGENT_TOOL_CALLS: int = 10

    # TIMEOUTS
    RAG_INTERACTION_TIMEOUT_SECONDS: float = 30.0
    RAG_STREAM_TIMEOUT_SECONDS: float = 60.0

    # RETRIES
    API_RETRY_ATTEMPTS: int = 3
    API_RETRY_INITIAL_DELAY: float = 1.0
    API_RETRY_MAX_DELAY: float = 5.0
    API_RETRY_EXP_BASE: float = 2.0
    API_RETRY_JITTER: float = 1.0


    @property
    def rag_database_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.RAG_DB_USER}:{self.RAG_DB_PASSWORD}"
            f"@{self.RAG_DB_HOST}:{self.RAG_DB_PORT}/{self.RAG_DB_NAME}"
        )


settings = Settings()