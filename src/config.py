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

    # LLM MODELS
    EMBEDDING_MODEL: str = "gemini-embedding-001"
    INTERACTION_MODEL: str = "gemini-3.1-flash-lite" # "gemini-3.8-flash"


    @property
    def rag_database_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.RAG_DB_USER}:{self.RAG_DB_PASSWORD}"
            f"@{self.RAG_DB_HOST}:{self.RAG_DB_PORT}/{self.RAG_DB_NAME}"
        )


settings = Settings()