from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env")

    RAG_DB_USER: str
    RAG_DB_PASSWORD: str
    RAG_DB_HOST: str
    RAG_DB_PORT: str
    RAG_DB_NAME: str


    @property
    def rag_database_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.RAG_DB_USER}:{self.RAG_DB_PASSWORD}"
            f"@{self.RAG_DB_HOST}:{self.RAG_DB_PORT}/{self.RAG_DB_NAME}"
        )


settings = Settings()