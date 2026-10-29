import time

from google import genai
from google.genai.client import AsyncClient

from src.config import settings
from src import enums
from src.logging_config import logger


class EmbeddingService:
    def __init__(self, client: AsyncClient):
        self.client = client

    async def embed_documents(self, documents: list[str]) -> list[list[float]]:
        start = time.perf_counter()

        response = await self.client.models.embed_content(
            model=settings.EMBEDDING_MODEL,
            contents=documents,
            config=genai.types.EmbedContentConfig(
                task_type=enums.EmbeddingTaskTypes.RETRIEVAL_DOCUMENT,
                output_dimensionality=enums.EmbeddingDimensions.OUTPUT_DIM
            )
        )

        duration_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "Documents embedding completed duration_ms=%.2f",
            duration_ms
        )

        return [embedding.values for embedding in response.embeddings]

    async def embed_query(self, query: str) -> list[float]:
        start = time.perf_counter()

        response = await self.client.models.embed_content(
            model=settings.EMBEDDING_MODEL,
            contents=query,
            config=genai.types.EmbedContentConfig(
                task_type=enums.EmbeddingTaskTypes.RETRIEVAL_QUERY,
                output_dimensionality=enums.EmbeddingDimensions.OUTPUT_DIM
            )
        )

        duration_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "Query embedding completed duration_ms=%.2f",
            duration_ms
        )

        return response.embeddings[0].values