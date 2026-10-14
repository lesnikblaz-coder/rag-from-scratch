from google import genai

from src.config import settings
from src import enums


class EmbeddingService:
    def __init__(self, client: genai.Client):
        self.client = client

    async def embed_documents(self, documents: list[str]) -> list[list[float]]:
        response = self.client.models.embed_content(
            model=settings.EMBEDDING_MODEL,
            contents=documents,
            config=genai.types.EmbedContentConfig(
                task_type=enums.EmbeddingTaskTypes.RETRIEVAL_DOCUMENT,
                output_dimensionality=enums.EmbeddingDimensions.OUTPUT_DIM
            )
        )

        return [embedding.values for embedding in response.embeddings]

    async def embed_query(self, query: str) -> list[float]:
        response = self.client.models.embed_content(
            model=settings.EMBEDDING_MODEL,
            contents=query,
            config=genai.types.EmbedContentConfig(
                task_type=enums.EmbeddingTaskTypes.RETRIEVAL_QUERY,
                output_dimensionality=enums.EmbeddingDimensions.OUTPUT_DIM
            )
        )

        return response.embeddings[0].values