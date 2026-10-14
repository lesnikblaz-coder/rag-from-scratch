import asyncio

from google import genai

from src.config import settings
from src.db.base import AsyncSessionLocal, async_engine
from src.repos.chunk_repo import ChunkRepo
from src.services.embedding_service import EmbeddingService

QUERY: str = input("Query: ")

async def main():
    client = genai.Client(
        api_key=settings.GEMINI_API_KEY
    )

    try:
        embedding_service = EmbeddingService(client)

        # generate query embedding
        query_embedding = await embedding_service.embed_query(QUERY)

        # database transaction
        async with AsyncSessionLocal() as session:
            repo = ChunkRepo(session)

            results = await repo.search_similar(
                query_embedding=query_embedding,
                limit=3
            )

        print(results)

    finally:
        await client.aio.aclose()
        await async_engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())