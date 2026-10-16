import asyncio

from google import genai

from src.config import settings
from src.db.base import async_engine, AsyncSessionLocal
from src.services.embedding_service import EmbeddingService
from src.repos.chunk_repo import ChunkRepo


DOCUMENTS = [
    "Python and FastAPI are used to build APIs.",
    "PostgreSQL is a relational database.",
    "Football players train to improve fitness.",
    "Neural networks learn patterns from data.",
    "Vegetables can contribute to a healthy diet.",
]


async def main():
    aclient = genai.Client(
        api_key=settings.GEMINI_API_KEY
    ).aio

    try:
        embedding_service = EmbeddingService(aclient)

        # generate embeddings
        embeddings = await embedding_service.embed_documents(DOCUMENTS)

        # database transaction
        async with AsyncSessionLocal.begin() as session:
            repo = ChunkRepo(session)

            for content, embedding in zip(DOCUMENTS, embeddings, strict=True):
                repo.add(content, embedding)

        # transaction automatically commits
        print("Documents successfully stored!")


    finally:
        await aclient.aclose()
        await async_engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())