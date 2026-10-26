import asyncio

from google import genai

from src.config import settings
from src.db.base import async_engine, AsyncSessionLocal
from src.services.embedding_service import EmbeddingService
from src.services.retrieval_service import RetrievalService
from src.services.document_agent import DocumentAgent
from src.repos.chunk_repo import ChunkRepo


async def main():
    aclient = genai.Client(
        api_key=settings.GEMINI_API_KEY
    ).aio

    try:
        async with AsyncSessionLocal.begin() as session:
            chunk_repo = ChunkRepo(session)
            embedding_service = EmbeddingService(aclient)
            retrieval_service = RetrievalService(embedding_service, chunk_repo)
            agent = DocumentAgent(aclient, retrieval_service)

            result = await agent.ask(
                question="Compare our company's annual leave policy and sick leave policy."
            )

            print(result)


    finally:
        await aclient.aclose()
        await async_engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())