import asyncio

from google import genai

from src.config import settings
from src.services.embedding_service import EmbeddingService
from src.services.rag_service import RAGService
from src.db.base import async_engine


async def main():
    client = genai.Client(
        api_key=settings.GEMINI_API_KEY
    )

    try:
        embedding_service = EmbeddingService(client)
        rag_service = RAGService(embedding_service, client)

        question: str = "How many vacation days do employees receive?" # input("Question: ")

        response = await rag_service.ask(question)

        print("QUESTION:", question)
        print("ANSWER:", response.answer)
        print("SOURCES:", response.sources)

    finally:
        await client.aio.aclose()
        await async_engine.dispose()

if __name__ == "__main__":
    asyncio.run(main())


"""
QUESTION: How many vacation days do employees receive?
ANSWER: Full-time employees receive 25 working days of annual leave per calendar year under the company's standard policy, subject to their employment agreements and applicable law [1].
SOURCES: [Source(source_file='sample.txt', chunk_index=4, similarity=1.0)]
"""

"""
QUESTION: Who won the 2022 FIFA World Cup?
ANSWER: I cannot answer this question from the provided documents, as they do not contain information about the 2022 FIFA World Cup.
SOURCES: []
"""