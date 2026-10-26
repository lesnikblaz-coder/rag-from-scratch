import asyncio

from google import genai

from src.config import settings
from src.db.base import async_engine
from src.services.embedding_service import EmbeddingService
from src.services.rag_service import RAGService
from src.services.retrieval_service import RetrievalService
from src.db.base import AsyncSessionLocal


async def main():
    aclient = genai.Client(
        api_key=settings.GEMINI_API_KEY
    ).aio

    try:
        embedding_service = EmbeddingService(aclient)
        retrieval_service = RetrievalService(embedding_service, AsyncSessionLocal)
        rag_service = RAGService(retrieval_service, aclient)

        question: str = input("Question: ") # "Compare our vacation policy with our sick leave policy."

        response = await rag_service.ask(question)

        print("QUESTION:", question)
        print("ANSWER:", response.answer)
        print("SOURCES:", response.sources)

    finally:
        await aclient.aclose()
        await async_engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())

"""
QUESTION: Compare our vacation policy with our sick leave policy.
ANSWER: The company policies for annual leave and sick leave differ in their requirements and processes:

* **Annual Leave:** Full-time employees are entitled to 25 working days per calendar year [2, 3]. Requests must be submitted at least 10 calendar days in advance, and leave is only officially approved once a manager confirms it in the leave management system [2].
* **Sick Leave:** This policy applies when an employee cannot work due to illness [1, 2]. Employees are required to notify their manager as soon as reasonably possible, ideally before their scheduled start time, and must follow applicable legal requirements and company procedures for documentation [1, 2].
SOURCES: [Source(source_file='sample.txt', chunk_index=5, similarity=0.693152139752336, rrf_score=0.01639344262295082), Source(source_file='sample.txt', chunk_index=4, similarity=0.6738917256948318, rrf_score=0.016129032258064516), Source(source_file='sample.txt', chunk_index=13, similarity=0.6303128816365907, rrf_score=0.015873015873015872)]
"""

"""
QUESTION: Who won the 2022 FIFA World Cup?
ANSWER: I cannot answer this question from the provided documents, as they do not contain information about the 2022 FIFA World Cup.
SOURCES: []
"""
