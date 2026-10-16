import asyncio

from google import genai

from src.config import settings
from src.db.base import AsyncSessionLocal, async_engine
from src.repos.chunk_repo import ChunkRepo
from src.services.embedding_service import EmbeddingService
from tests.evals.rag_cases import EVAL_CASES


async def main():
    aclient = genai.Client(
        api_key=settings.GEMINI_API_KEY
    ).aio

    try:
        embedding_service = EmbeddingService(aclient)

        for case in EVAL_CASES:
            question = case["question"]

            # generate the query embedding
            query_embedding = await embedding_service.embed_query(question)

            async with AsyncSessionLocal() as session:
                chunk_repo = ChunkRepo(session)
                chunks = await chunk_repo.search_similar(query_embedding)

            print(f"\nQUESTION: \n{question}\n")
            print("TOP RESULTS:")

            for i, c in enumerate(chunks, start=1):
                print(
                    f"{i}.",
                    f"{c['source_file']} | ",
                    f"Chunk: {c['chunk_index']} | ",
                    f"Similarity: {c['similarity']}"
                )

            if case["answerable"]:
                print(f"\nEXPECTED CHUNK:\n{case['expected_chunk_index']}\n")

                hit = any(
                    c['chunk_index'] == case['expected_chunk_index']
                    for c in chunks
                )

                print("HIT@3:")
                print("PASS" if hit else "FAIL")

            else:
                print("\nEXPECTED:\nNo relevant chunks")

    finally:
        await aclient.aclose()
        await async_engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())


# RESULTS
"""
QUESTION: 
How many vacation days do employees receive?

TOP RESULTS:
1. sample.txt |  Chunk: 4 |  Similarity: 0.6802954163852718
2. sample.txt |  Chunk: 3 |  Similarity: 0.6724797171126939
3. sample.txt |  Chunk: 5 |  Similarity: 0.6645193608561953

EXPECTED CHUNK:
4

HIT@3:
PASS

QUESTION: 
Who won the 2022 FIFA World Cup?

TOP RESULTS:
1. sample.txt |  Chunk: 14 |  Similarity: 0.4841368676352933
2. sample.txt |  Chunk: 2 |  Similarity: 0.46146308661008406
3. sample.txt |  Chunk: 7 |  Similarity: 0.45933999133688563

EXPECTED:
No relevant chunks
"""