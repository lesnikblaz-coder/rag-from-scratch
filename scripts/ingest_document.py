import asyncio

from google import genai
from pathlib import Path

from src.config import settings
from src.db.base import AsyncSessionLocal, async_engine
from src.repos.chunk_repo import ChunkRepo
from src.services.chunking import chunk_text
from src.services.document_loader import load_text_file
from src.services.embedding_service import EmbeddingService

DOCUMENT_PATH = Path(__file__).resolve().parent.parent / "sample.txt"


async def main():
    client = genai.Client(
        api_key=settings.GEMINI_API_KEY
    )

    try:
        embedding_service = EmbeddingService(client)

        # load document
        text = load_text_file(DOCUMENT_PATH)

        # chunk document
        chunks = chunk_text(text)

        if not chunks:
            raise ValueError("Document is empty.")

        print(f"Generated {len(chunks)} chunks")

        # generate embeddings
        embeddings = await embedding_service.embed_documents(chunks)

        # db transaction
        async with AsyncSessionLocal.begin() as session:
            chunk_repo = ChunkRepo(session)

            # delete old chunks for this document
            await chunk_repo.delete_by_source(DOCUMENT_PATH.name)

            # insert new chunks
            for i, (c, e) in enumerate(zip(chunks, embeddings, strict=True), start=1):
                chunk_repo.add(
                    c,
                    e,
                    DOCUMENT_PATH.name,
                    i
                )

        print("Document successfully ingested!")

    finally:
        await client.aio.aclose()
        await async_engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())