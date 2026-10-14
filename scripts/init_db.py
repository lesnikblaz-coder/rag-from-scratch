import asyncio

from sqlalchemy import text

from src.db.base import Base, async_engine

# registers its table in Base.metadata
from src.db.models.doc_chunk import DocumentChunk


async def main():

    async with async_engine.begin() as conn:

        await conn.execute(
            text("CREATE EXTENSION IF NOT EXISTS vector")
        )

        await conn.run_sync(Base.metadata.create_all)

    await async_engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())