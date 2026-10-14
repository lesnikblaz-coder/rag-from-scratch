from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models.doc_chunk import DocumentChunk


class ChunkRepo:

    def __init__(self, session: AsyncSession):
        self.session = session

    def add(self, content: str, embedding: list[float]) -> None:
        chunk = DocumentChunk(
            content=content,
            embedding=embedding
        )

        self.session.add(chunk)

    async def search_similar(self, query_embedding: list[float], limit: int = 3):
        distance = DocumentChunk.embedding.cosine_distance(query_embedding)

        similarity = (1 - distance).label("similarity")

        stmt = (
            select(
            DocumentChunk.content,
            similarity
        )
            .order_by(distance)
            .limit(limit)
        )

        result = await self.session.execute(stmt)

        return result.all()