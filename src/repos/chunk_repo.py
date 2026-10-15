from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models.doc_chunk import DocumentChunk


class ChunkRepo:

    def __init__(self, session: AsyncSession):
        self.session = session

    def add(
            self,
            content: str,
            embedding: list[float],
            source_file: str,
            chunk_index: int
    ) -> None:
        chunk = DocumentChunk(
            content=content,
            embedding=embedding,
            source_file=source_file,
            chunk_index=chunk_index
        )

        self.session.add(chunk)

    async def search_similar(self, query_embedding: list[float], limit: int = 3):
        distance = DocumentChunk.embedding.cosine_distance(query_embedding)

        similarity = (1 - distance).label("similarity")

        stmt = (
            select(
                DocumentChunk.source_file,
                DocumentChunk.chunk_index,
                DocumentChunk.content,
                similarity
            )
            .order_by(distance)
            .limit(limit)
        )

        result = await self.session.execute(stmt)

        return [dict(row) for row in result.mappings().all()]

    async def delete_by_source(self, source_file: str) -> None:
        stmt = delete(DocumentChunk).where(DocumentChunk.source_file == source_file)
        await self.session.execute(stmt)