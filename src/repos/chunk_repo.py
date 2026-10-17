from sqlalchemy import select, delete, func
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

    async def search_similar(self, query_embedding: list[float], limit: int = 5):
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

    async def search_keyword(self, query: str, limit: int = 5):

        # tsvector is a normalized representation of text that PSQL can use for efficient keyword searching ("The cats are running through the garden." -> cat:2, garden:7, run:4)
        text_vector = func.to_tsvector(
            "english",
            DocumentChunk.content
        )

        # converts the user's search text into a PSQL tsquery ("running cats in the garden" -> run & cat & garden)
        search_query = func.plainto_tsquery(
            "english",
            query
        )

        # calculates the relevance score for a document based on how it's tsvector matches the user's tsquery
        rank = func.ts_rank(
            text_vector,
            search_query
        ).label("keyword_score")

        # @@ gives you documents that match -> run @ cat @ garden...
        stmt = (
            select(
                DocumentChunk.source_file,
                DocumentChunk.chunk_index,
                DocumentChunk.content,
                rank
            )
            .where(
                text_vector.op("@@")(search_query)
            )
            .order_by(rank.desc())
            .limit(limit)
        )

        result = await self.session.execute(stmt)

        return [dict(row) for row in result.mappings().all()]

    async def delete_by_source(self, source_file: str) -> None:
        stmt = delete(DocumentChunk).where(DocumentChunk.source_file == source_file)
        await self.session.execute(stmt)