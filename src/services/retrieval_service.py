import time

from sqlalchemy.ext.asyncio import async_sessionmaker

from src.services.embedding_service import EmbeddingService
from src.repos.chunk_repo import ChunkRepo
from src.logging_config import logger
from src.services.fusion import reciprocal_rank_fusion


class RetrievalService:
    def __init__(self, embedding_service: EmbeddingService, session_factory: async_sessionmaker):
        self.embedding_service = embedding_service
        self.session_factory = session_factory

    async def search(self, query: str, limit: int = 5, min_similarity: float = 0.60) -> list[dict]:
        query_embedding = await self.embedding_service.embed_query(query)

        async with self.session_factory() as session:
            repo = ChunkRepo(session)

            start = time.perf_counter()

            semantic_chunks = await repo.search_similar(query_embedding, limit)
            keyword_chunks = await repo.search_keyword(query, limit)

            duration_ms = (time.perf_counter() - start) * 1000
            logger.info(
                "Database (cosine similarity & keyword) retrieval completed duration_ms=%.2f",
                duration_ms
            )

        relevant_semantic_chunks = [
            c for c in semantic_chunks
            if c["similarity"] >= min_similarity
        ]

        start = time.perf_counter()

        fused_chunks = reciprocal_rank_fusion(
            semantic_results=relevant_semantic_chunks,
            keyword_results=keyword_chunks
        )[:3]

        duration_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "Chunk fusion completed duration_ms=%.2f",
            duration_ms
        )

        return fused_chunks