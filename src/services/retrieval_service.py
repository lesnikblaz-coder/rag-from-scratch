from src.services.embedding_service import EmbeddingService
from src.repos.chunk_repo import ChunkRepo
from src.db.base import AsyncSessionLocal, async_engine
from src.logging_config import logger
from src.services.fusion import reciprocal_rank_fusion


class RetrievalService:
    def __init__(self, embedding_service: EmbeddingService, chunk_repo: ChunkRepo):
        self.embedding_service = embedding_service
        self.chunk_repo = chunk_repo

    async def search(self, query: str, limit: int = 5, min_similarity: float = 0.60) -> list[dict]:
        query_embedding = await self.embedding_service.embed_query(query)


        semantic_chunks = await self.chunk_repo.search_similar(query_embedding, limit)

        keyword_chunks = await self.chunk_repo.search_keyword(query, limit)

        logger.info("Semantic search finished | length=%s", len(semantic_chunks))
        logger.info("Keyword search finished | length=%s", len(keyword_chunks))

        relevant_semantic_chunks = [
            c for c in semantic_chunks
            if c["similarity"] >= min_similarity
        ]

        fused_chunks = reciprocal_rank_fusion(
            semantic_results=relevant_semantic_chunks,
            keyword_results=keyword_chunks
        )[:3]

        return fused_chunks