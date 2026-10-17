from google.genai.client import AsyncClient

from src.services.embedding_service import EmbeddingService
from src.repos.chunk_repo import ChunkRepo
from src.db.base import AsyncSessionLocal
from src.prompts.rag import build_rag_prompt, RAG_SYSTEM_INSTRUCTION
from src.config import settings
from src.schemas import ai
from src.logging_config import logger
from src.services.fusion import reciprocal_rank_fusion


# responsible for coordinating the complete question-answering process
class RAGService:
    def __init__(self, service: EmbeddingService, client: AsyncClient):
        self.embedding_service = service
        self.client = client

    async def ask(self, question: str, min_similarity: float = 0.60) -> ai.AIResponseAndSource:
        query_embedding = await self.embedding_service.embed_query(question)

        async with AsyncSessionLocal() as session:
            chunk_repo = ChunkRepo(session)
            semantic_chunks = await chunk_repo.search_similar(
                query_embedding
            )

            keyword_chunks = await chunk_repo.search_keyword(question)

            logger.debug("semantic_chunks=\n%s", [ss_c for ss_c in semantic_chunks])
            logger.debug("keyword_chunks=\n%s", [sk_c for sk_c in keyword_chunks])

        if not semantic_chunks and not keyword_chunks:
            return ai.AIResponseAndSource(
                answer=(
                    "I couldn't find the answer in the provided documents. "
                    "Try rephrasing your question."
                ),
                sources=[]
            )

        relevant_semantic_chunks = [
            c for c in semantic_chunks
            if c["similarity"] >= min_similarity
        ]

        if not relevant_semantic_chunks and not keyword_chunks:
            return ai.AIResponseAndSource(
                answer=(
                    "I couldn't find the answer in the provided documents. "
                    "Try rephrasing your question."
                ),
                sources=[]
            )

        fused_chunks = reciprocal_rank_fusion(
            semantic_results=relevant_semantic_chunks,
            keyword_results=keyword_chunks
        )[:3]

        sources = [
            ai.Source(
                source_file=c["source_file"],
                chunk_index=c["chunk_index"],
                similarity=c.get("similarity"),
                rrf_score=c["rrf_score"]
            )
            for c in fused_chunks
        ]

        rag_prompt = build_rag_prompt(question, fused_chunks)

        logger.debug("RAG_PROMPT=%s", rag_prompt)

        interaction = await self.client.interactions.create(
            model=settings.INTERACTION_MODEL,
            system_instruction=RAG_SYSTEM_INSTRUCTION,
            input=rag_prompt,
            response_format={
                "type": "text",
                "mime_type": "application/json",
                "schema": ai.AIInteractionResponse.model_json_schema()
            },
        )

        llm_response = ai.AIInteractionResponse.model_validate_json(
            interaction.output_text
        )

        return ai.AIResponseAndSource(
            answer=llm_response.answer,
            sources=sources
        )