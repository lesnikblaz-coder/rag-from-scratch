from google import genai

from src.services.embedding_service import EmbeddingService
from src.repos.chunk_repo import ChunkRepo
from src.db.base import AsyncSessionLocal
from src.prompts.rag import build_rag_prompt, RAG_SYSTEM_INSTRUCTION
from src.config import settings
from src.schemas import ai
from src.logging_config import logger


# responsible for coordinating the complete question-answering process
class RAGService:
    def __init__(self, service: EmbeddingService, client: genai.Client):
        self.embedding_service = service
        self.client = client

    async def ask(self, question: str) -> ai.AIInteractionResponse:
        query_embedding = await self.embedding_service.embed_query(question)

        async with AsyncSessionLocal.begin() as session:
            chunk_repo = ChunkRepo(session)
            chunks = await chunk_repo.search_similar(
                query_embedding
            )

            logger.debug("CHUNKS=\n%s", [c for c in chunks])

        if not chunks:
            return ai.AIInteractionResponse(
                answer="",
                sources=[]
            )

        rag_prompt = build_rag_prompt(question, chunks)

        logger.debug("RAG_PROMPT=%s", rag_prompt)

        interaction = self.client.interactions.create(
            model=settings.INTERACTION_MODEL,
            system_instruction=RAG_SYSTEM_INSTRUCTION,
            input=rag_prompt,
            response_format={
                "type": "text",
                "mime_type": "application/json",
                "schema": ai.AIInteractionResponse.model_json_schema()
            },
        )

        return ai.AIInteractionResponse.model_validate_json(interaction.output_text)