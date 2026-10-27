from google.genai.client import AsyncClient

from src.services.retrieval_service import RetrievalService
from src.agent.prompts.rag import build_rag_prompt, RAG_SYSTEM_INSTRUCTION
from src.config import settings
from src.schemas import ai
from src.logging_config import logger
from src.exceptions import custom as err


# responsible for coordinating the complete question-answering process
class RAGService:
    def __init__(self, retrieval_service: RetrievalService, client: AsyncClient):
        self.retrieval_service = retrieval_service
        self.client = client

    async def ask(self, question: str, limit: int = 5, min_similarity: float = 0.60) -> ai.AIResponseAndSource:
        chunks = await self.retrieval_service.search(
            query=question,
            limit=limit,
            min_similarity=min_similarity
        )

        if not chunks:
            return ai.AIResponseAndSource(
                answer=(
                    "I couldn't find the answer in the provided documents. "
                    "Try rephrasing your question."
                ),
                sources=[]
            )

        sources = [
            ai.Source(
                source_file=c["source_file"],
                chunk_index=c["chunk_index"],
                similarity=c.get("similarity"),
                rrf_score=c["rrf_score"]
            )
            for c in chunks
        ]

        rag_prompt = build_rag_prompt(question, chunks)

        logger.info("Built RAG prompt for question=%s", question)

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

    async def ask_stream(self, question: str, limit: int = 5, min_similarity: float = 0.60):
        chunks = await self.retrieval_service.search(
            query=question,
            limit=limit,
            min_similarity=min_similarity
        )

        if not chunks:
            yield "I couldn't find the answer in the provided documents. Try rephrasing your question."
            return

        rag_prompt = build_rag_prompt(question, chunks)

        logger.info("Built RAG prompt for question=%s", question)

        stream = await self.client.interactions.create(
            model=settings.INTERACTION_MODEL,
            system_instruction=RAG_SYSTEM_INSTRUCTION,
            input=rag_prompt,
            stream=True
        )

        completed = False

        async for event in stream:
            if event.event_type == "step.delta":
                if event.delta.type == "text":
                    yield event.delta.text

            elif event.event_type == "interaction.completed":
                completed = event.interaction.status == "completed"
                logger.info("Interaction stream completed.")

        if not completed:
            raise err.IncompleteUpstreamError()