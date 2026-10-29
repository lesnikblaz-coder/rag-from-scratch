import asyncio
import time

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

        async with asyncio.timeout(settings.RAG_INTERACTION_TIMEOUT_SECONDS):
            start = time.perf_counter()

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
            duration_ms = (time.perf_counter() - start) * 1000

            logger.info(
                "Gemini generation completed duration_ms=%.2f",
                duration_ms
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

        async with asyncio.timeout(settings.RAG_STREAM_TIMEOUT_SECONDS):
            start = time.perf_counter()

            stream = await self.client.interactions.create(
                model=settings.INTERACTION_MODEL,
                system_instruction=RAG_SYSTEM_INSTRUCTION,
                input=rag_prompt,
                stream=True
            )

            completed = False
            first_yield_logged = False

            async for event in stream:
                if event.event_type == "step.delta":
                    if event.delta.type == "text":

                        if not first_yield_logged:
                            first_yield_duration_ms = (time.perf_counter() - start) * 1000

                            logger.info(
                                "First yield of stream duration_ms=%.2f",
                                first_yield_duration_ms
                            )

                            first_yield_logged = True

                        yield event.delta.text

                elif event.event_type == "interaction.completed":
                    completed = event.interaction.status == "completed"
                    logger.info("Interaction stream completed.")

                    finished_stream_duration_ms = (time.perf_counter() - start) * 1000

                    logger.info(
                        "Finished stream duration_ms=%.2f",
                        finished_stream_duration_ms
                    )

            if not completed:
                unfinished_stream_duration_ms = (time.perf_counter() - start) * 1000

                logger.info(
                    "Unfinished stream duration_ms=%.2f",
                    unfinished_stream_duration_ms
                )

                raise err.IncompleteUpstreamError()