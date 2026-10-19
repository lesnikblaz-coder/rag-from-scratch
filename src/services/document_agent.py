import json

from google.genai.client import AsyncClient

from src.services.retrieval_service import RetrievalService
from src.config import settings
from src.agent.prompts.agent import AGENT_SYSTEM_INSTRUCTION
from src.agent.tools import definitions as df
from src.schemas import ai
from src.exceptions import custom as exc
from src.logging_config import logger


class DocumentAgent:
    def __init__(self, client: AsyncClient, retrieval_service: RetrievalService):
        self.client = client
        self.retrieval_service = retrieval_service

    async def ask(self, question: str) -> ai.AIAgentInteractionResponse:
        interaction = await self.client.interactions.create(
            model=settings.INTERACTION_MODEL,
            system_instruction=AGENT_SYSTEM_INSTRUCTION,
            input=question,
            tools=[df.SEARCH_DOCUMENT],
        )

        search_queries = []
        tool_call_count = 0

        for iteration in range(settings.MAX_AGENT_TOOL_ITERATIONS + 1):

            function_calls = [
                step for step in interaction.steps
                if step.type == "function_call"
            ]

            if not function_calls:
                return ai.AIAgentInteractionResponse(
                    answer=interaction.output_text,
                    search_queries=search_queries
                )

            if iteration == settings.MAX_AGENT_TOOL_ITERATIONS:
                raise exc.AIInteractionLimitError("Maximum agent tool iterations reached.")

            if tool_call_count + len(function_calls) > settings.MAX_AGENT_TOOL_CALLS:
                raise exc.AIInteractionLimitError("Maximum agent tool calls reached.")

            function_results = []

            for step in function_calls:
                if step.name != "search":
                    raise ValueError(f"Unknown function call: {step.name}")

                arguments = ai.SearchDocumentsArg.model_validate(step.arguments)

                result = await self.retrieval_service.search(
                    **arguments.model_dump()
                )

                # logger for checkign/validating LLM-generated citations
                retrieved_chunks = [
                    (chunk["source_file"], chunk["chunk_index"])
                    for chunk in result
                ]
                logger.info("retrieved_chunks = %s", retrieved_chunks)
                #

                search_queries.append(arguments.query)

                function_results.append({
                    "type": "function_result",
                    "name": step.name,
                    "call_id": step.id,
                    "result": [{
                        "type": "text",
                        "text": json.dumps(result, default=str)
                    }]
                })

            tool_call_count += len(function_calls)

            interaction = await self.client.interactions.create(
                model=settings.INTERACTION_MODEL,
                system_instruction=AGENT_SYSTEM_INSTRUCTION,
                tools=[df.SEARCH_DOCUMENT],
                previous_interaction_id=interaction.id,
                input=function_results
            )

        raise exc.AIInteractionLimitError("Maximum agent tool iterations reached.")