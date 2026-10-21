import json

from google.genai.client import AsyncClient

from src.services.retrieval_service import RetrievalService
from src.config import settings
from src.agent.prompts.agent import AGENT_SYSTEM_INSTRUCTION
from src.agent.tools import definitions as df
from src.schemas import ai
from src.exceptions import custom as exc
from src.services.citation_validation import validate_citations


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
            generation_config={"thinking_level": "minimal"}
        )

        search_queries = []
        tool_call_count: int = 0
        evidence_found: bool = False
        evidence_registry: dict[str, dict] = {}
        matches: list[dict] = []

        for iteration in range(settings.MAX_AGENT_TOOL_ITERATIONS + 1):

            function_calls = [
                step for step in interaction.steps
                if step.type == "function_call"
            ]

            if not function_calls:
                if not evidence_found:
                    return ai.AIAgentInteractionResponse(
                        answer="I couldn't find enough information in the documents to answer your question.",
                        search_queries=search_queries,
                        cited_sources=[]
                    )

                # validate citations -> raises error if invalid
                cited_sources = validate_citations(interaction.output_text, evidence_registry)

                return ai.AIAgentInteractionResponse(
                    answer=interaction.output_text,
                    search_queries=search_queries,
                    cited_sources=sorted(cited_sources, key=lambda x: x[1])
                )

            if iteration == settings.MAX_AGENT_TOOL_ITERATIONS:
                raise exc.AIInteractionLimitError("Maximum ai tool iterations reached.")

            if tool_call_count + len(function_calls) > settings.MAX_AGENT_TOOL_CALLS:
                raise exc.AIInteractionLimitError("Maximum ai tool calls reached.")

            function_results = []

            for step in function_calls:
                if step.name != "search":
                    raise ValueError(f"Unknown function call: {step.name}")

                arguments = ai.SearchDocumentsArg.model_validate(step.arguments)

                result = await self.retrieval_service.search(
                    **arguments.model_dump()
                )

                if result:
                    evidence_found = True
                    start_index = len(evidence_registry) + 1

                    for i, r in enumerate(result, start=start_index):
                        citation_id = f"S{i}"
                        evidence_registry[citation_id] = r

                        key = (r["source_file"], r["chunk_index"])
                        for m in matches:
                            if (m["source_file"], m["chunk_index"]) == key:
                                m.update(**r)
                                break
                        else:  # runs if break is not hit
                            r["citation_id"] = citation_id
                            matches.append({**r})

                search_queries.append(arguments.query)
                payload = {"matches": matches}

                function_results.append({
                    "type": "function_result",
                    "name": step.name,
                    "call_id": step.id,
                    "result": [{
                        "type": "text",
                        "text": json.dumps(payload, default=str)
                    }]
                })

            tool_call_count += len(function_calls)

            interaction = await self.client.interactions.create(
                model=settings.INTERACTION_MODEL,
                system_instruction=AGENT_SYSTEM_INSTRUCTION,
                tools=[df.SEARCH_DOCUMENT],
                previous_interaction_id=interaction.id,
                input=function_results,
                generation_config={"thinking_level": "minimal"}
            )

        raise exc.AIInteractionLimitError("Maximum ai tool iterations reached.")
