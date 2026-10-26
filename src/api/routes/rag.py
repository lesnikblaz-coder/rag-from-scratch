from fastapi import APIRouter
from fastapi.sse import EventSourceResponse, ServerSentEvent
from collections.abc import AsyncIterable

from src.schemas import ai
from src.schemas import api
from src.api import dependencies as dep

router = APIRouter(prefix="/v1")


# standard RAG assistant
@router.post("/rag/ask", response_model=ai.AIResponseAndSource)
async def rag_ask(request: api.AskQuestionRequest, rag_service: dep.RAGServiceDep) -> ai.AIResponseAndSource:
    return await rag_service.ask(request.question)  # base values --> limit=5  min_similarity=0.60


@router.post("/rag/stream", response_class=EventSourceResponse)
async def rag_stream(request: api.AskQuestionRequest, rag_service: dep.RAGServiceDep) -> AsyncIterable[ServerSentEvent]:
    try:
        async for s in rag_service.ask_stream(request.question):
            yield ServerSentEvent(data=s, event="token")
        yield ServerSentEvent(raw_data="{}", event="done")

    except Exception:
        yield ServerSentEvent(
            data="An error occurred while generating the answer.",
            event="error"
        )


# agentic RAG assistant
@router.post("/agent/ask", response_model=ai.AIAgentInteractionResponse)
async def agent_ask(request: api.AskQuestionRequest,
                    document_agent: dep.DocumentAgentDep) -> ai.AIAgentInteractionResponse:
    return await document_agent.ask(request.question)