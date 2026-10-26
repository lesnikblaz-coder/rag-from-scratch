from fastapi import APIRouter

from src.schemas import ai
from src.schemas import api
from src.api import dependencies as dep

router = APIRouter(prefix="/v1")


# ask the standard RAG assistant
@router.post("/rag/ask", response_model=ai.AIResponseAndSource)
async def rag_ask(request: api.AskQuestionRequest, rag_service: dep.RAGServiceDep) -> ai.AIResponseAndSource:
    return await rag_service.ask(request.question)  # base values --> limit=5  min_similarity=0.60


# ask the agentic RAG assistant
@router.post("/agent/ask", response_model=ai.AIAgentInteractionResponse)
async def agent_ask(request: api.AskQuestionRequest,
                    document_agent: dep.DocumentAgentDep) -> ai.AIAgentInteractionResponse:
    return await document_agent.ask(request.question)