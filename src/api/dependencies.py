from typing import Annotated
from fastapi import Request, Depends

from src.db.base import AsyncSessionLocal
from src.services.rag_service import RAGService
from src.services.document_agent import DocumentAgent
from src.services.retrieval_service import RetrievalService
from src.services.embedding_service import EmbeddingService


# ---------- SERVICES ----------
async def _get_embedding_service(request: Request):
    return EmbeddingService(
        client=request.app.state.aclient
    )

EmbeddingServiceDep = Annotated[EmbeddingService, Depends(_get_embedding_service)]


async def _get_retrieval_service(embedding_service: EmbeddingServiceDep):
    return RetrievalService(
        embedding_service=embedding_service,
        session_factory=AsyncSessionLocal
    )

RetrievalServiceDep = Annotated[RetrievalService, Depends(_get_retrieval_service)]


# ---------- standard RAG ----------
async def _get_rag_service(retrieval_service: RetrievalServiceDep, request: Request):
    return RAGService(
        retrieval_service=retrieval_service,
        client=request.app.state.aclient
    )

RAGServiceDep = Annotated[RAGService, Depends(_get_rag_service)]


# ---------- agentic RAG ----------
async def _get_document_agent(retrieval_service: RetrievalServiceDep, request: Request):
    return DocumentAgent(
        retrieval_service=retrieval_service,
        client=request.app.state.aclient
    )

DocumentAgentDep = Annotated[DocumentAgent, Depends(_get_document_agent)]