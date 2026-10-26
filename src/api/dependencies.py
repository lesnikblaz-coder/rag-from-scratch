from typing import Annotated
from fastapi import Request, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.base import AsyncSessionLocal
from src.services.rag_service import RAGService
from src.services.document_agent import DocumentAgent
from src.services.retrieval_service import RetrievalService
from src.services.embedding_service import EmbeddingService

from src.repos.chunk_repo import ChunkRepo


# ---------- SESSION ----------
async def _get_session():
    async with AsyncSessionLocal.begin() as session:
        yield session

SessionDep = Annotated[AsyncSession, Depends(_get_session)]


# ---------- REPOSITORIES ----------
async def _get_chunk_repo(session: SessionDep):
    return ChunkRepo(
        session=session
    )

ChunkRepoDep = Annotated[ChunkRepo, Depends(_get_chunk_repo)]


# ---------- SERVICES ----------
async def _get_embedding_service(request: Request):
    return EmbeddingService(
        client=request.app.state.aclient
    )

EmbeddingServiceDep = Annotated[EmbeddingService, Depends(_get_embedding_service)]


async def _get_retrieval_service(embedding_service: EmbeddingServiceDep, chunk_repo: ChunkRepoDep):
    return RetrievalService(
        embedding_service=embedding_service,
        chunk_repo=chunk_repo
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