import pytest
import httpx

from google import genai
from unittest.mock import MagicMock, AsyncMock

from src.services.retrieval_service import RetrievalService
from src.services.document_agent import DocumentAgent
from src.services.rag_service import RAGService
from src.main import app
from src.api import dependencies as dep


@pytest.fixture
def mock_rag_service():
    return AsyncMock(spec=RAGService)

@pytest.fixture
def mock_rag_agent():
    return AsyncMock(spec=DocumentAgent)

@pytest.fixture
async def client(mock_rag_agent, mock_rag_service):

    app.dependency_overrides[dep._get_rag_service] = lambda: mock_rag_service
    app.dependency_overrides[dep._get_document_agent] = lambda: mock_rag_agent

    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test"
        ) as ac:
            yield ac

    finally:
        app.dependency_overrides.clear()

@pytest.fixture
def mock_gemini_sdk():
    client = MagicMock(spec=genai.client.AsyncClient)
    client.interactions.create = AsyncMock()
    return client

@pytest.fixture
def mock_retrieval_service():
    return AsyncMock(spec=RetrievalService)

@pytest.fixture
def document_agent(mock_gemini_sdk, mock_retrieval_service):
    return DocumentAgent(
        client=mock_gemini_sdk,
        retrieval_service=mock_retrieval_service
    )