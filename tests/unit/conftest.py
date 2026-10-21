import pytest

from google import genai
from unittest.mock import MagicMock, AsyncMock

from src.services.retrieval_service import RetrievalService
from src.services.document_agent import DocumentAgent


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