import asyncio
import pytest
from unittest.mock import AsyncMock

from src.config import settings


async def test_timeout_stream(
        mock_gemini_sdk,
        mock_retrieval_service,
        rag_service,
        monkeypatch
):
    monkeypatch.setattr(
        settings,
        "RAG_STREAM_TIMEOUT_SECONDS",
        0.05
    )

    mock_retrieval_service.search.return_value = [{
        "source_file": "random.txt",
        "chunk_index": 1,
        "content": "random",
        "rrf_score": 0.6
    }]

    async def slow_stream():
        await asyncio.sleep(1)
        yield

    mock_gemini_sdk.interactions.create.return_value = slow_stream()

    with pytest.raises(TimeoutError):
        async for _ in rag_service.ask_stream("hello"):
            pass

async def test_timeout_interaction(
        rag_service,
        mock_retrieval_service,
        mock_gemini_sdk,
        monkeypatch
):
    monkeypatch.setattr(
        settings,
        "RAG_INTERACTION_TIMEOUT_SECONDS",
        0.05
    )

    mock_retrieval_service.search.return_value = [{
        "source_file": "random.txt",
        "chunk_index": 1,
        "content": "random",
        "rrf_score": 0.6
    }]

    async def slow_interaction(*args, **kwargs):
        await asyncio.sleep(1)

    mock_gemini_sdk.interactions.create = AsyncMock(
        side_effect=slow_interaction
    )

    with pytest.raises(TimeoutError):
        await rag_service.ask("hello")