import pytest
import json

from unittest.mock import AsyncMock, MagicMock
from types import SimpleNamespace
from google import genai

from src.services.document_agent import DocumentAgent
from src.services.retrieval_service import RetrievalService


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


async def test_sequential_retrieval(
        document_agent,
        mock_gemini_sdk,
        mock_retrieval_service
):
    first_response = SimpleNamespace(
        id="interaction-1",
        steps=[
            SimpleNamespace(
                type="function_call",
                name="search",
                id="call-1",
                arguments={
                    "query": "annual leave policy"
                }
            )
        ],
        output_text=None
    )

    second_response = SimpleNamespace(
        id="interaction-2",
        steps=[
            SimpleNamespace(
                type="function_call",
                name="search",
                id="call-2",
                arguments={
                    "query": "sick leave policy"
                }
            )
        ],
        output_text=None
    )

    third_response = SimpleNamespace(
        id="interaction-3",
        steps=[],
        output_text="final answer"
    )

    mock_retrieval_service.search.side_effect = [
        [
            {
                "source_file": "handbook.txt",
                "chunk_index": 1,
                "content": "Employees receive 25 annual leave days."
            }
        ],
        [
            {
                "source_file": "handbook.txt",
                "chunk_index": 5,
                "content": "Employees must notify their manager when sick."
            }
        ]
    ]

    mock_gemini_sdk.interactions.create.side_effect = [
        first_response,
        second_response,
        third_response
    ]

    result = await document_agent.ask(
        question="What is the leave policy?"
    )

    assert result.answer == "final answer"
    assert result.search_queries == [
        "annual leave policy",
        "sick leave policy"
    ]

    assert mock_retrieval_service.search.await_count == 2

    search_calls = mock_retrieval_service.search.await_args_list

    assert search_calls[0].kwargs["query"] == "annual leave policy"
    assert search_calls[1].kwargs["query"] == "sick leave policy"

    calls = mock_gemini_sdk.interactions.create.await_args_list

    assert calls[1].kwargs["previous_interaction_id"] == "interaction-1"
    assert calls[2].kwargs["previous_interaction_id"] == "interaction-2"

    function_result_1 = calls[1].kwargs["input"]
    function_result_2 = calls[2].kwargs["input"]

    assert function_result_1[0]["call_id"] == "call-1"
    assert function_result_1[0]["type"] == "function_result"
    assert function_result_1[0]["name"] == "search"

    assert function_result_2[0]["call_id"] == "call-2"
    assert function_result_2[0]["type"] == "function_result"
    assert function_result_2[0]["name"] == "search"

    parsed_result_1 = json.loads(function_result_1[0]["result"][0]["text"])
    parsed_result_2 = json.loads(function_result_2[0]["result"][0]["text"])

    assert parsed_result_1[0]["content"] == "Employees receive 25 annual leave days."
    assert parsed_result_2[0]["content"] == "Employees must notify their manager when sick."

async def test_unknown_tool_call(
        document_agent,
        mock_gemini_sdk,
        mock_retrieval_service
):
    response = SimpleNamespace(
        id="interaction-1",
        steps=[
            SimpleNamespace(
                type="function_call",
                name="delete_documents",
                id="call-1",
                arguments={
                    "query": "delete all documents"
                }
            )
        ],
        output_text=None
    )

    mock_gemini_sdk.interactions.create.return_value = response

    with pytest.raises(ValueError, match="Unknown function call"):
        await document_agent.ask(
            question="Please delete document."
        )

    mock_gemini_sdk.interactions.create.assert_awaited_once()

    mock_retrieval_service.search.assert_not_awaited()