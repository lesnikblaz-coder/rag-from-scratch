import pytest
import json

from types import SimpleNamespace

from src.exceptions import custom as err


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
        output_text="final answer [S1]"
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

    assert result.answer == "final answer [S1]"
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

    assert parsed_result_1["matches"][0]["content"] == "Employees receive 25 annual leave days."
    assert parsed_result_2["matches"][0]["content"] == "Employees must notify their manager when sick."

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

async def test_no_evidence_returns_fallback(
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
                    "query": "2022 World Cup winner"
                }
            )
        ],
        output_text=None
    )

    second_response = SimpleNamespace(
        id="interaction-2",
        steps=[],
        output_text="Argentina won the 2022 World Cup."
    )

    mock_retrieval_service.search.return_value = []

    mock_gemini_sdk.interactions.create.side_effect = [
        first_response,
        second_response
    ]

    result = await document_agent.ask(
        question="Who won the 2022 World Cup?"
    )

    assert result.answer == "I couldn't find enough information in the documents to answer your question."
    assert result.search_queries == ["2022 World Cup winner"]

async def test_unsupported_document_based_answer(
        document_agent,
        mock_gemini_sdk,
        mock_retrieval_service
):
    first_response = SimpleNamespace(
        id="interaction-1",
        steps=[],
        output_text="Unsupported document based answer that makes no sense and I made up!"
    )

    mock_gemini_sdk.interactions.create.return_value = first_response

    result = await document_agent.ask(
        "Make up an answer!"
    )

    assert result.answer == "I couldn't find enough information in the documents to answer your question."
    assert not result.search_queries

async def test_two_tool_calls_one_interaction(
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
                    "query": "annual leave"
                }
            ),
            SimpleNamespace(
                type="function_call",
                name="search",
                id="call-2",
                arguments={
                    "query": "sick leave"
                }
            )
        ],
        output_text=None
    )

    second_response = SimpleNamespace(
        id="interaction-2",
        steps=[],
        output_text="asd"
    )

    mock_gemini_sdk.interactions.create.side_effect = [
        first_response,
        second_response
    ]

    mock_retrieval_service.search.side_effect = [
        [
            {
                "source_file": "handbook.txt",
                "chunk_index": 4,
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

    with pytest.raises(err.InvalidCitationError):
        await document_agent.ask("compare annual leave with sick leave")

    second_interaction = mock_gemini_sdk.interactions.create.call_args_list[1]

    second_interaction_input = second_interaction.kwargs["input"]

    second_interaction_input_first = json.loads(second_interaction_input[0]["result"][0]["text"])
    second_interaction_input_second = json.loads(second_interaction_input[1]["result"][0]["text"])

    assert second_interaction_input_first["matches"][0]["citation_id"] == "S1"
    assert second_interaction_input_second["matches"][0]["citation_id"] == "S2"