import pytest

from types import SimpleNamespace

from src.exceptions import custom as err


async def test_valid_citation(
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
                    "query": "something something"
                }
            )
        ],
        output_text=None
    )

    second_response = SimpleNamespace(
        id="interaction-2",
        steps=[],
        output_text="something something [S1]"
    )

    mock_retrieval_service.search.return_value = [
        {
            "source_file": "sample.txt",
            "chunk_index": 4,
            "content": "..."
        }
    ]

    mock_gemini_sdk.interactions.create.side_effect = [
        first_response,
        second_response
    ]

    result = await document_agent.ask("something something")

    assert result.answer == "something something [S1]"
    assert result.search_queries == ['something something']
    assert result.cited_sources == ['S1']

    mock_retrieval_service.search.assert_awaited_once()

    assert mock_gemini_sdk.interactions.create.await_count == 2


async def test_invalid_citation(
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
                    "query": "something something"
                }
            )
        ],
        output_text=None
    )

    second_response = SimpleNamespace(
        id="interaction-2",
        steps=[],
        output_text="something invalid [S99]"
    )

    mock_retrieval_service.search.return_value = [
        {
            "source_file": "sample.txt",
            "chunk_index": 4,
            "content": "..."
        }
    ]

    mock_gemini_sdk.interactions.create.side_effect = [
        first_response,
        second_response
    ]

    with pytest.raises(err.InvalidCitationError):
        await document_agent.ask("something something")

    mock_retrieval_service.search.assert_awaited_once()

    assert mock_gemini_sdk.interactions.create.await_count == 2


async def test_missing_citations(
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
                    "query": "something something"
                }
            )
        ],
        output_text=None
    )

    second_response = SimpleNamespace(
        id="interaction-2",
        steps=[],
        output_text="something something [missing citation]"
    )

    mock_retrieval_service.search.return_value = [
        {
            "source_file": "sample.txt",
            "chunk_index": 4,
            "content": "..."
        }
    ]

    mock_gemini_sdk.interactions.create.side_effect = [
        first_response,
        second_response
    ]

    with pytest.raises(err.InvalidCitationError):
        await document_agent.ask("something something")

    mock_retrieval_service.search.assert_awaited_once()

    assert mock_gemini_sdk.interactions.create.await_count == 2


async def test_multiple_citations(
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
                    "query": "something something"
                }
            )
        ],
        output_text=None
    )

    second_response = SimpleNamespace(
        id="interaction-2",
        steps=[],
        output_text="something something [S1] [S2] [S3]"
    )

    mock_retrieval_service.search.return_value = [
        {
            "source_file": "sample.txt",
            "chunk_index": 4,
            "content": "..."
        },
        {
            "source_file": "sample.txt",
            "chunk_index": 2,
            "content": "..."
        },
        {
            "source_file": "sample.txt",
            "chunk_index": 5,
            "content": "..."
        },
    ]

    mock_gemini_sdk.interactions.create.side_effect = [
        first_response,
        second_response
    ]

    result = await document_agent.ask("something something")

    assert result.answer == "something something [S1] [S2] [S3]"
    assert result.search_queries == ['something something']
    assert result.cited_sources == ['S1', 'S2', 'S3']

    mock_retrieval_service.search.assert_awaited_once()

    assert mock_gemini_sdk.interactions.create.await_count == 2