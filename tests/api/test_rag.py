BASE_PREFIX = "/v1"


async def test_successful_request(
        client,
        mock_rag_service
):
    rag_service_response = {
        "answer": "hi",
        "sources": []
    }

    mock_rag_service.ask.return_value =  rag_service_response

    response = await client.post(
        url=BASE_PREFIX+"/rag/ask",
        json={"question": "hello"}
    )

    assert response.status_code == 200
    assert response.json() == rag_service_response

async def test_invalid_request(
        client
):
    response = await client.post(
        url=BASE_PREFIX + "/rag/ask",
        json={"question": ""}
    )

    assert response.status_code == 422

async def test_agent_fallback(
        client,
        mock_rag_agent
):
    agent_fallback = {
        "answer": "I couldn't find enough information in the documents to answer your question.",
        "search_queries": [],
        "cited_sources": []
    }

    mock_rag_agent.ask.return_value = agent_fallback

    response = await client.post(
        url=BASE_PREFIX + "/agent/ask",
        json={"question": "hello"}
    )

    assert response.status_code == 200
    assert response.json() == agent_fallback