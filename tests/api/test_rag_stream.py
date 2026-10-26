BASE_PREFIX = "/v1"


async def test_successful_request(
        client,
        mock_rag_service
):
    async def example():
        yield "Hello"
        yield " "
        yield "world"

    async def response():
        async for e in example():
            yield e

    mock_rag_service.ask_stream.return_value = response()

    response = await client.post(
        url=BASE_PREFIX+"/rag/stream",
        json={"question": "hello"}
    )

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]

    body = response.text

    hello_pos = body.index("Hello")
    world_pos = body.index("world")

    assert hello_pos < world_pos
    assert "Hello" in body
    assert "world" in body
    assert "event: token" in body
    assert "event: done" in body