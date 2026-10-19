SEARCH_DOCUMENT = {
    "type": "function",
    "name": "search",
    "description": "Embeds given query, does a semantic similarity search with the query embedding, does a keyword search with PostgreSQL, both return highest scored chunks. Semantic chunks are filtered by min_similarity value (0.60). Lastly, chunk fusion is made with reciprocal rank fusion between the semantic and keyword chunks. Fused chunks are returned. Can be called multiple times with different search queries when more information is needed.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Query used for the semantic and keyword search"
            },
            "limit": {
                "type": "integer",
                "description": "Upper limit of how many of the top chunks are returned by semantic and keyword searches",
                "minimum": 1,
                "default": 5,
                "maximum": 10
            }
        },
        "required": ["query"],
    }
}