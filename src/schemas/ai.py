from pydantic import BaseModel, ConfigDict, Field


class Source(BaseModel):
    source_file: str
    chunk_index: int
    similarity: float | None = None
    rrf_score: float

class AIInteractionResponse(BaseModel):
    answer: str

class AIResponseAndSource(BaseModel):
    answer: str
    sources: list[Source]

class AIAgentInteractionResponse(BaseModel):
    answer: str | None
    search_queries: list[str]


# function calling restrictions
class AIArgumentsValidation(BaseModel):
    model_config = ConfigDict(extra="forbid")

class SearchDocumentsArg(AIArgumentsValidation):
    query: str
    limit: int = Field(default=5, ge=1, le=10)