from pydantic import BaseModel


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