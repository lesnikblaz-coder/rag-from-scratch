from pydantic import BaseModel


class Source(BaseModel):
    source_file: str
    chunk_index: int
    similarity: float

class AIInteractionResponse(BaseModel):
    answer: str
    sources: list[Source]