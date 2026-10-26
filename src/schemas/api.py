from pydantic import BaseModel, Field

class AskQuestionRequest(BaseModel):
    question: str = Field(min_length=1)