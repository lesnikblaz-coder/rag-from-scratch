from pydantic import BaseModel

class AskQuestionRequest(BaseModel):
    question: str