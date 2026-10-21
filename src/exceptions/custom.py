from src.exceptions.base import AppException

class AIInteractionLimitError(AppException):
    status_code = 429
    detail = "The maximum number of AI tool interactions was exceeded."

class InvalidCitationError(AppException):
    detail = "Invalid citation usage by RAG Agent"