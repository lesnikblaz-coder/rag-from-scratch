from src.exceptions.base import AppException

class AIInteractionLimitError(AppException):
    status_code = 429
    detail = "The maximum number of AI tool interactions was exceeded."

class InvalidCitationError(AppException):
    detail = "Invalid citation usage by RAG Agent"

class IncompleteUpstreamError(AppException):
    """Raised when AI generation ends before completion."""

    status_code = 502
    detail = "AI response generation did not complete"