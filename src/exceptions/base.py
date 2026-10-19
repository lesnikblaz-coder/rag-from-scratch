class AppException(Exception):
    """
    Base class for application exceptions.
    """

    status_code = 500
    detail = "An unexpected error occurred."

    def __init__(self, detail: str | None = None):
        self.detail = detail or self.detail
        super().__init__(self.detail)