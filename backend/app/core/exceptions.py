"""
Global Exception Handlers and Custom Exception Definitions.
"""

from fastapi import HTTPException, status


class ModelNotFoundError(HTTPException):
    def __init__(self, detail: str = "ML model artifact or metadata could not be found."):
        super().__init__(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=detail)


class DatabaseConnectionError(HTTPException):
    def __init__(self, detail: str = "Database connection failed."):
        super().__init__(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=detail)


class ValidationError(HTTPException):
    def __init__(self, detail: str = "Invalid input dataset schema or values."):
        super().__init__(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=detail)


class UnauthorizedError(HTTPException):
    def __init__(self, detail: str = "Could not validate credentials"):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            headers={"WWW-Authenticate": "Bearer"},
        )


class ForbiddenError(HTTPException):
    def __init__(self, detail: str = "User does not have required permissions"):
        super().__init__(status_code=status.HTTP_403_FORBIDDEN, detail=detail)
