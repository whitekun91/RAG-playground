from typing import Any, Dict


class AppError(Exception):
    def __init__(self, message: str, code: str = "app_error", status: int = 400):
        super().__init__(message)
        self.code = code
        self.status = status


def error_response(err: Exception) -> Dict[str, Any]:
    if isinstance(err, AppError):
        return {"error": err.code, "message": str(err)}
    return {"error": "internal_error", "message": str(err)}


class ValidationError(AppError):
    def __init__(self, message: str = "Invalid input"):
        super().__init__(message, code="validation_error", status=422)


class NotFoundError(AppError):
    def __init__(self, message: str = "Resource not found"):
        super().__init__(message, code="not_found", status=404)


