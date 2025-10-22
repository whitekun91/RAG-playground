from typing import Dict, Any

class AppError(Exception):
    def __init__(self, message: str, status: int = 500):
        self.message = message
        self.status = status
        super().__init__(message)

def error_response(error: Exception) -> Dict[str, Any]:
    if isinstance(error, AppError):
        return {"error": error.message, "status": error.status}
    else:
        return {"error": str(error), "status": 500}
