from dataclasses import dataclass, field


@dataclass
class AppError(Exception):
    message: str
    code: str = "INTERNAL_ERROR"
    details: dict = field(default_factory=dict)

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"


class NotFoundError(AppError):
    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message=message, code="NOT_FOUND", details=details or {})


class ConflictError(AppError):
    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message=message, code="CONFLICT", details=details or {})


class ValidationError(AppError):
    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message=message, code="VALIDATION_ERROR", details=details or {})


class ExternalServiceError(AppError):
    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message=message, code="EXTERNAL_SERVICE_ERROR", details=details or {})
