import logging

from fastapi import Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from backend.exceptions import (
    AppError,
    ConflictError,
    ExternalServiceError,
    NotFoundError,
    ValidationError,
)

logger = logging.getLogger(__name__)


def _error_response(
    request: Request,
    status_code: int,
    code: str,
    message: str,
    details: dict | None = None,
) -> JSONResponse:
    request_id = getattr(request.state, "request_id", None)
    return JSONResponse(
        status_code=status_code,
        content={
            "error": code,
            "message": message,
            "details": details or {},
            **({"request_id": request_id} if request_id else {}),
        },
    )


async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    assert isinstance(exc, AppError)

    if isinstance(exc, NotFoundError):
        status_code = status.HTTP_404_NOT_FOUND
    elif isinstance(exc, ConflictError):
        status_code = status.HTTP_409_CONFLICT
    elif isinstance(exc, ValidationError):
        status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
    elif isinstance(exc, ExternalServiceError):
        status_code = status.HTTP_502_BAD_GATEWAY
    else:
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        logger.error(f"Unclassified AppError: {exc}", exc_info=True)

    return _error_response(
        request=request,
        status_code=status_code,
        code=exc.code,
        message=exc.message,
        details=exc.details,
    )


async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    fields = []
    for error in exc.errors():
        location = ".".join(str(loc) for loc in error["loc"])
        fields.append({"field": location, "message": error["msg"]})

    return _error_response(
        request=request,
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        code="VALIDATION_ERROR",
        message="Request validation failed",
        details={"fields": fields},
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    request_id = getattr(request.state, "request_id", "unknown")
    logger.error(
        f"Unhandled exception [{request_id}] on {request.method} {request.url.path}: {exc}",
        exc_info=True,
    )
    return _error_response(
        request=request,
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        code="INTERNAL_ERROR",
        message="An unexpected error occurred.",
        details={"request_id": request_id},
    )


async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    return _error_response(
        request=request,
        status_code=exc.status_code,
        code="HTTP_ERROR",
        message=str(exc.detail),
    )
