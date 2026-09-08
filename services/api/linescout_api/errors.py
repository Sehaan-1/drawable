"""Structured error responses shared by every router."""

from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

from linescout_api.schemas import ErrorDetail, ErrorResponse

RETRYABLE_STATUS = frozenset({429, 503})


def resolve_request_id(request: Request | None, explicit: UUID | None = None) -> UUID:
    if explicit is not None:
        return explicit
    if request is not None:
        incoming = request.headers.get("x-request-id")
        if incoming:
            try:
                return UUID(incoming.strip())
            except ValueError:
                pass
        existing = getattr(request.state, "request_id", None)
        if isinstance(existing, UUID):
            return existing
    return uuid4()


def error_body(
    *,
    code: str,
    message: str,
    field: str | None = None,
    status_code: int,
    request_id: UUID | None = None,
    retryable: bool | None = None,
    details: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, str]]:
    rid = request_id or uuid4()
    retry = retryable if retryable is not None else status_code in RETRYABLE_STATUS
    payload = ErrorResponse(
        schema_version=1,
        request_id=rid,
        retryable=retry,
        error=ErrorDetail(code=code, message=message, field=field, details=details),
    )
    return payload.model_dump(mode="json"), {"X-Request-Id": str(rid)}


def json_error(
    status_code: int,
    code: str,
    message: str,
    *,
    field: str | None = None,
    request_id: UUID | None = None,
    retryable: bool | None = None,
    headers: dict[str, str] | None = None,
    details: dict[str, Any] | None = None,
) -> JSONResponse:
    content, extra = error_body(
        code=code,
        message=message,
        field=field,
        status_code=status_code,
        request_id=request_id,
        retryable=retryable,
        details=details,
    )
    return JSONResponse(
        status_code=status_code, content=content, headers={**(headers or {}), **extra}
    )


class ApiError(HTTPException):
    """HTTPException whose detail is always an :class:`ErrorResponse` payload."""

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        field: str | None = None,
        headers: dict[str, str] | None = None,
        retryable: bool | None = None,
        request_id: UUID | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(status_code=status_code, detail=message, headers=headers)
        self.code = code
        self.message = message
        self.field = field
        self.retryable = retryable
        self.request_id = request_id
        self.details = details

    def to_response(self, request: Request | None = None) -> JSONResponse:
        content, extra = error_body(
            code=self.code,
            message=self.message,
            field=self.field,
            status_code=self.status_code,
            request_id=resolve_request_id(request, self.request_id),
            retryable=self.retryable,
            details=self.details,
        )
        headers = {**(self.headers or {}), **extra}
        return JSONResponse(status_code=self.status_code, content=content, headers=headers)


def bad_request(code: str, message: str, field: str | None = None) -> ApiError:
    return ApiError(400, code, message, field, retryable=False)


def too_large(
    code: str,
    message: str,
    field: str | None = None,
    details: dict[str, Any] | None = None,
) -> ApiError:
    return ApiError(413, code, message, field, retryable=False, details=details)


def unprocessable(code: str, message: str, field: str | None = None) -> ApiError:
    return ApiError(422, code, message, field, retryable=False)


def not_found(code: str, message: str) -> ApiError:
    return ApiError(404, code, message, retryable=False)


def conflict(code: str, message: str, field: str | None = None) -> ApiError:
    return ApiError(409, code, message, field, retryable=False)


def service_unavailable(code: str, message: str, field: str | None = None) -> ApiError:
    return ApiError(503, code, message, field, headers={"Retry-After": "5"}, retryable=True)


async def api_error_handler(request: Request, error: Exception) -> JSONResponse:
    assert isinstance(error, ApiError)
    return error.to_response(request)


async def http_error_handler(request: Request, error: Exception) -> JSONResponse:
    assert isinstance(error, HTTPException)
    if isinstance(error, ApiError):
        return error.to_response(request)
    retryable = error.status_code in RETRYABLE_STATUS
    raw_headers = error.headers
    headers = {str(key): str(value) for key, value in raw_headers.items()} if raw_headers else None
    return json_error(
        error.status_code,
        f"http_{error.status_code}",
        str(error.detail),
        request_id=resolve_request_id(request),
        retryable=retryable,
        headers=headers,
    )


async def validation_error_handler(request: Request, error: Exception) -> JSONResponse:
    assert isinstance(error, RequestValidationError)
    errors = error.errors()
    first = errors[0] if errors else {}
    location = ".".join(
        str(part) for part in first.get("loc", ()) if part not in ("body", "query", "path")
    )
    # ``details.errors`` carries every failed field, not just the first, so
    # clients can mark up the whole form in one round trip.
    detail_errors = [
        {
            "field": ".".join(str(part) for part in item.get("loc", ()))
            if item.get("loc")
            else None,
            "message": str(item.get("msg", "invalid value")),
            "type": str(item.get("type", "value_error")),
        }
        for item in errors
    ]
    return json_error(
        422,
        "validation_error",
        str(first.get("msg", "invalid request")),
        field=location or None,
        request_id=resolve_request_id(request),
        retryable=False,
        details={"errors": detail_errors} if detail_errors else None,
    )
