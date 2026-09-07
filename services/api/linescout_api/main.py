"""FastAPI application factory and Uvicorn entry point.

Run one worker only: model weights (Milestone 4) live in GPU memory and GPU
inference is serialised through ``AppState.inference_lock``.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from starlette.types import ASGIApp

from linescout_api import __version__
from linescout_api.config import Settings, get_settings
from linescout_api.errors import (
    ApiError,
    api_error_handler,
    http_error_handler,
    json_error,
    resolve_request_id,
    validation_error_handler,
)
from linescout_api.routers import assets, curation, events, health, preferences, search
from linescout_api.state import build_state

log = logging.getLogger(__name__)

# Loopback-only Host names. ``testserver`` is Starlette's TestClient default;
# production traffic never uses it, and DNS-rebinding hosts are still rejected.
ALLOWED_HOSTS = {"127.0.0.1", "localhost", "::1", "testserver"}
_MUTATING = frozenset({"POST", "PUT", "PATCH", "DELETE"})


def _host_name(host_header: str) -> str:
    host = host_header.strip()
    if host.startswith("["):
        end = host.find("]")
        return host[1:end].lower() if end != -1 else host.lower()
    return host.split(":", 1)[0].lower()


class LoopbackSecurityMiddleware(BaseHTTPMiddleware):
    """Reject non-loopback Host headers and cross-origin mutations (DNS rebinding / CSRF)."""

    def __init__(self, app: ASGIApp, allowed_origins: Sequence[str]) -> None:
        super().__init__(app)
        self.allowed_origins = frozenset(allowed_origins)

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = resolve_request_id(request)
        request.state.request_id = request_id
        host = _host_name(request.headers.get("host", ""))
        if host not in ALLOWED_HOSTS:
            return json_error(
                403,
                "invalid_host",
                "Host header is not a loopback address",
                request_id=request_id,
                retryable=False,
            )
        if request.method in _MUTATING:
            origin = request.headers.get("origin")
            if origin and origin not in self.allowed_origins:
                return json_error(
                    403,
                    "cross_origin_mutation_forbidden",
                    "cross-origin mutation is forbidden",
                    request_id=request_id,
                    retryable=False,
                )
        response = await call_next(request)
        response.headers.setdefault("X-Request-Id", str(request_id))
        return response


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    logging.basicConfig(
        level=settings.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        state = build_state(settings)
        app.state.linescout = state
        log.info(
            "LineScout API %s | device=%s%s | gallery=%s (%d assets) | ready=%s | fixture=%s",
            __version__,
            state.device.kind,
            f" ({state.device.gpu_name}, {state.device.vram_total_mb} MB)"
            if state.device.is_cuda
            else "",
            state.gallery.dataset_version if state.gallery else "none",
            len(state.assets),
            state.ready,
            settings.fixture_mode,
        )
        for warning in state.warnings:
            log.warning(warning)
        if state.setup_error:
            log.error("SETUP ERROR: %s", state.setup_error)
        try:
            yield
        finally:
            state.connection.close()

    app = FastAPI(
        title="LineScout API",
        version=__version__,
        description="Local line-art reference retrieval for character artists.",
        lifespan=lifespan,
        openapi_url="/api/v1/openapi.json",
        docs_url="/api/v1/docs",
        redoc_url=None,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST", "PUT", "OPTIONS"],
        allow_headers=["*"],
    )
    app.add_middleware(LoopbackSecurityMiddleware, allowed_origins=settings.cors_origins)
    app.add_exception_handler(ApiError, api_error_handler)
    app.add_exception_handler(HTTPException, http_error_handler)
    app.add_exception_handler(RequestValidationError, validation_error_handler)

    prefix = "/api/v1"
    app.include_router(health.router, prefix=prefix)
    app.include_router(search.router, prefix=prefix)
    app.include_router(events.router, prefix=prefix)
    app.include_router(preferences.router, prefix=prefix)
    app.include_router(assets.router, prefix=prefix)
    if settings.curation_mode:
        app.include_router(curation.router, prefix=prefix)
    return app


def run() -> None:
    settings = get_settings()
    uvicorn.run(
        "linescout_api.main:create_app",
        factory=True,
        host=settings.host,
        port=settings.port,
        workers=1,
        log_level=settings.log_level,
    )


if __name__ == "__main__":
    run()
