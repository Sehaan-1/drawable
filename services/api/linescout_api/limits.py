"""Receive-layer request bounds: total body bytes and multipart part count.

This middleware is the outermost layer of the ASGI stack (added last in
:func:`linescout_api.main.create_app`) so it runs *before* Starlette buffers,
parses, or spools anything — including before ``BaseHTTPMiddleware`` layers,
which eagerly cache the request body in recent Starlette versions.

Envelope
--------
The total raw body allowance is ``Settings.max_request_bytes``: the 4 MiB
snapshot image + 256 KiB of compressed strokes + a fixed multipart overhead
slack. ``Content-Length`` is used only as an early-rejection optimization;
the real guarantee comes from counting every ``http.request`` message body,
so chunked transfer-encoding and forged/missing ``Content-Length`` headers
cannot bypass it. For ``multipart/form-data`` the number of parts is bounded
as well, by counting boundary delimiters — the same rule the multipart
parser itself applies (a boundary byte string inside a payload would be a
real delimiter for the parser anyway).

Rejections use the standard structured error envelope
(:class:`linescout_api.schemas.ErrorResponse`): 413 ``request_too_large``,
413 ``too_many_parts``, and 400 ``invalid_content_length``.
"""

from __future__ import annotations

from starlette.datastructures import Headers
from starlette.requests import Request
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from linescout_api.errors import json_error, resolve_request_id


def _multipart_boundary(content_type: str | None) -> bytes | None:
    """Extract the boundary parameter from a ``multipart/form-data`` content type.

    Self-contained so the middleware does not depend on python-multipart's
    import name (it changed across releases). RFC 2046 restricts boundary
    characters to ASCII, so the latin-1 round-trip cannot fail or mangle it.
    """
    if not content_type:
        return None
    segments = content_type.split(";")
    if segments[0].strip().lower() != "multipart/form-data":
        return None
    for segment in segments[1:]:
        name, _, value = segment.partition("=")
        if name.strip().lower() != "boundary":
            continue
        boundary = value.strip()
        if len(boundary) >= 2 and boundary.startswith('"') and boundary.endswith('"'):
            boundary = boundary[1:-1]
        return boundary.encode("latin-1") or None
    return None


class RequestBodyLimitMiddleware:
    """Pure-ASGI body cap enforced on the receive channel before any parsing."""

    def __init__(self, app: ASGIApp, *, max_body_bytes: int, max_multipart_parts: int = 16) -> None:
        if max_body_bytes <= 0:
            msg = "max_body_bytes must be a positive byte count"
            raise ValueError(msg)
        self.app = app
        self.max_body_bytes = max_body_bytes
        self.max_multipart_parts = max_multipart_parts

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        declared = self._declared_content_length(headers)
        if isinstance(declared, int) and declared < 0:
            await self._reject(
                scope, send, 400, "invalid_content_length", "Content-Length must not be negative"
            )
            return
        if isinstance(declared, str):
            await self._reject(
                scope,
                send,
                400,
                "invalid_content_length",
                "Content-Length must be a single non-negative integer",
            )
            return
        if declared is not None and declared > self.max_body_bytes:
            # Early rejection only; the byte counter below is authoritative.
            await self._reject(
                scope,
                send,
                413,
                "request_too_large",
                f"request body exceeds the {self.max_body_bytes}-byte limit",
            )
            return

        boundary = _multipart_boundary(headers.get("content-type"))

        # Drain the receive channel once, counting actual bytes. The running
        # total — never the header — decides when to cut the request off.
        body = bytearray()
        more_body = True
        while more_body:
            message = await receive()
            if message["type"] == "http.disconnect":
                return  # client went away; nothing to answer
            body.extend(message.get("body", b""))
            more_body = bool(message.get("more_body", False))
            if len(body) > self.max_body_bytes:
                await self._reject(
                    scope,
                    send,
                    413,
                    "request_too_large",
                    f"request body exceeds the {self.max_body_bytes}-byte limit",
                )
                return

        if boundary is not None and self._part_count(bytes(body), boundary) > (
            self.max_multipart_parts
        ):
            await self._reject(
                scope,
                send,
                413,
                "too_many_parts",
                f"multipart body exceeds the {self.max_multipart_parts}-part limit",
            )
            return

        replay_body = bytes(body)
        replayed = False

        async def replay_receive() -> Message:
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": replay_body, "more_body": False}
            # The body is complete; only a disconnect can still arrive.
            return await receive()

        await self.app(scope, replay_receive, send)

    @staticmethod
    def _part_count(body: bytes, boundary: bytes) -> int:
        """Count multipart parts by delimiter occurrences, parser semantics.

        Every part opens with ``--boundary`` and the body closes with
        ``--boundary--``, so a well-formed body yields ``parts + 1``
        occurrences. Truncated bodies simply undercount by one and fail in
        the parser later, which never loosens the bound.
        """
        occurrences = body.count(b"--" + boundary)
        return max(0, occurrences - 1)

    @staticmethod
    def _declared_content_length(headers: Headers) -> int | str | None:
        """Return the declared length, ``None`` if absent, or a string on error."""
        values = headers.getlist("content-length")
        if not values:
            return None
        if len(set(values)) > 1:
            return "conflicting"  # request-smuggling shape; servers catch this too
        try:
            return int(values[0])
        except ValueError:
            return "malformed"

    async def _reject(self, scope: Scope, send: Send, status: int, code: str, message: str) -> None:
        response = json_error(
            status,
            code,
            message,
            request_id=resolve_request_id(Request(scope)),
            retryable=False,
            # The unread remainder of the body is dropped; close rather than
            # leave the connection half-consumed.
            headers={"Connection": "close"},
        )
        await response(scope, receive=_dead_receive, send=send)


async def _dead_receive() -> Message:
    return {"type": "http.disconnect"}
