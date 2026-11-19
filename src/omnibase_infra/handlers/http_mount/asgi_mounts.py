# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""aiohttp-to-ASGI bridge for contract-declared HTTP mounts (OMN-21288).

``AsgiMounts`` builds each declared ASGI application from its factory,
runs its lifespan, and exposes an aiohttp middleware that hands every request
under a mount's prefix to that application. Everything else falls through to the
runtime's own routes, so the runtime's port and its own endpoints do not change.

All or nothing: ``prepare`` raises at once for a declaration, prefix or factory
that cannot be served, and a lifespan that does not start takes the whole server
out of service (every request answers 503) instead of leaving it half mounted.

Nothing here knows a prefix, a factory or an application: all three arrive as
``ModelHttpMount`` values read from a contract.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import logging
from collections.abc import (
    AsyncIterator,
    Awaitable,
    Callable,
    Iterable,
    MutableMapping,
)
from typing import Any, cast

from aiohttp import hdrs, web

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import ModelInfraErrorContext, RuntimeHostError
from omnibase_infra.handlers.http_mount.bounded_body import (
    BodyTooLargeError,
    BoundedInflater,
    MalformedBodyError,
    UnsupportedCodingError,
    content_coding,
    decoded_chunks,
    replay,
)
from omnibase_infra.models.model_http_mount import ModelHttpMount
from omnibase_infra.runtime.auto_wiring.discovery import read_contract_yaml
from omnibase_infra.runtime.auto_wiring.models.model_discovered_contract import (
    ModelDiscoveredContract,
)

logger = logging.getLogger(__name__)

_AsgiScope = MutableMapping[str, Any]
_AsgiMessage = MutableMapping[str, Any]
_AsgiReceive = Callable[[], Awaitable[_AsgiMessage]]
_AsgiSend = Callable[[_AsgiMessage], Awaitable[None]]
_AsgiApp = Callable[[_AsgiScope, _AsgiReceive, _AsgiSend], Awaitable[None]]

_SKIPPED_RESPONSE_HEADERS = frozenset({"transfer-encoding", "connection"})


def listener_runner(
    app: web.Application, *, shutdown_timeout: float | None = None
) -> web.AppRunner:
    """The runner for an application that carries mounts (OMN-26085).

    ``auto_decompress=False`` leaves a request body as it came off the socket, so
    the bridge decodes it under the mount's cap; aiohttp's parser would expand up
    to a read buffer of it ahead of any check. ``lingering_time=0`` stops aiohttp
    from reading and discarding the rest of a refused body before it closes.
    """
    if shutdown_timeout is None:
        return web.AppRunner(app, auto_decompress=False, lingering_time=0)
    return web.AppRunner(
        app,
        auto_decompress=False,
        lingering_time=0,
        shutdown_timeout=shutdown_timeout,
    )


def _refuse(status: type[web.HTTPClientError], reason: str) -> web.HTTPClientError:
    error = status(text=reason)
    error.force_close()
    return error


class LifespanDriver:
    """Drives one application's ASGI lifespan protocol."""

    def __init__(self, app: _AsgiApp) -> None:
        self._app = app
        self._inbox: asyncio.Queue[_AsgiMessage] = asyncio.Queue()
        self._outbox: asyncio.Queue[_AsgiMessage] = asyncio.Queue()
        self._task: asyncio.Task[None] | None = None

    async def _run(self) -> None:
        scope: _AsgiScope = {
            "type": "lifespan",
            "asgi": {"version": "3.0", "spec_version": "2.0"},
            "state": {},
        }
        await self._app(scope, self._inbox.get, self._outbox.put)

    async def _next(self, expected: str) -> _AsgiMessage:
        assert self._task is not None
        reply = asyncio.ensure_future(self._outbox.get())
        await asyncio.wait({reply, self._task}, return_when=asyncio.FIRST_COMPLETED)
        if reply.done():
            return reply.result()
        reply.cancel()
        # The task ended without answering: surface its own error, or say so.
        self._task.result()
        raise RuntimeError(f"application ended its lifespan before {expected}")

    async def startup(self) -> None:
        """Send lifespan.startup and require lifespan.startup.complete."""
        self._task = asyncio.create_task(self._run(), name="asgi-mount-lifespan")
        await self._inbox.put({"type": "lifespan.startup"})
        reply = await self._next("lifespan.startup.complete")
        if reply["type"] != "lifespan.startup.complete":
            detail = reply.get("message", "")
            raise RuntimeError(f"lifespan startup failed: {detail}")

    async def shutdown(self) -> None:
        """Send lifespan.shutdown and wait for the application to finish."""
        if self._task is None:
            return
        if not self._task.done():
            await self._inbox.put({"type": "lifespan.shutdown"})
            with contextlib.suppress(Exception):
                await self._next("lifespan.shutdown.complete")
        with contextlib.suppress(Exception):
            await self._task
        self._task = None


class AsgiMounts:
    """The set of ASGI applications a runtime serves for its contracts."""

    def __init__(self) -> None:
        self._mounts: list[tuple[ModelHttpMount, _AsgiApp, LifespanDriver]] = []
        self._started = False
        self._attached = False
        self.failure = ""
        """Why the mounts failed to start; empty while none has failed."""

    @staticmethod
    def _failure(mount: ModelHttpMount, reason: str) -> RuntimeHostError:
        context = ModelInfraErrorContext.with_correlation(
            transport_type=EnumInfraTransportType.HTTP,
            operation="mount_http_app",
            target_name=mount.path_prefix,
        )
        return RuntimeHostError(
            f"HTTP mount {mount.path_prefix} ({mount.factory}) failed: {reason}",
            context=context,
        )

    @staticmethod
    def _build(mount: ModelHttpMount) -> _AsgiApp:
        module_name, _, attribute = mount.factory.partition(":")
        try:
            factory = getattr(importlib.import_module(module_name), attribute)
            app = factory()
        except Exception as exc:
            raise AsgiMounts._failure(mount, f"factory raised {exc!r}") from exc
        if not callable(app):
            raise AsgiMounts._failure(mount, "factory did not return an ASGI app")
        return cast("_AsgiApp", app)

    @staticmethod
    def declared(
        contracts: Iterable[ModelDiscoveredContract],
    ) -> tuple[ModelHttpMount, ...]:
        """The ``http_mount`` blocks of the given contracts' ``contract.yaml`` files.

        Raises:
            ValueError: A contract declares a malformed ``http_mount`` block.
        """
        blocks = (
            ModelHttpMount.from_contract_block(
                cast("dict[str, object]", read_contract_yaml(c.contract_path)).get(
                    "http_mount"
                )
            )
            for c in contracts
        )
        return tuple(block for block in blocks if block is not None)

    def prepare(
        self, contracts: Iterable[ModelDiscoveredContract], router: web.UrlDispatcher
    ) -> bool:
        """Validate and build every declared mount; start nothing.

        Args:
            contracts: Contracts whose ``contract.yaml`` may declare a mount.
            router: The runtime's own router; a mount that equals or contains one
                of its routes is refused rather than shadowing it.

        Returns:
            True when at least one mount is declared and `start` must run.

        Raises:
            RuntimeHostError: A declaration is malformed, a prefix overlaps
                another mount or a runtime route, or a factory fails.
        """
        try:
            declared = self.declared(contracts)
        except ValueError as exc:
            raise RuntimeHostError(
                f"malformed http_mount declaration: {exc}",
                context=ModelInfraErrorContext.with_correlation(
                    transport_type=EnumInfraTransportType.HTTP,
                    operation="mount_http_app",
                ),
            ) from exc
        reserved = [resource.canonical for resource in router.resources()]
        for index, mount in enumerate(declared):
            for other in declared[index + 1 :]:
                if mount.covers(other.path_prefix) or other.covers(mount.path_prefix):
                    raise self._failure(mount, f"overlaps mount {other.path_prefix}")
            for path in reserved:
                if mount.covers(path):
                    raise self._failure(mount, f"would shadow runtime route {path}")
        self._mounts = [
            (m, app, LifespanDriver(app))
            for m, app in ((m, self._build(m)) for m in declared)
        ]
        return bool(declared)

    def attach(
        self, contracts: Iterable[ModelDiscoveredContract], router: web.UrlDispatcher
    ) -> asyncio.Task[None] | None:
        """`prepare` once, then start the lifespans in a background task.

        Returns:
            The start task; None when no mount is declared or when the mounts
            were already attached.
        """
        if self._attached:
            return None
        declared = self.prepare(contracts, router)
        self._attached = True
        return (
            asyncio.get_running_loop().create_task(self.start()) if declared else None
        )

    async def start(self) -> None:
        """Run every prepared mount's lifespan startup, or none.

        Raises:
            RuntimeHostError: A lifespan did not start. Mounts already started
                are shut down, `failure` says why, and the middleware answers
                503 for every request from then on.
        """
        for mount, _, lifespan in self._mounts:
            try:
                await lifespan.startup()
            except Exception as exc:
                error = self._failure(mount, f"lifespan startup: {exc}")
                self.failure = str(error)
                logger.critical("%s", error)
                await self.stop()
                raise error from exc
            logger.info(
                "HTTP mount started: %s -> %s", mount.path_prefix, mount.factory
            )
        self._started = True

    async def stop(self) -> None:
        """Shut down every started mount, last first."""
        self._started = False
        self._attached = False
        while self._mounts:
            _, _, lifespan = self._mounts.pop()
            await lifespan.shutdown()

    @web.middleware
    async def middleware(
        self,
        request: web.Request,
        handler: Callable[[web.Request], Awaitable[web.StreamResponse]],
    ) -> web.StreamResponse:
        """Serve the request from its mount, or pass it to the runtime's routes."""
        if self.failure:
            return web.json_response(
                {"status": "unavailable", "detail": self.failure}, status=503
            )
        for mount, app, _ in self._mounts:
            if mount.covers(request.path):
                if not self._started:
                    return web.json_response({"status": "starting"}, status=503)
                return await self._serve(app, request, mount.max_body_bytes)
        try:
            coding = content_coding(request.headers.get(hdrs.CONTENT_ENCODING))
        except UnsupportedCodingError as exc:
            raise _refuse(web.HTTPUnsupportedMediaType, str(exc)) from exc
        if coding is not None:
            # The listener does not decode bodies; only a mount's bridge does.
            raise _refuse(web.HTTPUnsupportedMediaType, f"content-encoding {coding}")
        return await handler(request)

    @staticmethod
    async def _serve(
        app: _AsgiApp, request: web.Request, max_body_bytes: int | None
    ) -> web.StreamResponse:
        def reject_body(limit: int, size: int) -> web.HTTPRequestEntityTooLarge:
            error = web.HTTPRequestEntityTooLarge(max_size=limit, actual_size=size)
            error.force_close()
            return error

        if max_body_bytes is not None:
            content_length = request.content_length
            if content_length is not None and content_length > max_body_bytes:
                raise reject_body(max_body_bytes, content_length)

        # A compressed body is decoded here, under the cap, before dispatch.
        decoded: AsyncIterator[bytes] | None = None
        try:
            coding = content_coding(request.headers.get(hdrs.CONTENT_ENCODING))
            if coding is not None:
                decoded = decoded_chunks(
                    request.content,
                    BoundedInflater(coding, max_body_bytes),
                    max_body_bytes,
                )
                if max_body_bytes is not None:
                    decoded = replay([chunk async for chunk in decoded])
        except UnsupportedCodingError as exc:
            raise _refuse(web.HTTPUnsupportedMediaType, str(exc)) from exc
        except BodyTooLargeError as exc:
            raise reject_body(exc.limit, exc.size) from exc
        except MalformedBodyError as exc:
            raise _refuse(web.HTTPBadRequest, str(exc)) from exc

        peer = (
            request.transport.get_extra_info("peername") if request.transport else None
        )
        scope: _AsgiScope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": f"{request.version.major}.{request.version.minor}",
            "method": request.method,
            "scheme": request.scheme,
            "path": request.path,
            "raw_path": request.raw_path.split("?", 1)[0].encode(),
            "query_string": request.query_string.encode(),
            "root_path": "",
            "headers": [(k.lower(), v) for k, v in request.raw_headers],
            "client": (peer[0], peer[1]) if peer else None,
            "server": None,
            "state": {},
        }
        finished = asyncio.Event()
        body_done = False
        body_size = 0
        body_rejection: web.HTTPClientError | None = None
        response: web.StreamResponse | None = None

        async def receive() -> _AsgiMessage:
            nonlocal body_done, body_size, body_rejection
            if body_rejection is not None:
                raise body_rejection
            if decoded is not None and not body_done:
                try:
                    piece = await anext(decoded, None)
                except BodyTooLargeError as exc:
                    body_rejection = reject_body(exc.limit, exc.size)
                    raise body_rejection from exc
                except MalformedBodyError as exc:
                    body_rejection = _refuse(web.HTTPBadRequest, str(exc))
                    raise body_rejection from exc
                body_done = piece is None
                return {
                    "type": "http.request",
                    "body": piece or b"",
                    "more_body": not body_done,
                }
            if not body_done:
                if max_body_bytes is None:
                    chunk = await request.content.readany()
                else:
                    # Read at most the remaining allowance and one overflow byte.
                    chunk = await request.content.read(max_body_bytes - body_size + 1)
                    body_size += len(chunk)
                    if body_size > max_body_bytes:
                        body_rejection = reject_body(max_body_bytes, body_size)
                        raise body_rejection
                body_done = request.content.at_eof()
                return {
                    "type": "http.request",
                    "body": chunk,
                    "more_body": not body_done,
                }
            await finished.wait()
            return {"type": "http.disconnect"}

        async def send(message: _AsgiMessage) -> None:
            nonlocal response
            if body_rejection is not None:
                return
            if message["type"] == "http.response.start":
                response = web.StreamResponse(status=message["status"])
                for name, value in message.get("headers", ()):
                    if name.lower() not in _SKIPPED_RESPONSE_HEADERS:
                        response.headers.add(
                            name.decode("latin-1"), value.decode("latin-1")
                        )
            elif message["type"] == "http.response.body" and response is not None:
                if not response.prepared:
                    await response.prepare(request)
                await response.write(message.get("body", b""))
                if not message.get("more_body", False):
                    await response.write_eof()
                    finished.set()

        try:
            await app(scope, receive, send)
        except Exception:
            if body_rejection is None:
                logger.exception("mounted ASGI app raised for %s", request.path)
                if response is None or not response.prepared:
                    return web.Response(status=500, text="Internal Server Error")
                raise
        finally:
            finished.set()
        if body_rejection is not None:
            if response is not None and response.prepared:
                # An app streaming its response may already have sent the status.
                request.protocol.force_close()
                return response
            raise body_rejection
        if response is None:
            return web.Response(status=500, text="Internal Server Error")
        return response


__all__ = ["AsgiMounts", "listener_runner"]
