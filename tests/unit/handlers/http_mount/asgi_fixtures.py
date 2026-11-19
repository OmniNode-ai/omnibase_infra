# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""ASGI application factories the OMN-21288 mount tests declare in a contract."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response

LIFESPAN_LOG: list[str] = []


def create_body_echo(chunks: list[bytes], first_chunk: asyncio.Event) -> FastAPI:
    """Record the chunks handed to an app that consumes the request stream."""
    app = FastAPI()

    @app.post("/body")
    async def echo(request: Request) -> Response:
        async for chunk in request.stream():
            chunks.append(chunk)
            first_chunk.set()
        return Response(b"".join(chunks), media_type="application/octet-stream")

    return app


def create_app() -> FastAPI:
    """A FastAPI app shaped like the intelligence API: absolute /api/v1 routes."""

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        LIFESPAN_LOG.append("startup")
        yield
        LIFESPAN_LOG.append("shutdown")

    app = FastAPI(lifespan=lifespan)

    @app.get("/api/v1/patterns")
    async def patterns(domain: str = "none") -> dict[str, str]:
        return {"domain": domain, "source": "mounted-app"}

    @app.post("/api/v1/echo")
    async def echo(payload: dict[str, str]) -> dict[str, str]:
        return payload

    return app


def create_app_planted_error() -> FastAPI:
    """Planted factory error: the factory itself raises."""
    raise RuntimeError("planted factory error")


def create_app_failing_lifespan() -> FastAPI:
    """Factory that returns an app whose lifespan startup raises (e.g. DB down)."""

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        raise ConnectionError("database unreachable")
        yield

    return FastAPI(lifespan=lifespan)


def create_not_an_app() -> object:
    """Factory that returns something that is not an ASGI application."""
    return "not callable"
