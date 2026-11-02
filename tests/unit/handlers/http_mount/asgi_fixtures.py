# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""ASGI application factories the OMN-21288 mount tests declare in a contract."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

LIFESPAN_LOG: list[str] = []


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
