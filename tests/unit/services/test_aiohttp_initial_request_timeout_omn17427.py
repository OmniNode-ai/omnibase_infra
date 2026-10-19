# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Exercise the aiohttp timeout patch required by the runtime dependency floor."""

from __future__ import annotations

import asyncio
import socket

import pytest
from aiohttp import ClientSession, web

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.mark.parametrize("initial_bytes", [b"", b"GET / HTTP/1.1\r\nHost:"])
async def test_initial_request_timeout_closes_idle_or_incomplete_connections(
    initial_bytes: bytes,
) -> None:
    """An idle peer is closed while a complete health request still succeeds."""

    async def health(request: web.Request) -> web.Response:
        return web.Response(text="healthy")

    app = web.Application()
    app.router.add_get("/", health)
    runner = web.AppRunner(app, keepalive_timeout=0.05)
    await runner.setup()
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        site = web.SockSite(runner, listener)
        try:
            await site.start()
            async with ClientSession() as client:
                async with client.get(f"http://127.0.0.1:{port}/") as response:
                    assert response.status == 200
                    assert await response.text() == "healthy"
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            try:
                writer.write(initial_bytes)
                await writer.drain()
                assert await asyncio.wait_for(reader.read(1), timeout=2) == b""
            finally:
                writer.close()
                await writer.wait_closed()
        finally:
            await runner.cleanup()
