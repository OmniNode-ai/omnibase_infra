# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""An idle initial HTTP connection must respect the server keepalive timeout."""

from __future__ import annotations

import asyncio

import pytest
from aiohttp import web


@pytest.mark.asyncio
async def test_initial_request_timeout_closes_idle_connection() -> None:
    """Refuse the unbounded idle connection from GHSA-5wr8-gcqf-gqf3."""
    runner = web.AppRunner(web.Application(), keepalive_timeout=0.05)
    await runner.setup()
    try:
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        host, port = runner.addresses[0]
        reader, writer = await asyncio.open_connection(host, port)
        try:
            assert await asyncio.wait_for(reader.read(1), timeout=1.0) == b""
        finally:
            writer.close()
            await writer.wait_closed()
    finally:
        await runner.cleanup()
