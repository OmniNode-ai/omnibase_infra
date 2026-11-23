# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The boot summary line is logged by a real HTTP ``/health`` answer (OMN-26561).

Unlike the unit tests, which call the health handler directly, this serves
``ServiceHealth`` on a real aiohttp server and reads the line a real ``GET /health``
causes, so the request path a boot's health probe takes is the one under test.
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import ClientSession, web

from omnibase_infra.models.node_announce.model_boot_clock import ModelBootClock
from omnibase_infra.services.health_checker import ServiceHealth

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_real_health_request_logs_one_boot_summary_line(
    caplog: pytest.LogCaptureFixture,
) -> None:
    ModelBootClock.reset()
    caplog.set_level(logging.INFO)
    runtime = MagicMock()
    runtime.health_check = AsyncMock(
        return_value={
            "healthy": True,
            "degraded": False,
            "is_running": True,
            "runtime_attached": True,
            "event_bus_healthy": True,
        }
    )
    health = ServiceHealth(runtime=runtime, version="1.0.0")
    app = web.Application()
    app.router.add_get("/health", health._handle_health)
    runner = web.AppRunner(app)
    await runner.setup()
    try:
        await web.TCPSite(runner, "127.0.0.1", 0).start()
        addresses = runner.addresses
        assert addresses
        port = addresses[0][1]
        async with ClientSession() as session:
            for _ in range(2):
                async with session.get(f"http://127.0.0.1:{port}/health") as resp:
                    assert resp.status == 200
    finally:
        await runner.cleanup()
        ModelBootClock.reset()

    lines = [
        r.getMessage()
        for r in caplog.records
        if "boot summary: parse_runs=" in r.getMessage()
    ]
    assert len(lines) == 1, lines
