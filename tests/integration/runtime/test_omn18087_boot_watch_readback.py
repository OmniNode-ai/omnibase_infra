# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Read the boot watch against a running in-memory runtime and its HTTP server."""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import cast

import httpx
import pytest

from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.handlers.boot_readiness_checks import watch_ready_bound
from omnibase_infra.handlers.handler_http import HandlerHttpRest
from omnibase_infra.protocols.protocol_container_aware import ProtocolContainerAware
from omnibase_infra.runtime.registry.registry_protocol_binding import (
    RegistryProtocolBinding,
)
from omnibase_infra.runtime.runtime_host_process import RuntimeHostProcess
from omnibase_infra.services.health_checker import ServiceHealth

pytestmark = pytest.mark.integration


async def test_boot_watch_preserves_503_and_records_late_runtime_readiness(
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    registry = RegistryProtocolBinding()
    registry.register("http", cast("type[ProtocolContainerAware]", HandlerHttpRest))
    process = RuntimeHostProcess(
        event_bus=EventBusInmemory(),
        handler_registry=registry,
        contract_paths=[str(tmp_path)],
        config={
            "service_name": "boot-watch-readback",
            "node_name": "boot-watch-readback",
            "env": "test",
            "version": "v1",
            "handler_source": {"mode": "contract"},
        },
    )
    gate_open = False

    def gate() -> tuple[bool, dict[str, object]]:
        return gate_open, {"summary": "lab boot dependency"}

    process.register_readiness_probe("lab_boot_dependency", gate)
    server = ServiceHealth(runtime=process, host="127.0.0.1", port=0)
    watch: asyncio.Task[bool] | None = None
    try:
        await asyncio.wait_for(process.start(), timeout=10)
        await server.start()
        assert server._runner is not None
        port = server._runner.addresses[0][1]
        url = f"http://127.0.0.1:{port}/ready"
        polled = asyncio.Event()

        async def probe() -> dict[str, object]:
            result = await process.readiness_check()
            polled.set()
            return result

        with caplog.at_level(logging.INFO):
            watch = asyncio.create_task(
                watch_ready_bound(
                    probe,
                    bound_seconds=1,
                    started_monotonic=time.monotonic() - 2,
                    poll_seconds=0.001,
                    observe_late_ready=True,
                )
            )
            await asyncio.wait_for(polled.wait(), timeout=1)
            async with httpx.AsyncClient() as client:
                blocked = await client.get(url)
                assert blocked.status_code == 503
                assert blocked.json()["details"]["ready"] is False
                assert "Boot bound exceeded" in caplog.text
                assert "lab_boot_dependency" in caplog.text
                assert not watch.done()

                gate_open = True
                ready = await client.get(url)
                assert ready.status_code == 200
                assert ready.json()["details"]["ready"] is True
                assert await asyncio.wait_for(watch, timeout=1) is False

            assert caplog.text.count("Boot bound exceeded") == 1
            assert "within_bound=False" in caplog.text
    finally:
        if watch is not None:
            watch.cancel()
            await asyncio.gather(watch, return_exceptions=True)
        await server.stop()
        await process.stop()
