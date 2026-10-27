# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Native runtime health after configuration prefetch (OMN-21423)."""

from pathlib import Path

import aiohttp
import pytest
import yaml

from omnibase_core.container import ModelONEXContainer
from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.runtime.handler_registry import RegistryProtocolBinding
from omnibase_infra.runtime.runtime_host_process import (
    RuntimeHostProcess,
    prefetch_runtime_config,
)
from omnibase_infra.services.health_checker import ServiceHealth

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


async def test_prefetch_before_construction_preserves_native_runtime_health(
    tmp_path: Path,
) -> None:
    """Bind real HTTP health before constructing and starting a real runtime."""
    (tmp_path / "handler_contract.yaml").write_text(
        yaml.safe_dump(
            {
                "handler_name": "http",
                "handler_class": "omnibase_infra.handlers.handler_http.HandlerHttpRest",
                "handler_type": "effect",
                "protocol_type": "http",
            }
        )
    )
    container = ModelONEXContainer()
    prefetch_status = await prefetch_runtime_config(
        container=container,
        service_slug="prefetch-health-proof",
        prefetch_policy="disabled",
    )
    assert prefetch_status == "skipped"

    health = ServiceHealth(container=container, port=0, version="test")
    await health.start()
    runtime = RuntimeHostProcess(
        container=container,
        event_bus=EventBusInmemory(),
        handler_registry=RegistryProtocolBinding(),
        contract_paths=[str(tmp_path)],
        config={
            "service_name": "prefetch-health-proof",
            "node_name": "prefetch-health-proof",
            "env": "local",
            "version": "v1",
            "handler_source": {"mode": "contract"},
        },
        config_prefetch_status=prefetch_status,
    )
    health.attach_runtime(runtime)
    try:
        await runtime.start()
        assert health._site is not None
        assert health._site._server is not None
        port = health._site._server.sockets[0].getsockname()[1]
        async with (
            aiohttp.ClientSession() as session,
            session.get(f"http://127.0.0.1:{port}/health") as response,
        ):
            body = await response.json()
            assert response.status == 200
            assert body["status"] == "healthy"
            assert body["details"]["is_running"] is True
            assert body["details"]["config_prefetch_status"] == "skipped"
        assert not runtime._failed_handlers
        assert set(runtime._handlers) == {"http"}
    finally:
        await health.stop()
        await runtime.stop()
