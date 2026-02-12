# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Integration tests for runtime startup readiness (OMN-2081).

Tests that the ONEX runtime reaches a ready state in under 10 seconds
using the in-memory event bus. These tests verify:

1. Runtime reaches healthy/ready state within the 10-second SLA
2. Health HTTP endpoint returns 200 when runtime is ready
3. Readiness check includes event bus health details

Related:
    - OMN-2081: Investor demo - runtime contract routing verification
    - src/omnibase_infra/runtime/service_runtime_host_process.py
    - src/omnibase_infra/services/service_health.py
"""

from __future__ import annotations

import time
from unittest.mock import patch

import aiohttp
import pytest

from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.runtime.service_runtime_host_process import RuntimeHostProcess
from omnibase_infra.services.service_health import ServiceHealth
from tests.helpers.aiohttp_utils import get_aiohttp_bound_port
from tests.helpers.runtime_helpers import make_runtime_config, seed_mock_handlers

pytestmark = pytest.mark.integration


# SLA: Runtime must reach ready state within this many seconds.
READY_STATE_SLA_SECONDS = 10.0

# Test config for RuntimeHostProcess
_STARTUP_TEST_CONFIG = make_runtime_config(
    service_name="startup-readiness-test",
    node_name="test-node-readiness",
)


class TestRuntimeStartupReadiness:
    """Integration tests for runtime startup readiness."""

    @pytest.mark.asyncio
    async def test_runtime_reaches_ready_state_within_sla(self) -> None:
        """Start RuntimeHostProcess and verify healthy=True within SLA.

        Measures wall-clock time from process.start() to health_check()
        returning healthy=True. Asserts the elapsed time is under the
        10-second SLA threshold.
        """
        event_bus = EventBusInmemory()
        runtime = RuntimeHostProcess(event_bus=event_bus, config=_STARTUP_TEST_CONFIG)

        async def noop_populate() -> None:
            pass

        with patch.object(runtime, "_populate_handlers_from_registry", noop_populate):
            seed_mock_handlers(runtime)

            t_start = time.monotonic()
            await runtime.start()
            health = await runtime.health_check()
            t_elapsed = time.monotonic() - t_start

            try:
                # Verify the runtime reached healthy state
                assert health["healthy"] is True, (
                    f"Runtime did not reach healthy state: {health}"
                )
                assert health["is_running"] is True

                # Verify the startup met the SLA
                assert t_elapsed < READY_STATE_SLA_SECONDS, (
                    f"Runtime startup took {t_elapsed:.2f}s, exceeding "
                    f"{READY_STATE_SLA_SECONDS}s SLA"
                )
            finally:
                await runtime.stop()

    @pytest.mark.asyncio
    async def test_runtime_health_endpoint_returns_200_when_ready(self) -> None:
        """Verify the HTTP health endpoint returns 200 and ready=True.

        Starts RuntimeHostProcess with ServiceHealth on port 0
        (auto-assigned), then issues an HTTP GET to /health and verifies
        a 200 status with ``status: healthy`` in the JSON body.
        """
        event_bus = EventBusInmemory()
        runtime = RuntimeHostProcess(event_bus=event_bus, config=_STARTUP_TEST_CONFIG)
        health_server = ServiceHealth(
            runtime=runtime, port=0, version="test-readiness-1.0.0"
        )

        async def noop_populate() -> None:
            pass

        with patch.object(runtime, "_populate_handlers_from_registry", noop_populate):
            seed_mock_handlers(runtime)
            await runtime.start()
            await health_server.start()

            # Retrieve the auto-assigned port.
            actual_port = get_aiohttp_bound_port(health_server)

            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"http://127.0.0.1:{actual_port}/health"
                    ) as response:
                        assert response.status == 200, (
                            f"Expected 200, got {response.status}"
                        )
                        data = await response.json()
                        assert data["status"] == "healthy"
                        assert data["details"]["is_running"] is True
                        assert data["details"]["healthy"] is True
            finally:
                await health_server.stop()
                await runtime.stop()

    @pytest.mark.asyncio
    async def test_runtime_ready_state_includes_event_bus_health(self) -> None:
        """Verify readiness_check includes event_bus details and reports ready.

        After starting the runtime, calls readiness_check() and verifies
        that the result contains event bus readiness information and the
        overall ready flag is True.
        """
        event_bus = EventBusInmemory()
        runtime = RuntimeHostProcess(event_bus=event_bus, config=_STARTUP_TEST_CONFIG)

        async def noop_populate() -> None:
            pass

        with patch.object(runtime, "_populate_handlers_from_registry", noop_populate):
            seed_mock_handlers(runtime)
            await runtime.start()

            try:
                readiness = await runtime.readiness_check()

                # Overall readiness should be True
                assert readiness["ready"] is True, f"Runtime not ready: {readiness}"
                assert readiness["is_running"] is True

                # Event bus readiness details should be present
                assert "event_bus_readiness" in readiness
                eb_readiness = readiness["event_bus_readiness"]
                assert isinstance(eb_readiness, dict)

                # Health check should also report event bus as healthy
                health = await runtime.health_check()
                assert health["event_bus_healthy"] is True
                assert isinstance(health["event_bus"], dict)
            finally:
                await runtime.stop()


__all__: list[str] = [
    "TestRuntimeStartupReadiness",
]
