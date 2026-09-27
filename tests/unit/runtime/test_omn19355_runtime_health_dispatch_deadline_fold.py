# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19355: an abandoned dispatch reaches the runtime's own health at once.

The monitor dimension runs on its interval (300s by default). The process-level
fold is what makes ``/health`` say DEGRADED on the next probe after a dispatch
is abandoned, and UNHEALTHY (HTTP 503) the moment the orphan limit is reached,
which is what the container healthcheck and a Kubernetes liveness probe act on.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omnibase_infra.runtime.runtime_host_process import RuntimeHostProcess
from tests.helpers.runtime_helpers import make_runtime_config, seed_mock_handlers

pytestmark = pytest.mark.unit


def _bus_health(*, healthy: bool, degraded: bool) -> dict[str, object]:
    return {
        "healthy": healthy,
        "degraded": degraded,
        "started": True,
        "dispatch_deadline": {"orphaned_dispatches": 1 if degraded else 0},
    }


async def _runtime_health(bus_health: dict[str, object]) -> dict[str, object]:
    process = RuntimeHostProcess(config=make_runtime_config())
    seed_mock_handlers(process)
    process._is_running = True
    handler_result = MagicMock(healthy=True, handler_type="mock", details={})
    with (
        patch.object(
            process._event_bus, "health_check", AsyncMock(return_value=bus_health)
        ),
        patch.object(
            process._lifecycle_executor,
            "check_handler_health",
            AsyncMock(return_value=handler_result),
        ),
        patch.object(
            process, "_check_published_events_map", AsyncMock(return_value=None)
        ),
    ):
        return await process.health_check()


@pytest.mark.asyncio
async def test_a_degraded_bus_makes_the_runtime_degraded_not_healthy() -> None:
    health = await _runtime_health(_bus_health(healthy=True, degraded=True))
    assert health["event_bus_healthy"] is True
    assert health["degraded"] is True
    assert health["healthy"] is False


@pytest.mark.asyncio
async def test_a_bus_at_its_orphan_limit_makes_the_runtime_unhealthy() -> None:
    health = await _runtime_health(_bus_health(healthy=False, degraded=False))
    assert health["healthy"] is False
    assert health["degraded"] is False, (
        "at the limit the runtime is unhealthy, not merely degraded, so the "
        "HTTP status is 503"
    )


@pytest.mark.asyncio
async def test_a_healthy_bus_leaves_the_runtime_healthy() -> None:
    """Negative control: the fold changes nothing when nothing was abandoned."""
    health = await _runtime_health(_bus_health(healthy=True, degraded=False))
    assert health["healthy"] is True
    assert health["degraded"] is False
