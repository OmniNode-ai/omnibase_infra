# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Integration proof for OMN-9741 effects health liveness during startup.

OMN-9741 fixes the health liveness contract: when the runtime is not yet
attached (subscription startup in progress), the /health endpoint must return
HTTP 200 with status=degraded rather than HTTP 503, so Docker/autoheal does
not recycle a live effects process during long Kafka subscription setup.

Ticket: OMN-9741
Integration Test Coverage gate: OMN-7005 (hard gate since 2026-04-13).
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import web

from omnibase_infra.services.service_health import ServiceHealth


@pytest.mark.integration
@pytest.mark.asyncio
async def test_health_endpoint_returns_200_degraded_when_runtime_not_attached() -> None:
    """Health endpoint must return HTTP 200 degraded when runtime is not yet attached.

    This is the liveness invariant for OMN-9741: Docker autoheal kills containers
    on HTTP 503. During Kafka subscription startup, runtime is None — the endpoint
    must return degraded (HTTP 200), not unhealthy (HTTP 503).
    """
    container = MagicMock()
    server = ServiceHealth(container=container)
    mock_request = MagicMock(spec=web.Request)

    response = await server._handle_health(mock_request)

    assert response.status == 200, (
        f"Expected HTTP 200 (degraded) when runtime not attached, got {response.status}. "
        "Docker autoheal would kill a container returning 503 during startup."
    )
    body = json.loads(response.text)
    assert body["status"] == "degraded", (
        f"Expected status=degraded when runtime not attached, got {body['status']}"
    )
    assert body["details"]["startup_phase"] == "runtime_pending"
    assert body["details"]["runtime_attached"] is False


@pytest.mark.integration
@pytest.mark.asyncio
async def test_health_endpoint_returns_200_healthy_after_attach_runtime() -> None:
    """Health endpoint returns 200 healthy once attach_runtime() is called and runtime is healthy."""
    mock_runtime = MagicMock()
    mock_runtime.health_check = AsyncMock(
        return_value={
            "healthy": True,
            "degraded": False,
            "startup_in_progress": False,
            "is_running": True,
            "event_bus_healthy": True,
            "handlers": {},
            "failed_handlers": {},
        }
    )
    container = MagicMock()
    server = ServiceHealth(container=container)
    server.attach_runtime(mock_runtime)
    mock_request = MagicMock(spec=web.Request)

    response = await server._handle_health(mock_request)

    assert response.status == 200
    body = json.loads(response.text)
    assert body["status"] == "healthy"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_health_endpoint_returns_200_degraded_during_runtime_startup() -> None:
    """Health endpoint returns degraded 200 when runtime reports startup_in_progress."""
    mock_runtime = MagicMock()
    mock_runtime.health_check = AsyncMock(
        return_value={
            "healthy": False,
            "degraded": True,
            "startup_in_progress": True,
            "is_running": False,
            "event_bus_healthy": True,
            "handlers": {},
            "failed_handlers": {},
        }
    )
    server = ServiceHealth(runtime=mock_runtime)
    mock_request = MagicMock(spec=web.Request)

    response = await server._handle_health(mock_request)

    assert response.status == 200
    body = json.loads(response.text)
    assert body["status"] == "degraded"
    assert body["details"]["startup_in_progress"] is True
