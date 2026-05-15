# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Integration proof for runtime health liveness during startup.

OMN-9741 fixes the health liveness contract: when the runtime is not yet
attached, the /health endpoint can expose structured runtime_pending details
before RuntimeHostProcess exists.

OMN-11068 tightens the Docker probe contract: runtime_pending must return HTTP
503 so Docker health cannot report success while runtime_attached=false.

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
async def test_health_endpoint_returns_503_degraded_when_runtime_not_attached() -> None:
    """Health endpoint must fail Docker probes when runtime is not yet attached.

    This is the OMN-11068 masking regression: plain curl -sf Docker healthchecks
    must fail while runtime_attached=false and startup_phase=runtime_pending.
    """
    container = MagicMock()
    server = ServiceHealth(container=container)
    mock_request = MagicMock(spec=web.Request)

    response = await server._handle_health(mock_request)

    assert response.status == 503, (
        f"Expected HTTP 503 (degraded) when runtime not attached, got {response.status}. "
        "Docker healthchecks must not pass while runtime_attached=false."
    )
    body = json.loads(response.text)
    assert body["status"] == "degraded", (
        f"Expected status=degraded when runtime not attached, got {body['status']}"
    )
    assert body["details"]["is_running"] is False
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
async def test_health_endpoint_returns_503_degraded_until_runtime_is_running() -> None:
    """Health endpoint fails Docker probes while attached runtime is not running."""
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

    assert response.status == 503
    body = json.loads(response.text)
    assert body["status"] == "degraded"
    assert body["details"]["startup_in_progress"] is True
    assert body["details"]["is_running"] is False
