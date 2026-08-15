# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for the /health <- runtime health monitor join (OMN-15217).

The mask this ticket was filed for had two layers. The outer layer was the
container healthcheck (`curl -sf` reads only the status code). The inner layer,
proven by a live probe of the stability lane on 2026-07-27T12:58Z, was that the
payload itself lied: ``/health`` returned ``status=healthy, degraded=false``
while ``ServiceRuntimeHealthMonitor`` logged ``status=DEGRADED ... errors=4``
every five minutes. Parsing the response body — the fix the ticket's summary
proposed — would NOT have caught it, because the monitor's verdict was never
joined to the payload.

These tests pin the join.

Related Tickets:
    - OMN-15217: stability-lane runtime reports DEGRADED while Docker reads healthy
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from aiohttp import web

from omnibase_infra.models.health.model_runtime_health_check_event import (
    ModelRuntimeHealthCheckEvent,
)
from omnibase_infra.models.health.model_runtime_health_dimension import (
    ModelRuntimeHealthDimension,
)
from omnibase_infra.runtime.health.runtime_health_block import (
    RUNTIME_HEALTH_DETAIL_KEY,
)
from omnibase_infra.services.health_checker import ServiceHealth


def _live_runtime() -> MagicMock:
    """A runtime whose process-local health is perfect.

    This is exactly what RuntimeHostProcess.health_check() reported on the
    stability lane while four contracts were failing to load: the process knows
    nothing about contract discovery.
    """
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
    return runtime


def _verdict(status: str = "DEGRADED") -> ModelRuntimeHealthCheckEvent:
    return ModelRuntimeHealthCheckEvent(
        correlation_id=uuid4(),
        timestamp=datetime.now(UTC),
        status=status,  # type: ignore[arg-type]
        dimensions=(
            ModelRuntimeHealthDimension(
                name="discovery_errors",
                status="DEGRADED" if status != "HEALTHY" else "HEALTHY",
                detail="4 contract(s) failed to load: node_alpha, node_beta",
            ),
        ),
        contract_count=296,
        discovery_error_count=4,
    )


async def _get_health(server: ServiceHealth) -> tuple[int, dict]:
    response = await server._handle_health(MagicMock(spec=web.Request))
    assert response.text is not None
    return response.status, json.loads(response.text)


@pytest.mark.unit
class TestRuntimeVerdictJoin:
    @pytest.mark.asyncio
    async def test_without_a_provider_the_payload_cannot_see_degradation(self) -> None:
        """The pre-fix state, pinned: process-local health only.

        This documents the inner mask. A payload with no verdict reports the
        runtime as healthy no matter what the monitor found, which is why body
        parsing alone was never a sufficient fix.
        """
        server = ServiceHealth(runtime=_live_runtime(), version="1.0.0")

        status, body = await _get_health(server)

        assert status == 200
        assert body["status"] == "healthy"
        assert body["details"][RUNTIME_HEALTH_DETAIL_KEY] is None

    @pytest.mark.asyncio
    async def test_degraded_verdict_degrades_the_reported_status(self) -> None:
        """The fix: a DEGRADED monitor verdict is visible in the body."""
        server = ServiceHealth(runtime=_live_runtime(), version="1.0.0")
        server.set_runtime_health_provider(_verdict)

        status, body = await _get_health(server)

        assert body["status"] == "degraded"
        assert body["details"]["degraded"] is True

        block = body["details"][RUNTIME_HEALTH_DETAIL_KEY]
        assert block["status"] == "DEGRADED"
        assert block["discovery_error_count"] == 4
        assert block["dimensions"][0]["name"] == "discovery_errors"
        assert "node_alpha" in block["dimensions"][0]["detail"]

        # The HTTP status code is deliberately unchanged: /health is also the
        # liveness probe autoheal watches, and a restart-immune degradation must
        # not become a restart loop. Strictness lives in the container check.
        assert status == 200

    @pytest.mark.asyncio
    async def test_critical_verdict_reports_unhealthy_without_flipping_liveness(
        self,
    ) -> None:
        server = ServiceHealth(runtime=_live_runtime(), version="1.0.0")
        server.set_runtime_health_provider(lambda: _verdict("CRITICAL"))

        status, body = await _get_health(server)

        assert body["status"] == "unhealthy"
        assert body["details"][RUNTIME_HEALTH_DETAIL_KEY]["status"] == "CRITICAL"
        assert status == 200

    @pytest.mark.asyncio
    async def test_healthy_verdict_leaves_a_healthy_runtime_healthy(self) -> None:
        server = ServiceHealth(runtime=_live_runtime(), version="1.0.0")
        server.set_runtime_health_provider(lambda: _verdict("HEALTHY"))

        status, body = await _get_health(server)

        assert status == 200
        assert body["status"] == "healthy"
        assert body["details"][RUNTIME_HEALTH_DETAIL_KEY]["status"] == "HEALTHY"

    @pytest.mark.asyncio
    async def test_provider_returning_none_is_reported_as_an_absent_verdict(
        self,
    ) -> None:
        """Absent verdict is null in the payload — never silently omitted.

        Consumers must be able to tell "monitor has not reported yet" apart from
        "this build does not publish verdicts", which is why the key is always
        present.
        """
        server = ServiceHealth(runtime=_live_runtime(), version="1.0.0")
        server.set_runtime_health_provider(lambda: None)

        status, body = await _get_health(server)

        assert status == 200
        assert body["status"] == "healthy"
        assert RUNTIME_HEALTH_DETAIL_KEY in body["details"]
        assert body["details"][RUNTIME_HEALTH_DETAIL_KEY] is None

    @pytest.mark.asyncio
    async def test_a_raising_provider_never_breaks_the_health_probe(self) -> None:
        """A best-effort verdict source must not take down liveness."""

        def _boom() -> ModelRuntimeHealthCheckEvent | None:
            raise RuntimeError("monitor exploded")

        server = ServiceHealth(runtime=_live_runtime(), version="1.0.0")
        server.set_runtime_health_provider(_boom)

        status, body = await _get_health(server)

        assert status == 200
        assert body["details"][RUNTIME_HEALTH_DETAIL_KEY] is None

    @pytest.mark.asyncio
    async def test_runtime_pending_payload_declares_the_verdict_absent(self) -> None:
        """The pre-attach 503 path carries the key too, so parsers see one shape."""
        server = ServiceHealth(container=MagicMock(), version="1.0.0")

        status, body = await _get_health(server)

        assert status == 503
        assert body["details"][RUNTIME_HEALTH_DETAIL_KEY] is None

    @pytest.mark.asyncio
    async def test_provider_can_be_detached(self) -> None:
        server = ServiceHealth(runtime=_live_runtime(), version="1.0.0")
        server.set_runtime_health_provider(_verdict)
        server.set_runtime_health_provider(None)

        _, body = await _get_health(server)

        assert body["status"] == "healthy"
        assert body["details"][RUNTIME_HEALTH_DETAIL_KEY] is None
