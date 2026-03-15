# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for OMN-519: Health Check - Add degraded status detailed diagnostics.

Tests cover:
- ModelComponentHealth factory methods and serialization
- ModelDetailedHealthResponse construction
- /health/detailed endpoint responses for healthy/degraded/unhealthy states
- Component health breakdown in /health response
- Last successful health check timestamp tracking
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import web

from omnibase_infra.runtime.models.model_component_health import ModelComponentHealth
from omnibase_infra.runtime.models.model_detailed_health_response import (
    ModelDetailedHealthResponse,
)
from omnibase_infra.services.service_health import ServiceHealth

# ---------------------------------------------------------------------------
# ModelComponentHealth tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestModelComponentHealth:
    """Tests for ModelComponentHealth model."""

    def test_healthy_factory(self) -> None:
        """Test healthy factory creates correct model."""
        comp = ModelComponentHealth.healthy(
            "kafka", latency_ms=5.2, last_healthy="2025-12-08T10:00:00Z"
        )

        assert comp.name == "kafka"
        assert comp.status == "healthy"
        assert comp.latency_ms == 5.2
        assert comp.last_healthy == "2025-12-08T10:00:00Z"
        assert comp.error is None

    def test_degraded_factory(self) -> None:
        """Test degraded factory creates correct model with error."""
        comp = ModelComponentHealth.degraded(
            "consul",
            error="connection timeout",
            last_healthy="2025-12-08T09:55:00Z",
        )

        assert comp.name == "consul"
        assert comp.status == "degraded"
        assert comp.error == "connection timeout"
        assert comp.last_healthy == "2025-12-08T09:55:00Z"

    def test_unhealthy_factory(self) -> None:
        """Test unhealthy factory creates correct model."""
        comp = ModelComponentHealth.unhealthy(
            "postgres",
            error="connection refused",
        )

        assert comp.name == "postgres"
        assert comp.status == "unhealthy"
        assert comp.error == "connection refused"
        assert comp.last_healthy is None

    def test_serialization_excludes_none(self) -> None:
        """Test that model_dump excludes None fields when requested."""
        comp = ModelComponentHealth.healthy("kafka")
        dumped = comp.model_dump(exclude_none=True)

        assert "name" in dumped
        assert "status" in dumped
        assert "latency_ms" not in dumped
        assert "error" not in dumped

    def test_frozen_model(self) -> None:
        """Test that model is frozen (immutable)."""
        comp = ModelComponentHealth.healthy("kafka")

        with pytest.raises(Exception):
            comp.status = "unhealthy"  # type: ignore[misc]

    def test_healthy_with_details(self) -> None:
        """Test healthy factory with additional details dict."""
        details = {"lag": 100, "partitions": 3}
        comp = ModelComponentHealth.healthy("kafka", details=details)

        assert comp.details == {"lag": 100, "partitions": 3}

    def test_json_serialization(self) -> None:
        """Test JSON round-trip serialization."""
        comp = ModelComponentHealth.degraded(
            "consul", error="timeout", latency_ms=150.5
        )
        json_str = comp.model_dump_json(exclude_none=True)
        parsed = json.loads(json_str)

        assert parsed["name"] == "consul"
        assert parsed["status"] == "degraded"
        assert parsed["error"] == "timeout"
        assert parsed["latency_ms"] == 150.5


# ---------------------------------------------------------------------------
# ModelDetailedHealthResponse tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestModelDetailedHealthResponse:
    """Tests for ModelDetailedHealthResponse model."""

    def test_success_factory(self) -> None:
        """Test success factory with components."""
        components = {
            "kafka": ModelComponentHealth.healthy("kafka", latency_ms=5.0),
            "consul": ModelComponentHealth.degraded("consul", error="timeout"),
        }

        response = ModelDetailedHealthResponse.success(
            status="degraded",
            version="1.0.0",
            checked_at="2025-12-08T10:05:00Z",
            components=components,
            details={"healthy": False, "degraded": True},
        )

        assert response.status == "degraded"
        assert response.version == "1.0.0"
        assert response.checked_at == "2025-12-08T10:05:00Z"
        assert len(response.components) == 2
        assert response.components["kafka"].status == "healthy"
        assert response.components["consul"].status == "degraded"

    def test_failure_factory(self) -> None:
        """Test failure factory for error responses."""
        response = ModelDetailedHealthResponse.failure(
            version="1.0.0",
            error="Connection refused",
            error_type="ConnectionError",
            correlation_id="abc-123",
        )

        assert response.status == "unhealthy"
        assert response.error == "Connection refused"
        assert response.correlation_id == "abc-123"
        assert response.components is None

    def test_json_serialization_with_components(self) -> None:
        """Test full JSON serialization with nested components."""
        components = {
            "event_bus": ModelComponentHealth.healthy("event_bus"),
        }

        response = ModelDetailedHealthResponse.success(
            status="healthy",
            version="2.0.0",
            checked_at="2025-12-08T10:00:00Z",
            components=components,
            details={"healthy": True},
        )

        json_str = response.model_dump_json(exclude_none=True)
        parsed = json.loads(json_str)

        assert parsed["status"] == "healthy"
        assert "components" in parsed
        assert parsed["components"]["event_bus"]["status"] == "healthy"


# ---------------------------------------------------------------------------
# ServiceHealth /health/detailed endpoint tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestServiceHealthDetailedEndpoint:
    """Tests for /health/detailed endpoint handler."""

    @pytest.mark.asyncio
    async def test_detailed_healthy(self) -> None:
        """Test /health/detailed returns 200 with components when healthy."""
        mock_runtime = MagicMock()
        mock_runtime.health_check = AsyncMock(
            return_value={
                "healthy": True,
                "degraded": False,
                "is_running": True,
                "event_bus_healthy": True,
                "event_bus": {"healthy": True},
                "handlers": {
                    "http": {"healthy": True},
                    "db": {"healthy": True},
                },
                "failed_handlers": {},
                "registered_handlers": ["http", "db"],
            }
        )

        server = ServiceHealth(runtime=mock_runtime, version="1.0.0")
        mock_request = MagicMock(spec=web.Request)

        response = await server._handle_health_detailed(mock_request)

        assert response.status == 200
        assert response.content_type == "application/json"
        data = json.loads(response.text)

        assert data["status"] == "healthy"
        assert data["version"] == "1.0.0"
        assert "checked_at" in data
        assert "components" in data
        assert data["components"]["event_bus"]["status"] == "healthy"
        assert data["components"]["http"]["status"] == "healthy"
        assert data["components"]["db"]["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_detailed_degraded(self) -> None:
        """Test /health/detailed returns 200 with degraded component info."""
        mock_runtime = MagicMock()
        mock_runtime.health_check = AsyncMock(
            return_value={
                "healthy": False,
                "degraded": True,
                "is_running": True,
                "event_bus_healthy": True,
                "event_bus": {"healthy": True},
                "handlers": {"http": {"healthy": True}},
                "failed_handlers": {"consul": "connection timeout"},
                "registered_handlers": ["http"],
            }
        )

        server = ServiceHealth(runtime=mock_runtime, version="1.0.0")
        mock_request = MagicMock(spec=web.Request)

        response = await server._handle_health_detailed(mock_request)

        assert response.status == 200
        data = json.loads(response.text)

        assert data["status"] == "degraded"
        assert data["components"]["consul"]["status"] == "degraded"
        assert data["components"]["consul"]["error"] == "connection timeout"
        assert data["components"]["event_bus"]["status"] == "healthy"
        assert data["components"]["http"]["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_detailed_unhealthy(self) -> None:
        """Test /health/detailed returns 503 when unhealthy."""
        mock_runtime = MagicMock()
        mock_runtime.health_check = AsyncMock(
            return_value={
                "healthy": False,
                "degraded": False,
                "is_running": False,
                "event_bus_healthy": False,
                "event_bus": {"healthy": False, "error": "connection refused"},
                "handlers": {},
                "failed_handlers": {},
                "registered_handlers": [],
            }
        )

        server = ServiceHealth(runtime=mock_runtime, version="1.0.0")
        mock_request = MagicMock(spec=web.Request)

        response = await server._handle_health_detailed(mock_request)

        assert response.status == 503
        data = json.loads(response.text)

        assert data["status"] == "unhealthy"
        assert data["components"]["event_bus"]["status"] == "unhealthy"
        assert data["components"]["event_bus"]["error"] == "connection refused"

    @pytest.mark.asyncio
    async def test_detailed_exception_handling(self) -> None:
        """Test /health/detailed returns 503 on exception."""
        mock_runtime = MagicMock()
        mock_runtime.health_check = AsyncMock(
            side_effect=Exception("Detailed check failed")
        )

        server = ServiceHealth(runtime=mock_runtime, version="1.0.0")
        mock_request = MagicMock(spec=web.Request)

        response = await server._handle_health_detailed(mock_request)

        assert response.status == 503
        data = json.loads(response.text)
        assert data["status"] == "unhealthy"
        assert "Detailed check failed" in data["error"]

    @pytest.mark.asyncio
    async def test_detailed_includes_check_latency(self) -> None:
        """Test /health/detailed includes check_latency_ms in details."""
        mock_runtime = MagicMock()
        mock_runtime.health_check = AsyncMock(
            return_value={
                "healthy": True,
                "degraded": False,
                "is_running": True,
                "event_bus_healthy": True,
                "event_bus": {"healthy": True},
                "handlers": {},
                "failed_handlers": {},
                "registered_handlers": [],
            }
        )

        server = ServiceHealth(runtime=mock_runtime, version="1.0.0")
        mock_request = MagicMock(spec=web.Request)

        response = await server._handle_health_detailed(mock_request)
        data = json.loads(response.text)

        assert "check_latency_ms" in data["details"]
        assert isinstance(data["details"]["check_latency_ms"], float)


# ---------------------------------------------------------------------------
# ServiceHealth /health endpoint component enrichment tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestServiceHealthComponentEnrichment:
    """Tests for component health info added to /health response."""

    @pytest.mark.asyncio
    async def test_health_includes_components(self) -> None:
        """Test /health response includes components breakdown in details."""
        mock_runtime = MagicMock()
        mock_runtime.health_check = AsyncMock(
            return_value={
                "healthy": True,
                "degraded": False,
                "is_running": True,
                "event_bus_healthy": True,
                "event_bus": {"healthy": True},
                "handlers": {"http": {"healthy": True}},
                "failed_handlers": {},
                "registered_handlers": ["http"],
            }
        )

        server = ServiceHealth(runtime=mock_runtime, version="1.0.0")
        mock_request = MagicMock(spec=web.Request)

        response = await server._handle_health(mock_request)
        data = json.loads(response.text)

        assert "components" in data["details"]
        assert data["details"]["components"]["event_bus"]["status"] == "healthy"
        assert data["details"]["components"]["http"]["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_health_degraded_shows_failed_component(self) -> None:
        """Test /health response shows failed handler as degraded component."""
        mock_runtime = MagicMock()
        mock_runtime.health_check = AsyncMock(
            return_value={
                "healthy": False,
                "degraded": True,
                "is_running": True,
                "event_bus_healthy": True,
                "event_bus": {"healthy": True},
                "handlers": {"http": {"healthy": True}},
                "failed_handlers": {"vault": "auth token expired"},
                "registered_handlers": ["http"],
            }
        )

        server = ServiceHealth(runtime=mock_runtime, version="1.0.0")
        mock_request = MagicMock(spec=web.Request)

        response = await server._handle_health(mock_request)
        data = json.loads(response.text)

        assert data["status"] == "degraded"
        assert data["details"]["components"]["vault"]["status"] == "degraded"
        assert data["details"]["components"]["vault"]["error"] == "auth token expired"


# ---------------------------------------------------------------------------
# Timestamp tracking tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestServiceHealthTimestampTracking:
    """Tests for last successful health check timestamp tracking."""

    @pytest.mark.asyncio
    async def test_timestamps_updated_on_healthy_check(self) -> None:
        """Test that healthy component timestamps are recorded."""
        mock_runtime = MagicMock()
        mock_runtime.health_check = AsyncMock(
            return_value={
                "healthy": True,
                "degraded": False,
                "is_running": True,
                "event_bus_healthy": True,
                "event_bus": {"healthy": True},
                "handlers": {"http": {"healthy": True}},
                "failed_handlers": {},
                "registered_handlers": ["http"],
            }
        )

        server = ServiceHealth(runtime=mock_runtime, version="1.0.0")
        mock_request = MagicMock(spec=web.Request)

        # Initially empty
        assert len(server._last_healthy_timestamps) == 0

        await server._handle_health_detailed(mock_request)

        # After healthy check, timestamps should be recorded
        assert "event_bus" in server._last_healthy_timestamps
        assert "http" in server._last_healthy_timestamps

    @pytest.mark.asyncio
    async def test_timestamps_preserved_when_component_becomes_unhealthy(
        self,
    ) -> None:
        """Test that last_healthy timestamp is preserved when a component goes down."""
        mock_runtime = MagicMock()

        # First call: everything healthy
        mock_runtime.health_check = AsyncMock(
            return_value={
                "healthy": True,
                "degraded": False,
                "is_running": True,
                "event_bus_healthy": True,
                "event_bus": {"healthy": True},
                "handlers": {"http": {"healthy": True}},
                "failed_handlers": {},
                "registered_handlers": ["http"],
            }
        )

        server = ServiceHealth(runtime=mock_runtime, version="1.0.0")
        mock_request = MagicMock(spec=web.Request)

        await server._handle_health_detailed(mock_request)
        first_http_timestamp = server._last_healthy_timestamps["http"]

        # Second call: http is now unhealthy
        mock_runtime.health_check = AsyncMock(
            return_value={
                "healthy": False,
                "degraded": False,
                "is_running": True,
                "event_bus_healthy": True,
                "event_bus": {"healthy": True},
                "handlers": {"http": {"healthy": False, "error": "timeout"}},
                "failed_handlers": {},
                "registered_handlers": ["http"],
            }
        )

        response = await server._handle_health_detailed(mock_request)
        data = json.loads(response.text)

        # The http component should still have last_healthy from first call
        assert data["components"]["http"]["last_healthy"] == first_http_timestamp
        assert data["components"]["http"]["status"] == "unhealthy"


# ---------------------------------------------------------------------------
# Real HTTP integration test for /health/detailed
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestServiceHealthDetailedIntegration:
    """Integration test for /health/detailed with real HTTP server."""

    @pytest.mark.asyncio
    async def test_real_detailed_endpoint(self) -> None:
        """Test /health/detailed with real HTTP server."""
        mock_runtime = MagicMock()
        mock_runtime.health_check = AsyncMock(
            return_value={
                "healthy": True,
                "degraded": False,
                "is_running": True,
                "event_bus_healthy": True,
                "event_bus": {"healthy": True},
                "handlers": {"http": {"healthy": True}},
                "failed_handlers": {},
                "registered_handlers": ["http"],
            }
        )
        mock_runtime.readiness_check = AsyncMock(
            return_value={
                "ready": True,
                "is_running": True,
                "is_draining": False,
            }
        )

        server = ServiceHealth(
            runtime=mock_runtime,
            port=0,
            version="test-detailed-1.0.0",
        )

        try:
            await server.start()
            assert server.is_running

            # Get actual port
            site = server._site
            assert site is not None
            internal_server = site._server
            assert internal_server is not None
            sockets = getattr(internal_server, "sockets", None)
            assert sockets is not None and len(sockets) > 0
            actual_port: int = sockets[0].getsockname()[1]

            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"http://127.0.0.1:{actual_port}/health/detailed"
                ) as response:
                    assert response.status == 200
                    data = await response.json()
                    assert data["status"] == "healthy"
                    assert data["version"] == "test-detailed-1.0.0"
                    assert "components" in data
                    assert "checked_at" in data
                    assert data["components"]["event_bus"]["status"] == "healthy"

        finally:
            await server.stop()
            assert not server.is_running
