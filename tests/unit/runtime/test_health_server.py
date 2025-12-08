# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for the ONEX runtime health server.

Tests the HTTP health server including:
- Server lifecycle (start/stop)
- Health endpoint responses
- Error handling
- Port configuration
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web

from omnibase_infra.errors import RuntimeHostError
from omnibase_infra.runtime.health_server import (
    DEFAULT_HTTP_HOST,
    DEFAULT_HTTP_PORT,
    HealthServer,
)


class TestHealthServerInit:
    """Tests for HealthServer initialization."""

    def test_init_with_defaults(self) -> None:
        """Test initialization with default values."""
        mock_runtime = MagicMock()
        server = HealthServer(runtime=mock_runtime)

        assert server._runtime is mock_runtime
        assert server._port == DEFAULT_HTTP_PORT
        assert server._host == DEFAULT_HTTP_HOST
        assert server._version == "unknown"
        assert not server.is_running

    def test_init_with_custom_values(self) -> None:
        """Test initialization with custom values."""
        mock_runtime = MagicMock()
        server = HealthServer(
            runtime=mock_runtime,
            port=9000,
            host="127.0.0.1",
            version="1.2.3",
        )

        assert server._port == 9000
        assert server._host == "127.0.0.1"
        assert server._version == "1.2.3"

    def test_port_property(self) -> None:
        """Test port property returns configured port."""
        mock_runtime = MagicMock()
        server = HealthServer(runtime=mock_runtime, port=9999)

        assert server.port == 9999


class TestHealthServerLifecycle:
    """Tests for HealthServer start/stop lifecycle."""

    async def test_start_creates_app_and_runner(self) -> None:
        """Test that start() creates aiohttp app and runner."""
        mock_runtime = MagicMock()
        server = HealthServer(runtime=mock_runtime, port=0)  # Port 0 for auto-assign

        # Patch aiohttp components
        with patch("omnibase_infra.runtime.health_server.web.Application") as mock_app:
            with patch(
                "omnibase_infra.runtime.health_server.web.AppRunner"
            ) as mock_runner:
                with patch(
                    "omnibase_infra.runtime.health_server.web.TCPSite"
                ) as mock_site:
                    mock_app_instance = MagicMock()
                    mock_app_instance.router = MagicMock()
                    mock_app.return_value = mock_app_instance

                    mock_runner_instance = MagicMock()
                    mock_runner_instance.setup = AsyncMock()
                    mock_runner.return_value = mock_runner_instance

                    mock_site_instance = MagicMock()
                    mock_site_instance.start = AsyncMock()
                    mock_site.return_value = mock_site_instance

                    await server.start()

                    assert server.is_running
                    mock_app.assert_called_once()
                    mock_runner.assert_called_once()
                    mock_site.assert_called_once()

                    # Cleanup
                    mock_site_instance.stop = AsyncMock()
                    mock_runner_instance.cleanup = AsyncMock()
                    await server.stop()

    async def test_start_idempotent(self) -> None:
        """Test that calling start() twice is safe."""
        mock_runtime = MagicMock()
        server = HealthServer(runtime=mock_runtime)

        with patch("omnibase_infra.runtime.health_server.web.Application") as mock_app:
            with patch(
                "omnibase_infra.runtime.health_server.web.AppRunner"
            ) as mock_runner:
                with patch(
                    "omnibase_infra.runtime.health_server.web.TCPSite"
                ) as mock_site:
                    mock_app_instance = MagicMock()
                    mock_app_instance.router = MagicMock()
                    mock_app.return_value = mock_app_instance

                    mock_runner_instance = MagicMock()
                    mock_runner_instance.setup = AsyncMock()
                    mock_runner.return_value = mock_runner_instance

                    mock_site_instance = MagicMock()
                    mock_site_instance.start = AsyncMock()
                    mock_site.return_value = mock_site_instance

                    await server.start()
                    await server.start()  # Second call should be no-op

                    # Only called once
                    assert mock_app.call_count == 1

                    # Cleanup
                    mock_site_instance.stop = AsyncMock()
                    mock_runner_instance.cleanup = AsyncMock()
                    await server.stop()

    async def test_stop_idempotent(self) -> None:
        """Test that calling stop() twice is safe."""
        mock_runtime = MagicMock()
        server = HealthServer(runtime=mock_runtime)

        # Server not started - stop should be no-op
        await server.stop()
        await server.stop()

        assert not server.is_running

    async def test_start_raises_on_port_binding_error(self) -> None:
        """Test that port binding errors raise RuntimeHostError."""
        mock_runtime = MagicMock()
        server = HealthServer(runtime=mock_runtime, port=8085)

        with patch("omnibase_infra.runtime.health_server.web.Application") as mock_app:
            with patch(
                "omnibase_infra.runtime.health_server.web.AppRunner"
            ) as mock_runner:
                with patch(
                    "omnibase_infra.runtime.health_server.web.TCPSite"
                ) as mock_site:
                    mock_app_instance = MagicMock()
                    mock_app_instance.router = MagicMock()
                    mock_app.return_value = mock_app_instance

                    mock_runner_instance = MagicMock()
                    mock_runner_instance.setup = AsyncMock()
                    mock_runner.return_value = mock_runner_instance

                    mock_site_instance = MagicMock()
                    mock_site_instance.start = AsyncMock(
                        side_effect=OSError("Address already in use")
                    )
                    mock_site.return_value = mock_site_instance

                    with pytest.raises(RuntimeHostError) as exc_info:
                        await server.start()

                    assert "Address already in use" in str(exc_info.value)


class TestHealthServerEndpoints:
    """Tests for health endpoint responses."""

    async def test_health_endpoint_healthy(self) -> None:
        """Test /health returns 200 when runtime is healthy."""
        mock_runtime = MagicMock()
        mock_runtime.health_check = AsyncMock(
            return_value={
                "healthy": True,
                "degraded": False,
                "is_running": True,
            }
        )

        server = HealthServer(runtime=mock_runtime, version="1.0.0")

        # Create a mock request
        mock_request = MagicMock(spec=web.Request)

        response = await server._handle_health(mock_request)

        assert response.status == 200
        assert response.content_type == "application/json"
        response_text = response.text
        assert response_text is not None
        assert '"status": "healthy"' in response_text
        assert '"version": "1.0.0"' in response_text

    async def test_health_endpoint_degraded(self) -> None:
        """Test /health returns 200 when runtime is degraded.

        Degraded means core is running but some handlers failed.
        Returns 200 so Docker/K8s considers container healthy.
        """
        mock_runtime = MagicMock()
        mock_runtime.health_check = AsyncMock(
            return_value={
                "healthy": False,
                "degraded": True,
                "is_running": True,
            }
        )

        server = HealthServer(runtime=mock_runtime, version="1.0.0")
        mock_request = MagicMock(spec=web.Request)

        response = await server._handle_health(mock_request)

        assert response.status == 200
        response_text = response.text
        assert response_text is not None
        assert '"status": "degraded"' in response_text

    async def test_health_endpoint_unhealthy(self) -> None:
        """Test /health returns 503 when runtime is unhealthy."""
        mock_runtime = MagicMock()
        mock_runtime.health_check = AsyncMock(
            return_value={
                "healthy": False,
                "degraded": False,
                "is_running": False,
            }
        )

        server = HealthServer(runtime=mock_runtime, version="1.0.0")
        mock_request = MagicMock(spec=web.Request)

        response = await server._handle_health(mock_request)

        assert response.status == 503
        response_text = response.text
        assert response_text is not None
        assert '"status": "unhealthy"' in response_text

    async def test_health_endpoint_exception(self) -> None:
        """Test /health returns 503 on exception."""
        mock_runtime = MagicMock()
        mock_runtime.health_check = AsyncMock(
            side_effect=Exception("Health check failed")
        )

        server = HealthServer(runtime=mock_runtime, version="1.0.0")
        mock_request = MagicMock(spec=web.Request)

        response = await server._handle_health(mock_request)

        assert response.status == 503
        response_text = response.text
        assert response_text is not None
        assert '"status": "unhealthy"' in response_text
        assert "Health check failed" in response_text


class TestHealthServerIntegration:
    """Integration tests for HealthServer with real HTTP requests."""

    async def test_real_health_endpoint(self) -> None:
        """Test health endpoint with real HTTP server."""
        mock_runtime = MagicMock()
        mock_runtime.health_check = AsyncMock(
            return_value={
                "healthy": True,
                "degraded": False,
                "is_running": True,
                "event_bus_healthy": True,
                "registered_handlers": ["http", "database"],
            }
        )

        # Use port 0 for automatic port assignment to avoid conflicts
        server = HealthServer(
            runtime=mock_runtime,
            port=0,
            version="test-1.0.0",
        )

        try:
            await server.start()
            assert server.is_running

            # Get actual port after binding - use type assertions for mypy
            site = server._site
            assert site is not None
            internal_server = site._server
            assert internal_server is not None
            sockets = getattr(internal_server, "sockets", None)
            assert sockets is not None and len(sockets) > 0
            actual_port: int = sockets[0].getsockname()[1]

            # Make real HTTP request using aiohttp
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"http://127.0.0.1:{actual_port}/health"
                ) as response:
                    assert response.status == 200
                    data = await response.json()
                    assert data["status"] == "healthy"
                    assert data["version"] == "test-1.0.0"
                    assert data["details"]["healthy"] is True

                # Test /ready endpoint (alias)
                async with session.get(
                    f"http://127.0.0.1:{actual_port}/ready"
                ) as response:
                    assert response.status == 200

        finally:
            await server.stop()
            assert not server.is_running


class TestHealthServerConstants:
    """Tests for health server constants."""

    def test_default_port(self) -> None:
        """Test default HTTP port value."""
        assert DEFAULT_HTTP_PORT == 8085

    def test_default_host(self) -> None:
        """Test default HTTP host value (0.0.0.0 required for container networking)."""
        # S104: Binding to all interfaces is intentional for Docker/K8s health checks
        assert DEFAULT_HTTP_HOST == "0.0.0.0"  # noqa: S104
