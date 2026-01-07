# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for the ONEX runtime health service.

Tests the HTTP health service including:
- Service lifecycle (start/stop)
- Health endpoint responses
- Error handling
- Port configuration
- Container-based dependency injection (OMN-529)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web
from omnibase_core.container import ModelONEXContainer

from omnibase_infra.errors import ProtocolConfigurationError, RuntimeHostError
from omnibase_infra.services.service_health import (
    DEFAULT_HTTP_HOST,
    DEFAULT_HTTP_PORT,
    ServiceHealth,
)


class TestServiceHealthInit:
    """Tests for ServiceHealth initialization."""

    def test_init_with_defaults(self) -> None:
        """Test initialization with default values."""
        mock_runtime = MagicMock()
        server = ServiceHealth(runtime=mock_runtime)

        assert server._runtime is mock_runtime
        assert server._port == DEFAULT_HTTP_PORT
        assert server._host == DEFAULT_HTTP_HOST
        assert server._version == "unknown"
        assert not server.is_running

    def test_init_with_custom_values(self) -> None:
        """Test initialization with custom values."""
        mock_runtime = MagicMock()
        server = ServiceHealth(
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
        server = ServiceHealth(runtime=mock_runtime, port=9999)

        assert server.port == 9999


class TestServiceHealthLifecycle:
    """Tests for ServiceHealth start/stop lifecycle."""

    async def test_start_creates_app_and_runner(self) -> None:
        """Test that start() creates aiohttp app and runner."""
        mock_runtime = MagicMock()
        server = ServiceHealth(runtime=mock_runtime, port=0)  # Port 0 for auto-assign

        # Patch aiohttp components
        with patch(
            "omnibase_infra.services.service_health.web.Application"
        ) as mock_app:
            with patch(
                "omnibase_infra.services.service_health.web.AppRunner"
            ) as mock_runner:
                with patch(
                    "omnibase_infra.services.service_health.web.TCPSite"
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
        server = ServiceHealth(runtime=mock_runtime)

        with patch(
            "omnibase_infra.services.service_health.web.Application"
        ) as mock_app:
            with patch(
                "omnibase_infra.services.service_health.web.AppRunner"
            ) as mock_runner:
                with patch(
                    "omnibase_infra.services.service_health.web.TCPSite"
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
        server = ServiceHealth(runtime=mock_runtime)

        # Server not started - stop should be no-op
        await server.stop()
        await server.stop()

        assert not server.is_running

    async def test_start_raises_on_port_binding_error(self) -> None:
        """Test that port binding errors raise RuntimeHostError."""
        mock_runtime = MagicMock()
        server = ServiceHealth(runtime=mock_runtime, port=8085)

        with patch(
            "omnibase_infra.services.service_health.web.Application"
        ) as mock_app:
            with patch(
                "omnibase_infra.services.service_health.web.AppRunner"
            ) as mock_runner:
                with patch(
                    "omnibase_infra.services.service_health.web.TCPSite"
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


class TestServiceHealthEndpoints:
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

        server = ServiceHealth(runtime=mock_runtime, version="1.0.0")

        # Create a mock request
        mock_request = MagicMock(spec=web.Request)

        response = await server._handle_health(mock_request)

        assert response.status == 200
        assert response.content_type == "application/json"
        response_text = response.text
        assert response_text is not None
        # Pydantic model_dump_json() uses compact JSON format (no space after colon)
        assert '"status":"healthy"' in response_text
        assert '"version":"1.0.0"' in response_text

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

        server = ServiceHealth(runtime=mock_runtime, version="1.0.0")
        mock_request = MagicMock(spec=web.Request)

        response = await server._handle_health(mock_request)

        assert response.status == 200
        response_text = response.text
        assert response_text is not None
        # Pydantic model_dump_json() uses compact JSON format (no space after colon)
        assert '"status":"degraded"' in response_text

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

        server = ServiceHealth(runtime=mock_runtime, version="1.0.0")
        mock_request = MagicMock(spec=web.Request)

        response = await server._handle_health(mock_request)

        assert response.status == 503
        response_text = response.text
        assert response_text is not None
        # Pydantic model_dump_json() uses compact JSON format (no space after colon)
        assert '"status":"unhealthy"' in response_text

    async def test_health_endpoint_exception(self) -> None:
        """Test /health returns 503 on exception."""
        mock_runtime = MagicMock()
        mock_runtime.health_check = AsyncMock(
            side_effect=Exception("Health check failed")
        )

        server = ServiceHealth(runtime=mock_runtime, version="1.0.0")
        mock_request = MagicMock(spec=web.Request)

        response = await server._handle_health(mock_request)

        assert response.status == 503
        response_text = response.text
        assert response_text is not None
        # Pydantic model_dump_json() uses compact JSON format (no space after colon)
        assert '"status":"unhealthy"' in response_text
        assert "Health check failed" in response_text


class TestServiceHealthIntegration:
    """Integration tests for ServiceHealth with real HTTP requests."""

    async def test_real_health_endpoint(self) -> None:
        """Test health endpoint with real HTTP server."""
        mock_runtime = MagicMock()
        mock_runtime.health_check = AsyncMock(
            return_value={
                "healthy": True,
                "degraded": False,
                "is_running": True,
                "event_bus_healthy": True,
                "registered_handlers": ["http", "db"],
            }
        )

        # Use port 0 for automatic port assignment to avoid conflicts
        server = ServiceHealth(
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


class TestServiceHealthConstants:
    """Tests for health server constants."""

    def test_default_port(self) -> None:
        """Test default HTTP port value."""
        assert DEFAULT_HTTP_PORT == 8085

    def test_default_host(self) -> None:
        """Test default HTTP host value (0.0.0.0 required for container networking)."""
        # S104: Binding to all interfaces is intentional for Docker/K8s health checks
        assert DEFAULT_HTTP_HOST == "0.0.0.0"  # noqa: S104


class TestServiceHealthContainerInjection:
    """Tests for container-based dependency injection per OMN-529.

    These tests verify the ServiceHealth's support for ModelONEXContainer
    as the primary dependency injection mechanism, following the ONEX
    container injection pattern.
    """

    def test_container_property_returns_stored_container(self) -> None:
        """Container property should return the stored container."""
        mock_container = MagicMock(spec=ModelONEXContainer)
        mock_runtime = MagicMock()

        server = ServiceHealth(container=mock_container, runtime=mock_runtime)

        assert server.container is mock_container

    def test_container_property_returns_none_when_not_provided(self) -> None:
        """Container property should return None when only runtime provided."""
        mock_runtime = MagicMock()

        server = ServiceHealth(runtime=mock_runtime)

        assert server.container is None

    def test_raises_protocol_configuration_error_when_no_container_or_runtime(
        self,
    ) -> None:
        """Should raise ProtocolConfigurationError when neither container nor runtime provided.

        Per ONEX error conventions, missing required initialization parameters
        is a configuration error, not a generic ValueError.
        """
        with pytest.raises(ProtocolConfigurationError) as exc_info:
            ServiceHealth()

        assert "requires either 'container' or 'runtime'" in str(exc_info.value)

    def test_runtime_property_raises_when_not_available(self) -> None:
        """Should raise ProtocolConfigurationError when accessing runtime property without runtime.

        When ServiceHealth is initialized with only container (no runtime), accessing
        the runtime property should raise ProtocolConfigurationError since the runtime
        was never resolved from the container.
        """
        mock_container = MagicMock(spec=ModelONEXContainer)
        server = ServiceHealth(container=mock_container)

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            _ = server.runtime

        assert "RuntimeHostProcess not available" in str(exc_info.value)

    def test_instantiation_with_container_only(self) -> None:
        """Test that ServiceHealth can be instantiated with container parameter only.

        When container is provided without runtime, the server should still initialize.
        Runtime can be resolved from container or created lazily.
        """
        mock_container = MagicMock(spec=ModelONEXContainer)

        server = ServiceHealth(container=mock_container)

        assert server.container is mock_container
        assert server._port == DEFAULT_HTTP_PORT
        assert server._host == DEFAULT_HTTP_HOST

    def test_instantiation_with_container_and_custom_values(self) -> None:
        """Test container injection with custom port/host/version."""
        mock_container = MagicMock(spec=ModelONEXContainer)

        server = ServiceHealth(
            container=mock_container,
            port=9000,
            host="127.0.0.1",
            version="2.0.0",
        )

        assert server.container is mock_container
        assert server._port == 9000
        assert server._host == "127.0.0.1"
        assert server._version == "2.0.0"

    def test_instantiation_with_both_container_and_runtime(self) -> None:
        """Test that both container and runtime can be provided together.

        When both are provided, both should be stored for flexibility.
        """
        mock_container = MagicMock(spec=ModelONEXContainer)
        mock_runtime = MagicMock()

        server = ServiceHealth(container=mock_container, runtime=mock_runtime)

        assert server.container is mock_container
        assert server._runtime is mock_runtime

    @pytest.mark.asyncio
    async def test_create_from_container_factory(self) -> None:
        """Test the async factory method for container-based creation.

        The create_from_container() factory should create a fully configured
        ServiceHealth instance from a container.
        """
        mock_runtime = MagicMock()
        mock_container = MagicMock(spec=ModelONEXContainer)
        mock_container.service_registry = MagicMock()
        mock_container.service_registry.resolve_service = AsyncMock(
            return_value=mock_runtime
        )

        server = await ServiceHealth.create_from_container(
            container=mock_container,
            port=8090,
            version="factory-1.0.0",
        )

        assert server.container is mock_container
        assert server._port == 8090
        assert server._version == "factory-1.0.0"
        assert server._runtime is mock_runtime
        assert not server.is_running

    @pytest.mark.asyncio
    async def test_create_from_container_factory_with_defaults(self) -> None:
        """Test factory method with default values."""
        mock_runtime = MagicMock()
        mock_container = MagicMock(spec=ModelONEXContainer)
        mock_container.service_registry = MagicMock()
        mock_container.service_registry.resolve_service = AsyncMock(
            return_value=mock_runtime
        )

        server = await ServiceHealth.create_from_container(container=mock_container)

        assert server.container is mock_container
        assert server._port == DEFAULT_HTTP_PORT
        assert server._host == DEFAULT_HTTP_HOST
        assert server._version == "unknown"
        assert server._runtime is mock_runtime

    @pytest.mark.asyncio
    async def test_container_based_server_health_endpoint(self) -> None:
        """Test health endpoint works with container-based initialization."""
        mock_container = MagicMock(spec=ModelONEXContainer)
        mock_runtime = MagicMock()
        mock_runtime.health_check = AsyncMock(
            return_value={
                "healthy": True,
                "degraded": False,
                "is_running": True,
            }
        )

        server = ServiceHealth(
            container=mock_container,
            runtime=mock_runtime,
            version="container-1.0.0",
        )

        mock_request = MagicMock(spec=web.Request)
        response = await server._handle_health(mock_request)

        assert response.status == 200
        assert response.content_type == "application/json"
        response_text = response.text
        assert response_text is not None
        assert '"status":"healthy"' in response_text
        assert '"version":"container-1.0.0"' in response_text

    def test_container_storage_with_container_only_init(self) -> None:
        """Container should be properly stored and accessible with container-only init.

        When ServiceHealth is initialized with only a container (no runtime),
        the container should be stored and accessible via the property.
        """
        mock_container = MagicMock(spec=ModelONEXContainer)

        server = ServiceHealth(container=mock_container)

        # Verify container is stored
        assert server.container is mock_container
        # Verify runtime is None (not resolved yet)
        assert server._runtime is None
        # Verify other defaults are set correctly
        assert server._port == DEFAULT_HTTP_PORT
        assert server._host == DEFAULT_HTTP_HOST
        assert server._version == "unknown"
        assert not server.is_running

    @pytest.mark.asyncio
    async def test_create_from_container_factory_resolution_failure(self) -> None:
        """Factory method should propagate exception when container resolution fails.

        When container.service_registry.resolve_service() raises an exception,
        the create_from_container() factory should propagate that exception
        rather than silently failing.
        """
        mock_container = MagicMock(spec=ModelONEXContainer)
        mock_container.service_registry = MagicMock()
        mock_container.service_registry.resolve_service = AsyncMock(
            side_effect=Exception("Service resolution failed")
        )

        with pytest.raises(Exception) as exc_info:
            await ServiceHealth.create_from_container(container=mock_container)

        assert "Service resolution failed" in str(exc_info.value)

    def test_container_accessible_after_initialization_with_runtime(self) -> None:
        """Container should be accessible even when runtime is also provided.

        When both container and runtime are provided, the container should
        still be stored and accessible via the property.
        """
        mock_container = MagicMock(spec=ModelONEXContainer)
        mock_runtime = MagicMock()

        server = ServiceHealth(container=mock_container, runtime=mock_runtime)

        # Both should be accessible
        assert server.container is mock_container
        assert server._runtime is mock_runtime
        # runtime property should return the runtime (not raise)
        assert server.runtime is mock_runtime

    @pytest.mark.asyncio
    async def test_container_based_server_with_resolved_runtime(self) -> None:
        """ServiceHealth should work correctly with container-resolved runtime.

        This tests the full container injection flow where runtime is resolved
        from the container via the factory method.
        """
        mock_runtime = MagicMock()
        mock_runtime.health_check = AsyncMock(
            return_value={
                "healthy": True,
                "degraded": False,
                "is_running": True,
            }
        )

        mock_container = MagicMock(spec=ModelONEXContainer)
        mock_container.service_registry = MagicMock()
        mock_container.service_registry.resolve_service = AsyncMock(
            return_value=mock_runtime
        )

        # Use factory to create server
        server = await ServiceHealth.create_from_container(
            container=mock_container,
            version="resolved-1.0.0",
        )

        # Verify both container and runtime are accessible
        assert server.container is mock_container
        assert server.runtime is mock_runtime

        # Verify health endpoint works
        mock_request = MagicMock(spec=web.Request)
        response = await server._handle_health(mock_request)

        assert response.status == 200
        response_text = response.text
        assert response_text is not None
        assert '"status":"healthy"' in response_text
        assert '"version":"resolved-1.0.0"' in response_text
