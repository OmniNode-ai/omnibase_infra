# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
# ruff: noqa: S104
# S104 disabled: Tests verify 0.0.0.0 binding for container networking
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


@pytest.mark.unit
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


@pytest.mark.unit
class TestServiceHealthLifecycle:
    """Tests for ServiceHealth start/stop lifecycle."""

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
    async def test_stop_idempotent(self) -> None:
        """Test that calling stop() twice is safe."""
        mock_runtime = MagicMock()
        server = ServiceHealth(runtime=mock_runtime)

        # Server not started - stop should be no-op
        await server.stop()
        await server.stop()

        assert not server.is_running

    @pytest.mark.asyncio
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


@pytest.mark.unit
class TestServiceHealthEndpoints:
    """Tests for health endpoint responses."""

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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


@pytest.mark.unit
class TestServiceHealthIntegration:
    """Integration tests for ServiceHealth with real HTTP requests."""

    @pytest.mark.asyncio
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


@pytest.mark.unit
class TestServiceHealthConstants:
    """Tests for health server constants."""

    def test_default_port(self) -> None:
        """Test default HTTP port value."""
        assert DEFAULT_HTTP_PORT == 8085

    def test_default_host(self) -> None:
        """Test default HTTP host value (0.0.0.0 required for container networking)."""
        # S104: Binding to all interfaces is intentional for Docker/K8s health checks
        assert DEFAULT_HTTP_HOST == "0.0.0.0"


@pytest.mark.unit
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

    def test_container_none_after_runtime_only_init(self) -> None:
        """Verify container property returns None when initialized with runtime only.

        This test ensures backward compatibility with the legacy runtime-only
        initialization pattern. When only runtime is provided, container should
        be None but runtime should work correctly.
        """
        mock_runtime = MagicMock()
        server = ServiceHealth(runtime=mock_runtime)

        # Container should be None when not provided
        assert server.container is None
        # But runtime should work
        assert server.runtime is mock_runtime
        # And server should be functional
        assert server._port == DEFAULT_HTTP_PORT
        assert not server.is_running

    @pytest.mark.asyncio
    async def test_create_from_container_wraps_resolution_exception(self) -> None:
        """Verify exception wrapping when container resolution fails.

        When the container's service_registry.resolve_service() raises an exception,
        the create_from_container() factory should wrap it in ProtocolConfigurationError
        with correlation_id context for proper error tracking and debugging.

        This verifies that:
        1. The original error message is preserved
        2. correlation_id is logged for tracing
        3. The exception is properly chained (from e)
        """
        mock_container = MagicMock(spec=ModelONEXContainer)
        mock_container.service_registry = MagicMock()
        mock_container.service_registry.resolve_service = AsyncMock(
            side_effect=KeyError("RuntimeHostProcess not registered")
        )

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            await ServiceHealth.create_from_container(container=mock_container)

        # Original error message should be included
        assert "RuntimeHostProcess not registered" in str(exc_info.value)
        # Should be wrapped in ProtocolConfigurationError
        assert "Failed to resolve RuntimeHostProcess from container" in str(
            exc_info.value
        )
        # Original exception should be chained
        assert exc_info.value.__cause__ is not None

    @pytest.mark.asyncio
    async def test_create_from_container_wraps_attribute_error(self) -> None:
        """Verify AttributeError wrapping when container lacks service_registry.

        This tests the edge case where a malformed container is provided that
        doesn't have the expected service_registry attribute or method.
        The exception should be wrapped in ProtocolConfigurationError with
        correlation_id for debugging.
        """
        mock_container = MagicMock(spec=ModelONEXContainer)
        # Simulate missing resolve_service method
        mock_container.service_registry = MagicMock()
        mock_container.service_registry.resolve_service = AsyncMock(
            side_effect=AttributeError(
                "'NoneType' object has no attribute 'resolve_service'"
            )
        )

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            await ServiceHealth.create_from_container(container=mock_container)

        # Original error message should be included
        assert "resolve_service" in str(exc_info.value)
        # Should indicate resolution failure
        assert "Failed to resolve RuntimeHostProcess from container" in str(
            exc_info.value
        )
        # Original exception should be chained
        assert exc_info.value.__cause__ is not None

    @pytest.mark.asyncio
    async def test_create_from_container_logs_correlation_id_on_failure(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Verify correlation_id is logged when container resolution fails.

        The create_from_container() factory method should log errors with
        correlation_id for distributed tracing and debugging support.
        """
        import logging

        mock_container = MagicMock(spec=ModelONEXContainer)
        mock_container.service_registry = MagicMock()
        mock_container.service_registry.resolve_service = AsyncMock(
            side_effect=Exception("Test resolution failure")
        )

        with caplog.at_level(logging.ERROR):
            with pytest.raises(ProtocolConfigurationError):
                await ServiceHealth.create_from_container(container=mock_container)

        # Verify correlation_id was logged
        assert any("correlation_id=" in record.message for record in caplog.records)
        # Verify error was logged with appropriate context
        assert any(
            "Failed to resolve RuntimeHostProcess from container" in record.message
            for record in caplog.records
        )

    def test_error_context_includes_transport_type_on_missing_deps(self) -> None:
        """Verify ProtocolConfigurationError includes proper error context.

        When neither container nor runtime is provided, the raised
        ProtocolConfigurationError should include ModelInfraErrorContext
        with the HTTP transport type for debugging infrastructure issues.
        """
        with pytest.raises(ProtocolConfigurationError) as exc_info:
            ServiceHealth()

        error = exc_info.value
        assert "requires either 'container' or 'runtime'" in str(error)
        # Verify the error has context (implementation detail but important for debugging)
        assert hasattr(error, "context")

    def test_container_only_init_does_not_create_app_components(self) -> None:
        """Verify container-only initialization doesn't prematurely create aiohttp components.

        When ServiceHealth is initialized with only a container (no runtime),
        it should not create any aiohttp Application, AppRunner, or TCPSite
        components until start() is called.
        """
        mock_container = MagicMock(spec=ModelONEXContainer)

        server = ServiceHealth(container=mock_container)

        # Verify no aiohttp components created during init
        assert server._app is None
        assert server._runner is None
        assert server._site is None
        assert not server.is_running

    def test_container_preserved_through_lifecycle(self) -> None:
        """Verify container reference is preserved throughout server lifecycle.

        The container reference should remain accessible after initialization
        and should not be modified during normal server operations.
        """
        mock_container = MagicMock(spec=ModelONEXContainer)
        mock_runtime = MagicMock()

        server = ServiceHealth(container=mock_container, runtime=mock_runtime)

        # Container should be same object (identity check)
        assert server.container is mock_container
        assert server._container is mock_container

        # Multiple accesses should return same object
        container_ref1 = server.container
        container_ref2 = server.container
        assert container_ref1 is container_ref2

    @pytest.mark.asyncio
    async def test_create_from_container_with_none_service_registry(self) -> None:
        """Test factory method when container.service_registry is None.

        When the container has a None service_registry, the factory method should
        raise ProtocolConfigurationError with appropriate error context rather than
        an unhandled AttributeError.
        """
        mock_container = MagicMock(spec=ModelONEXContainer)
        mock_container.service_registry = None

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            await ServiceHealth.create_from_container(container=mock_container)

        # Should indicate resolution failure
        assert "Failed to resolve RuntimeHostProcess from container" in str(
            exc_info.value
        )
        # Original AttributeError should be chained
        assert exc_info.value.__cause__ is not None

    @pytest.mark.asyncio
    async def test_create_from_container_when_resolve_returns_none(self) -> None:
        """Test factory method when service_registry.resolve_service returns None.

        When resolve_service returns None instead of a RuntimeHostProcess instance,
        the factory should still complete but accessing runtime property will raise.
        This tests the edge case where the service registry doesn't throw but also
        doesn't return a valid runtime.
        """
        mock_container = MagicMock(spec=ModelONEXContainer)
        mock_container.service_registry = MagicMock()
        mock_container.service_registry.resolve_service = AsyncMock(return_value=None)

        # Factory should complete (None is a valid return, constructor accepts it)
        server = await ServiceHealth.create_from_container(container=mock_container)

        # Container should be preserved
        assert server.container is mock_container

        # But runtime property should raise since runtime is None
        with pytest.raises(ProtocolConfigurationError) as exc_info:
            _ = server.runtime

        assert "RuntimeHostProcess not available" in str(exc_info.value)

    def test_container_preserved_when_runtime_is_none_after_init(self) -> None:
        """Verify container property returns container when runtime is None.

        When ServiceHealth is initialized with only a container (no runtime),
        the container property should still return the container correctly,
        even though the runtime is None. This ensures container access does
        not depend on runtime being set.
        """
        mock_container = MagicMock(spec=ModelONEXContainer)

        server = ServiceHealth(container=mock_container)

        # Verify runtime is None
        assert server._runtime is None

        # Container should still be accessible and correct
        assert server.container is mock_container
        assert server._container is mock_container

        # Multiple accesses should work consistently
        assert server.container is server.container

    @pytest.mark.asyncio
    async def test_health_endpoint_raises_when_runtime_not_resolved(self) -> None:
        """Test health endpoint behavior when runtime was never resolved.

        When ServiceHealth is initialized with container-only (no runtime)
        and create_from_container was not used, the health endpoint should
        raise ProtocolConfigurationError when trying to access runtime.
        """
        mock_container = MagicMock(spec=ModelONEXContainer)
        server = ServiceHealth(container=mock_container)

        mock_request = MagicMock(spec=web.Request)

        # Health endpoint should return 503 with error details
        response = await server._handle_health(mock_request)

        assert response.status == 503
        response_text = response.text
        assert response_text is not None
        assert '"status":"unhealthy"' in response_text
        # The error should mention runtime not available
        assert "RuntimeHostProcess not available" in response_text

    @pytest.mark.asyncio
    async def test_container_with_service_registry_returning_wrong_type(self) -> None:
        """Test factory when service_registry returns wrong type.

        When resolve_service returns something other than RuntimeHostProcess
        (e.g., a string or different object), the server should still initialize
        but may fail at runtime depending on how the object is used.
        """
        mock_container = MagicMock(spec=ModelONEXContainer)
        mock_container.service_registry = MagicMock()
        # Return a string instead of RuntimeHostProcess
        mock_container.service_registry.resolve_service = AsyncMock(
            return_value="not_a_runtime"
        )

        # Factory completes - Python doesn't enforce type at runtime here
        server = await ServiceHealth.create_from_container(container=mock_container)

        # Container should still be set
        assert server.container is mock_container

        # Runtime returns the wrong type - this is a bug in wiring, not in ServiceHealth
        # The server stores whatever was returned
        assert server._runtime == "not_a_runtime"


@pytest.mark.unit
class TestServiceHealthPrivateAttributeDocumentation:
    """Documentation tests for private attribute access patterns.

    These tests document and validate the use of private attributes in test code.
    Private attributes like `_site`, `_server`, `_app`, `_runner` are accessed
    in integration tests to verify internal state that has no public API.

    IMPORTANT NOTE ON PRIVATE ATTRIBUTE ACCESS:
    -------------------------------------------
    Some integration tests in this module access private attributes such as:

    - `server._site`: The aiohttp TCPSite instance, accessed to get actual
      bound port when using port=0 (auto-assign). There is no public API to
      retrieve the actual port after binding.

    - `server._site._server`: The underlying server object containing sockets,
      accessed to get socket information for verifying port binding.

    - `server._app`, `server._runner`: Internal aiohttp components, accessed
      to verify initialization state in tests.

    - `server._runtime`, `server._container`: Internal state storage, accessed
      to verify proper initialization in unit tests.

    This private attribute access is ACCEPTABLE IN TESTS because:

    1. Tests need to verify internal behavior that has no public API
    2. Port 0 binding requires inspecting actual bound port from sockets
    3. State verification requires checking internal attributes
    4. Breaking changes to internals should break tests (canary behavior)

    This private attribute access should NEVER be done in production code.
    Production code should only use the public API.
    """

    def test_private_attribute_access_documented(self) -> None:
        """Document that integration tests may access private attributes.

        NOTE: Some integration tests access private attributes like `_site` and `_server`
        to get the actual bound port when using port=0 (auto-assign). This is necessary
        because there's no public API to get the actual port after binding.

        This is acceptable in tests but should not be done in production code.

        Private attributes accessed in this test module:
        - _site: TCPSite instance for getting bound socket info
        - _site._server: Server object containing sockets list
        - _app: aiohttp Application instance
        - _runner: aiohttp AppRunner instance
        - _runtime: RuntimeHostProcess reference
        - _container: ModelONEXContainer reference
        - _port, _host, _version: Configuration values
        - _is_running: Server running state flag

        See TestServiceHealthIntegration.test_real_health_endpoint() for an
        example of necessary private attribute access for port discovery.
        """
        # This is a documentation test - verify basic initialization works
        mock_runtime = MagicMock()
        server = ServiceHealth(runtime=mock_runtime)

        # Document the private attributes that exist
        assert hasattr(server, "_site")
        assert hasattr(server, "_app")
        assert hasattr(server, "_runner")
        assert hasattr(server, "_runtime")
        assert hasattr(server, "_container")
        assert hasattr(server, "_port")
        assert hasattr(server, "_host")
        assert hasattr(server, "_version")
        assert hasattr(server, "_is_running")

    def test_port_discovery_requires_private_attribute_access(self) -> None:
        """Document why port=0 tests require private attribute access.

        When using port=0 for automatic port assignment (to avoid test conflicts),
        there is no public API to discover the actual assigned port. The only way
        to get this information is through private attribute access:

            server._site._server.sockets[0].getsockname()[1]

        This pattern is necessary for integration tests that need to make real
        HTTP requests to the server after it binds to an auto-assigned port.

        Alternative approaches considered:
        1. Adding a public `actual_port` property - Would add API surface just for tests
        2. Using fixed ports - Risk of test conflicts in parallel execution
        3. Mocking everything - Loses integration test value

        The chosen approach (private attribute access in tests only) provides
        the best balance of test coverage without polluting the public API.
        """
        # This test documents the pattern, no assertions needed
        # The actual usage is in TestServiceHealthIntegration.test_real_health_endpoint()
