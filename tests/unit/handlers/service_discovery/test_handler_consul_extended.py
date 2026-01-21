# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for extended HandlerServiceDiscoveryConsul methods.

Tests for:
- list_all_services(): Consul catalog API for listing all service names
- get_all_service_instances(): Consul health API for getting all instances

These tests use mocked consul client to validate behavior without requiring
actual Consul server infrastructure.

Related Tickets:
    - OMN-1278: Contract-Driven Dashboard - Registry Discovery (Phase 1b)
"""

from __future__ import annotations

from unittest.mock import MagicMock
from uuid import uuid4

import consul
import pytest

from omnibase_infra.errors import InfraConnectionError, InfraTimeoutError
from omnibase_infra.handlers.service_discovery import HandlerServiceDiscoveryConsul
from omnibase_infra.nodes.node_service_discovery_effect.models.enum_health_status import (
    EnumHealthStatus,
)


@pytest.fixture
def mock_consul_client() -> MagicMock:
    """Provide mocked consul.Consul client for service discovery tests."""
    client = MagicMock()

    # Mock Catalog operations
    client.catalog = MagicMock()
    client.catalog.services = MagicMock(
        return_value=(
            0,
            {
                "my-service": ["web", "api"],
                "other-service": ["db"],
                "consul": [],  # Consul agent itself
            },
        )
    )

    # Mock Health operations with full details
    client.health = MagicMock()
    client.health.service = MagicMock(
        return_value=(
            0,
            [
                {
                    "Node": {
                        "ID": "node-1",
                        "Node": "test-node",
                        "Address": "192.168.1.1",
                    },
                    "Service": {
                        "ID": "550e8400-e29b-41d4-a716-446655440001",
                        "Service": "my-service",
                        "Address": "192.168.1.100",
                        "Port": 8080,
                        "Tags": ["web", "api"],
                        "Meta": {"version": "1.0.0"},
                    },
                    "Checks": [
                        {
                            "ServiceID": "550e8400-e29b-41d4-a716-446655440001",
                            "Status": "passing",
                            "Output": "HTTP GET http://192.168.1.100:8080/health: 200 OK",
                            "Name": "Service check",
                        },
                    ],
                },
                {
                    "Node": {
                        "ID": "node-2",
                        "Node": "test-node-2",
                        "Address": "192.168.1.2",
                    },
                    "Service": {
                        "ID": "550e8400-e29b-41d4-a716-446655440002",
                        "Service": "my-service",
                        "Address": "192.168.1.101",
                        "Port": 8080,
                        "Tags": ["web", "api"],
                        "Meta": {"version": "1.0.0"},
                    },
                    "Checks": [
                        {
                            "ServiceID": "550e8400-e29b-41d4-a716-446655440002",
                            "Status": "critical",
                            "Output": "HTTP GET http://192.168.1.101:8080/health: 503 Service Unavailable",
                            "Name": "Service check",
                        },
                    ],
                },
            ],
        )
    )

    # Mock Status operations (for health check)
    client.status = MagicMock()
    client.status.leader = MagicMock(return_value="192.168.1.1:8300")

    return client


class TestListAllServices:
    """Test HandlerServiceDiscoveryConsul.list_all_services() method."""

    @pytest.mark.asyncio
    async def test_list_all_services_success(
        self,
        mock_consul_client: MagicMock,
    ) -> None:
        """Test successful listing of all services from Consul catalog."""
        handler = HandlerServiceDiscoveryConsul(
            consul_client=mock_consul_client,
        )

        services = await handler.list_all_services()

        assert isinstance(services, dict)
        assert "my-service" in services
        assert "other-service" in services
        assert "consul" in services
        assert services["my-service"] == ["web", "api"]
        assert services["other-service"] == ["db"]
        mock_consul_client.catalog.services.assert_called_once()

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_list_all_services_with_tag_filter(
        self,
        mock_consul_client: MagicMock,
    ) -> None:
        """Test listing services filtered by tag."""
        handler = HandlerServiceDiscoveryConsul(
            consul_client=mock_consul_client,
        )

        # Filter by "web" tag - only my-service has it
        services = await handler.list_all_services(tag_filter="web")

        assert isinstance(services, dict)
        assert "my-service" in services
        assert "other-service" not in services  # Does not have "web" tag
        assert "consul" not in services  # Does not have "web" tag

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_list_all_services_with_db_tag_filter(
        self,
        mock_consul_client: MagicMock,
    ) -> None:
        """Test listing services filtered by 'db' tag."""
        handler = HandlerServiceDiscoveryConsul(
            consul_client=mock_consul_client,
        )

        # Filter by "db" tag - only other-service has it
        services = await handler.list_all_services(tag_filter="db")

        assert isinstance(services, dict)
        assert "other-service" in services
        assert "my-service" not in services
        assert "consul" not in services

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_list_all_services_with_nonexistent_tag(
        self,
        mock_consul_client: MagicMock,
    ) -> None:
        """Test listing services with tag that doesn't exist returns empty."""
        handler = HandlerServiceDiscoveryConsul(
            consul_client=mock_consul_client,
        )

        services = await handler.list_all_services(tag_filter="nonexistent")

        assert isinstance(services, dict)
        assert len(services) == 0

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_list_all_services_with_correlation_id(
        self,
        mock_consul_client: MagicMock,
    ) -> None:
        """Test list_all_services with explicit correlation ID."""
        handler = HandlerServiceDiscoveryConsul(
            consul_client=mock_consul_client,
        )

        correlation_id = uuid4()
        services = await handler.list_all_services(correlation_id=correlation_id)

        assert isinstance(services, dict)
        assert len(services) > 0

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_list_all_services_timeout(
        self,
        mock_consul_client: MagicMock,
    ) -> None:
        """Test list_all_services raises InfraTimeoutError on timeout."""
        handler = HandlerServiceDiscoveryConsul(
            consul_client=mock_consul_client,
            timeout_seconds=0.001,  # Very short timeout
        )

        # Make the catalog call slow
        def slow_services() -> tuple[int, dict[str, list[str]]]:
            import time

            time.sleep(0.1)
            return (0, {})

        mock_consul_client.catalog.services = slow_services

        with pytest.raises(InfraTimeoutError):
            await handler.list_all_services()

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_list_all_services_consul_exception(
        self,
        mock_consul_client: MagicMock,
    ) -> None:
        """Test list_all_services raises InfraConnectionError on Consul error."""
        handler = HandlerServiceDiscoveryConsul(
            consul_client=mock_consul_client,
        )

        mock_consul_client.catalog.services.side_effect = consul.ConsulException(
            "Connection refused"
        )

        with pytest.raises(InfraConnectionError):
            await handler.list_all_services()

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_list_all_services_after_shutdown(
        self,
        mock_consul_client: MagicMock,
    ) -> None:
        """Test list_all_services raises error after handler shutdown.

        Note: When a handler doesn't own its client (injected via constructor),
        shutdown() doesn't clear the client reference. To test the shutdown
        protection, we manually set _consul_client to None after shutdown.
        """
        handler = HandlerServiceDiscoveryConsul(
            consul_client=mock_consul_client,
        )

        await handler.shutdown()

        # Manually clear client to simulate full shutdown
        # (In production, an owned client would be cleared automatically)
        handler._consul_client = None

        with pytest.raises(InfraConnectionError):
            await handler.list_all_services()


class TestGetAllServiceInstances:
    """Test HandlerServiceDiscoveryConsul.get_all_service_instances() method."""

    @pytest.mark.asyncio
    async def test_get_all_service_instances_including_unhealthy(
        self,
        mock_consul_client: MagicMock,
    ) -> None:
        """Test getting all instances including unhealthy ones."""
        handler = HandlerServiceDiscoveryConsul(
            consul_client=mock_consul_client,
        )

        instances = await handler.get_all_service_instances(
            service_name="my-service",
            include_unhealthy=True,
        )

        assert len(instances) == 2
        # Both healthy and unhealthy instances should be returned
        health_statuses = [i.health_status for i in instances]
        assert EnumHealthStatus.HEALTHY in health_statuses
        assert EnumHealthStatus.UNHEALTHY in health_statuses

        # Verify health.service was called with passing=False
        mock_consul_client.health.service.assert_called_once_with(
            "my-service",
            passing=False,  # include_unhealthy=True means passing=False
        )

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_get_all_service_instances_healthy_only(
        self,
        mock_consul_client: MagicMock,
    ) -> None:
        """Test getting only healthy instances."""
        # Set up mock to return only healthy instance when passing=True
        mock_consul_client.health.service.return_value = (
            0,
            [
                {
                    "Node": {
                        "ID": "node-1",
                        "Node": "test-node",
                        "Address": "192.168.1.1",
                    },
                    "Service": {
                        "ID": "550e8400-e29b-41d4-a716-446655440001",
                        "Service": "my-service",
                        "Address": "192.168.1.100",
                        "Port": 8080,
                        "Tags": ["web", "api"],
                        "Meta": {"version": "1.0.0"},
                    },
                    "Checks": [
                        {
                            "ServiceID": "550e8400-e29b-41d4-a716-446655440001",
                            "Status": "passing",
                            "Output": "HTTP GET http://192.168.1.100:8080/health: 200 OK",
                            "Name": "Service check",
                        },
                    ],
                },
            ],
        )

        handler = HandlerServiceDiscoveryConsul(
            consul_client=mock_consul_client,
        )

        instances = await handler.get_all_service_instances(
            service_name="my-service",
            include_unhealthy=False,
        )

        assert len(instances) == 1
        assert instances[0].health_status == EnumHealthStatus.HEALTHY

        # Verify health.service was called with passing=True
        mock_consul_client.health.service.assert_called_once_with(
            "my-service",
            passing=True,  # include_unhealthy=False means passing=True
        )

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_get_all_service_instances_health_output(
        self,
        mock_consul_client: MagicMock,
    ) -> None:
        """Test that health output is captured from Consul checks."""
        handler = HandlerServiceDiscoveryConsul(
            consul_client=mock_consul_client,
        )

        instances = await handler.get_all_service_instances(
            service_name="my-service",
            include_unhealthy=True,
        )

        # Find the healthy instance
        healthy_instance = next(
            i for i in instances if i.health_status == EnumHealthStatus.HEALTHY
        )
        assert healthy_instance.health_output is not None
        assert "200 OK" in healthy_instance.health_output

        # Find the unhealthy instance
        unhealthy_instance = next(
            i for i in instances if i.health_status == EnumHealthStatus.UNHEALTHY
        )
        assert unhealthy_instance.health_output is not None
        assert "503 Service Unavailable" in unhealthy_instance.health_output

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_get_all_service_instances_metadata(
        self,
        mock_consul_client: MagicMock,
    ) -> None:
        """Test that service metadata is captured."""
        handler = HandlerServiceDiscoveryConsul(
            consul_client=mock_consul_client,
        )

        instances = await handler.get_all_service_instances(
            service_name="my-service",
            include_unhealthy=True,
        )

        for instance in instances:
            assert "version" in instance.metadata
            assert instance.metadata["version"] == "1.0.0"

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_get_all_service_instances_tags(
        self,
        mock_consul_client: MagicMock,
    ) -> None:
        """Test that service tags are captured."""
        handler = HandlerServiceDiscoveryConsul(
            consul_client=mock_consul_client,
        )

        instances = await handler.get_all_service_instances(
            service_name="my-service",
            include_unhealthy=True,
        )

        for instance in instances:
            assert "web" in instance.tags
            assert "api" in instance.tags

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_get_all_service_instances_uuid_conversion(
        self,
        mock_consul_client: MagicMock,
    ) -> None:
        """Test that valid UUID service IDs are preserved."""
        handler = HandlerServiceDiscoveryConsul(
            consul_client=mock_consul_client,
        )

        instances = await handler.get_all_service_instances(
            service_name="my-service",
            include_unhealthy=True,
        )

        # Service IDs in mock are valid UUIDs, should be preserved
        from uuid import UUID

        for instance in instances:
            assert isinstance(instance.service_id, UUID)
            # Should match one of the mock UUIDs
            assert str(instance.service_id) in [
                "550e8400-e29b-41d4-a716-446655440001",
                "550e8400-e29b-41d4-a716-446655440002",
            ]

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_get_all_service_instances_non_uuid_service_id(
        self,
        mock_consul_client: MagicMock,
    ) -> None:
        """Test that non-UUID service IDs are converted to deterministic UUID5."""
        # Set up mock with non-UUID service ID
        mock_consul_client.health.service.return_value = (
            0,
            [
                {
                    "Node": {
                        "ID": "node-1",
                        "Node": "test-node",
                        "Address": "192.168.1.1",
                    },
                    "Service": {
                        "ID": "my-service-instance-1",  # Non-UUID ID
                        "Service": "my-service",
                        "Address": "192.168.1.100",
                        "Port": 8080,
                        "Tags": [],
                        "Meta": {},
                    },
                    "Checks": [
                        {
                            "ServiceID": "my-service-instance-1",
                            "Status": "passing",
                            "Output": "OK",
                            "Name": "Service check",
                        },
                    ],
                },
            ],
        )

        handler = HandlerServiceDiscoveryConsul(
            consul_client=mock_consul_client,
        )

        instances = await handler.get_all_service_instances(
            service_name="my-service",
            include_unhealthy=True,
        )

        from uuid import UUID

        assert len(instances) == 1
        # Should be converted to a valid UUID
        assert isinstance(instances[0].service_id, UUID)

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_get_all_service_instances_empty_result(
        self,
        mock_consul_client: MagicMock,
    ) -> None:
        """Test get_all_service_instances returns empty list for unknown service."""
        mock_consul_client.health.service.return_value = (0, [])

        handler = HandlerServiceDiscoveryConsul(
            consul_client=mock_consul_client,
        )

        instances = await handler.get_all_service_instances(
            service_name="nonexistent-service",
            include_unhealthy=True,
        )

        assert instances == []

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_get_all_service_instances_with_correlation_id(
        self,
        mock_consul_client: MagicMock,
    ) -> None:
        """Test get_all_service_instances with explicit correlation ID."""
        handler = HandlerServiceDiscoveryConsul(
            consul_client=mock_consul_client,
        )

        correlation_id = uuid4()
        instances = await handler.get_all_service_instances(
            service_name="my-service",
            include_unhealthy=True,
            correlation_id=correlation_id,
        )

        # All instances should have the correlation ID set
        for instance in instances:
            assert instance.correlation_id == correlation_id

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_get_all_service_instances_timeout(
        self,
        mock_consul_client: MagicMock,
    ) -> None:
        """Test get_all_service_instances raises InfraTimeoutError on timeout."""
        handler = HandlerServiceDiscoveryConsul(
            consul_client=mock_consul_client,
            timeout_seconds=0.001,  # Very short timeout
        )

        # Make the health call slow
        def slow_health(service_name: str, passing: bool = False) -> tuple[int, list]:
            import time

            time.sleep(0.1)
            return (0, [])

        mock_consul_client.health.service = slow_health

        with pytest.raises(InfraTimeoutError):
            await handler.get_all_service_instances(
                service_name="my-service",
                include_unhealthy=True,
            )

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_get_all_service_instances_consul_exception(
        self,
        mock_consul_client: MagicMock,
    ) -> None:
        """Test get_all_service_instances raises InfraConnectionError on Consul error."""
        handler = HandlerServiceDiscoveryConsul(
            consul_client=mock_consul_client,
        )

        mock_consul_client.health.service.side_effect = consul.ConsulException(
            "Connection refused"
        )

        with pytest.raises(InfraConnectionError):
            await handler.get_all_service_instances(
                service_name="my-service",
                include_unhealthy=True,
            )

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_get_all_service_instances_after_shutdown(
        self,
        mock_consul_client: MagicMock,
    ) -> None:
        """Test get_all_service_instances raises error after handler shutdown.

        Note: When a handler doesn't own its client (injected via constructor),
        shutdown() doesn't clear the client reference. To test the shutdown
        protection, we manually set _consul_client to None after shutdown.
        """
        handler = HandlerServiceDiscoveryConsul(
            consul_client=mock_consul_client,
        )

        await handler.shutdown()

        # Manually clear client to simulate full shutdown
        # (In production, an owned client would be cleared automatically)
        handler._consul_client = None

        with pytest.raises(InfraConnectionError):
            await handler.get_all_service_instances(
                service_name="my-service",
                include_unhealthy=True,
            )


class TestHealthStatusMapping:
    """Test health status mapping from Consul check statuses."""

    @pytest.mark.asyncio
    async def test_passing_status_maps_to_healthy(
        self,
        mock_consul_client: MagicMock,
    ) -> None:
        """Test that 'passing' check status maps to HEALTHY."""
        mock_consul_client.health.service.return_value = (
            0,
            [
                {
                    "Node": {"Address": "192.168.1.1"},
                    "Service": {
                        "ID": "550e8400-e29b-41d4-a716-446655440001",
                        "Service": "test-service",
                        "Address": "192.168.1.100",
                        "Port": 8080,
                        "Tags": [],
                        "Meta": {},
                    },
                    "Checks": [
                        {
                            "ServiceID": "550e8400-e29b-41d4-a716-446655440001",
                            "Status": "passing",
                            "Output": "OK",
                        },
                    ],
                },
            ],
        )

        handler = HandlerServiceDiscoveryConsul(consul_client=mock_consul_client)
        instances = await handler.get_all_service_instances(
            service_name="test-service",
            include_unhealthy=True,
        )

        assert len(instances) == 1
        assert instances[0].health_status == EnumHealthStatus.HEALTHY

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_critical_status_maps_to_unhealthy(
        self,
        mock_consul_client: MagicMock,
    ) -> None:
        """Test that 'critical' check status maps to UNHEALTHY."""
        mock_consul_client.health.service.return_value = (
            0,
            [
                {
                    "Node": {"Address": "192.168.1.1"},
                    "Service": {
                        "ID": "550e8400-e29b-41d4-a716-446655440001",
                        "Service": "test-service",
                        "Address": "192.168.1.100",
                        "Port": 8080,
                        "Tags": [],
                        "Meta": {},
                    },
                    "Checks": [
                        {
                            "ServiceID": "550e8400-e29b-41d4-a716-446655440001",
                            "Status": "critical",
                            "Output": "Service down",
                        },
                    ],
                },
            ],
        )

        handler = HandlerServiceDiscoveryConsul(consul_client=mock_consul_client)
        instances = await handler.get_all_service_instances(
            service_name="test-service",
            include_unhealthy=True,
        )

        assert len(instances) == 1
        assert instances[0].health_status == EnumHealthStatus.UNHEALTHY

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_warning_status_maps_to_unhealthy(
        self,
        mock_consul_client: MagicMock,
    ) -> None:
        """Test that 'warning' check status maps to UNHEALTHY."""
        mock_consul_client.health.service.return_value = (
            0,
            [
                {
                    "Node": {"Address": "192.168.1.1"},
                    "Service": {
                        "ID": "550e8400-e29b-41d4-a716-446655440001",
                        "Service": "test-service",
                        "Address": "192.168.1.100",
                        "Port": 8080,
                        "Tags": [],
                        "Meta": {},
                    },
                    "Checks": [
                        {
                            "ServiceID": "550e8400-e29b-41d4-a716-446655440001",
                            "Status": "warning",
                            "Output": "Response slow",
                        },
                    ],
                },
            ],
        )

        handler = HandlerServiceDiscoveryConsul(consul_client=mock_consul_client)
        instances = await handler.get_all_service_instances(
            service_name="test-service",
            include_unhealthy=True,
        )

        assert len(instances) == 1
        assert instances[0].health_status == EnumHealthStatus.UNHEALTHY

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_unknown_status_maps_to_unknown(
        self,
        mock_consul_client: MagicMock,
    ) -> None:
        """Test that unknown check status maps to UNKNOWN."""
        mock_consul_client.health.service.return_value = (
            0,
            [
                {
                    "Node": {"Address": "192.168.1.1"},
                    "Service": {
                        "ID": "550e8400-e29b-41d4-a716-446655440001",
                        "Service": "test-service",
                        "Address": "192.168.1.100",
                        "Port": 8080,
                        "Tags": [],
                        "Meta": {},
                    },
                    "Checks": [
                        {
                            "ServiceID": "550e8400-e29b-41d4-a716-446655440001",
                            "Status": "unknown_status",
                            "Output": "",
                        },
                    ],
                },
            ],
        )

        handler = HandlerServiceDiscoveryConsul(consul_client=mock_consul_client)
        instances = await handler.get_all_service_instances(
            service_name="test-service",
            include_unhealthy=True,
        )

        assert len(instances) == 1
        assert instances[0].health_status == EnumHealthStatus.UNKNOWN

        await handler.shutdown()

    @pytest.mark.asyncio
    async def test_no_checks_maps_to_unknown(
        self,
        mock_consul_client: MagicMock,
    ) -> None:
        """Test that service with no checks maps to UNKNOWN."""
        mock_consul_client.health.service.return_value = (
            0,
            [
                {
                    "Node": {"Address": "192.168.1.1"},
                    "Service": {
                        "ID": "550e8400-e29b-41d4-a716-446655440001",
                        "Service": "test-service",
                        "Address": "192.168.1.100",
                        "Port": 8080,
                        "Tags": [],
                        "Meta": {},
                    },
                    "Checks": [],  # No health checks
                },
            ],
        )

        handler = HandlerServiceDiscoveryConsul(consul_client=mock_consul_client)
        instances = await handler.get_all_service_instances(
            service_name="test-service",
            include_unhealthy=True,
        )

        assert len(instances) == 1
        assert instances[0].health_status == EnumHealthStatus.UNKNOWN

        await handler.shutdown()


__all__: list[str] = [
    "TestListAllServices",
    "TestGetAllServiceInstances",
    "TestHealthStatusMapping",
]
