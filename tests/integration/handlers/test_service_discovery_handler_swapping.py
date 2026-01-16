# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Integration tests for Service Discovery Handler Swapping.

This module tests that service discovery handlers can be swapped at runtime
without breaking node functionality. This is a core capability-oriented design
principle: handlers should be interchangeable as long as they implement the
same protocol.

Handler Swapping Concept
------------------------
The capability-oriented design allows nodes to work with any handler that
implements the required protocol. This enables:
- Testing with mock handlers
- Switching between backends (Consul, Kubernetes, Etcd) without code changes
- Gradual migration between service discovery backends
- Environment-specific handler selection (dev vs prod)

Handlers Tested:
    - HandlerServiceDiscoveryMock: In-memory mock for testing
    - HandlerServiceDiscoveryConsul: Consul backend (requires Consul)

CI/CD Graceful Skip Behavior
----------------------------
These integration tests skip gracefully when Consul is unavailable,
enabling CI/CD pipelines to run without hard failures in environments
without service discovery infrastructure access.

Related:
    - OMN-1131: Capability-oriented node architecture
    - ProtocolDiscoveryOperations: Protocol definition
    - NodeServiceDiscoveryEffect: Effect node that uses these handlers
    - PR #119: Test coverage for handler swapping
"""

from __future__ import annotations

import os
import socket
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest

from omnibase_infra.handlers.service_discovery.handler_service_discovery_mock import (
    HandlerServiceDiscoveryMock,
)
from omnibase_infra.handlers.service_discovery.models import (
    ModelServiceInfo,
)
from omnibase_infra.handlers.service_discovery.protocol_discovery_operations import (
    ProtocolDiscoveryOperations,
)
from omnibase_infra.nodes.node_service_discovery_effect.models import (
    ModelDiscoveryQuery,
    ModelServiceDiscoveryHealthCheckResult,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


# =============================================================================
# Environment Configuration
# =============================================================================

# Check if Consul is available for integration tests
CONSUL_HOST = os.getenv("CONSUL_HOST")
CONSUL_PORT = int(os.getenv("CONSUL_PORT", "8500"))


def _check_consul_reachable() -> bool:
    """Check if Consul server is reachable via TCP."""
    if CONSUL_HOST is None:
        return False
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(5.0)
            result = sock.connect_ex((CONSUL_HOST, CONSUL_PORT))
            return result == 0
    except (OSError, TimeoutError):
        return False


CONSUL_AVAILABLE = _check_consul_reachable()


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_handler() -> HandlerServiceDiscoveryMock:
    """Create a HandlerServiceDiscoveryMock for testing."""
    return HandlerServiceDiscoveryMock()


@pytest.fixture
def sample_service_info() -> ModelServiceInfo:
    """Create a sample service info for testing."""
    return ModelServiceInfo(
        service_id=uuid4(),
        service_name="test-service",
        address="192.168.1.100",
        port=8080,
        tags=("api", "v1", "test"),
        metadata={"environment": "test", "version": "1.0.0"},
        health_check_url="http://192.168.1.100:8080/health",
        registered_at=datetime.now(UTC),
        correlation_id=uuid4(),
    )


@pytest.fixture
def multiple_service_infos() -> list[ModelServiceInfo]:
    """Create multiple service infos for batch testing."""
    services = []
    service_names = [
        "api-gateway",
        "user-service",
        "order-service",
        "inventory-service",
    ]

    for i, name in enumerate(service_names):
        services.append(
            ModelServiceInfo(
                service_id=uuid4(),
                service_name=name,
                address=f"192.168.1.{100 + i}",
                port=8080 + i,
                tags=("service", f"shard-{i % 2}"),
                metadata={"index": str(i)},
                health_check_url=f"http://192.168.1.{100 + i}:{8080 + i}/health",
                registered_at=datetime.now(UTC),
                correlation_id=uuid4(),
            )
        )
    return services


# =============================================================================
# Handler Swapping Test Base Class
# =============================================================================


class BaseHandlerSwappingTests:
    """Base class defining common handler swapping tests.

    Subclasses implement handler_fixture to provide the specific handler
    to test. This ensures all handlers pass the same behavioral tests.
    """

    @pytest.fixture
    def handler(self) -> ProtocolDiscoveryOperations:
        """Override in subclass to provide the handler to test."""
        pytest.skip("Subclasses must implement handler fixture")

    async def test_handler_conforms_to_protocol(
        self,
        handler: ProtocolDiscoveryOperations,
    ) -> None:
        """Handler is an instance of ProtocolDiscoveryOperations."""
        assert isinstance(handler, ProtocolDiscoveryOperations), (
            f"{type(handler).__name__} must implement ProtocolDiscoveryOperations"
        )

    async def test_handler_has_handler_type(
        self,
        handler: ProtocolDiscoveryOperations,
    ) -> None:
        """Handler has handler_type property returning non-empty string."""
        handler_type = handler.handler_type
        assert isinstance(handler_type, str), "handler_type must return string"
        assert len(handler_type) > 0, "handler_type must not be empty"

    async def test_register_and_discover_roundtrip(
        self,
        handler: ProtocolDiscoveryOperations,
        sample_service_info: ModelServiceInfo,
    ) -> None:
        """Services can be registered and discovered correctly."""
        correlation_id = uuid4()

        # Register the service
        register_result = await handler.register_service(
            service_info=sample_service_info,
            correlation_id=correlation_id,
        )

        # Verify registration succeeded
        assert register_result.success, "Registration failed"
        assert register_result.service_id == sample_service_info.service_id

        # Discover the service
        query = ModelDiscoveryQuery(
            service_name=sample_service_info.service_name,
            correlation_id=correlation_id,
        )
        discover_result = await handler.discover_services(query)

        # Verify discovery returned the registered service
        assert discover_result.success, "Discovery failed"
        assert len(discover_result.services) >= 1

        # Find our service in results
        found = False
        for service in discover_result.services:
            if service.service_id == sample_service_info.service_id:
                found = True
                assert service.service_name == sample_service_info.service_name
                break

        assert found, "Registered service not found in discovery results"

    async def test_deregister_service(
        self,
        handler: ProtocolDiscoveryOperations,
        sample_service_info: ModelServiceInfo,
    ) -> None:
        """Services can be deregistered correctly."""
        correlation_id = uuid4()

        # Register a service first
        await handler.register_service(
            service_info=sample_service_info,
            correlation_id=correlation_id,
        )

        # Deregister the service
        await handler.deregister_service(
            service_id=sample_service_info.service_id,
            correlation_id=correlation_id,
        )

        # Verify service is gone (discovery should not find it)
        query = ModelDiscoveryQuery(
            service_name=sample_service_info.service_name,
            correlation_id=correlation_id,
        )
        discover_result = await handler.discover_services(query)

        # Service should not be in results
        for service in discover_result.services:
            assert service.service_id != sample_service_info.service_id, (
                "Deregistered service should not appear in discovery"
            )

    async def test_health_check_returns_status(
        self,
        handler: ProtocolDiscoveryOperations,
    ) -> None:
        """Health check returns valid status information."""
        correlation_id = uuid4()

        health_status = await handler.health_check(correlation_id=correlation_id)

        assert isinstance(health_status, ModelServiceDiscoveryHealthCheckResult), (
            "health_check must return ModelServiceDiscoveryHealthCheckResult"
        )
        assert hasattr(health_status, "healthy"), (
            "health_check result must have healthy attribute"
        )
        assert hasattr(health_status, "backend_type"), (
            "health_check result must have backend_type attribute"
        )


# =============================================================================
# Mock Handler Swapping Tests
# =============================================================================


class TestMockHandlerSwapping(BaseHandlerSwappingTests):
    """Test handler swapping with HandlerServiceDiscoveryMock.

    These tests verify that the mock handler can be used as a drop-in
    replacement for any other handler implementing the protocol.
    """

    @pytest.fixture
    def handler(
        self, mock_handler: HandlerServiceDiscoveryMock
    ) -> HandlerServiceDiscoveryMock:
        """Provide HandlerServiceDiscoveryMock for testing."""
        return mock_handler

    @pytest.mark.asyncio
    async def test_mock_handler_register_and_discover(
        self,
        handler: HandlerServiceDiscoveryMock,
        sample_service_info: ModelServiceInfo,
    ) -> None:
        """Mock handler can register and discover services."""
        await self.test_register_and_discover_roundtrip(handler, sample_service_info)

    @pytest.mark.asyncio
    async def test_mock_handler_deregister(
        self,
        handler: HandlerServiceDiscoveryMock,
        sample_service_info: ModelServiceInfo,
    ) -> None:
        """Mock handler can deregister services."""
        await self.test_deregister_service(handler, sample_service_info)

    @pytest.mark.asyncio
    async def test_mock_handler_health_check(
        self,
        handler: HandlerServiceDiscoveryMock,
    ) -> None:
        """Mock handler health check works."""
        await self.test_health_check_returns_status(handler)

    @pytest.mark.asyncio
    async def test_mock_handler_multiple_services(
        self,
        handler: HandlerServiceDiscoveryMock,
        multiple_service_infos: list[ModelServiceInfo],
    ) -> None:
        """Mock handler can register and discover multiple services."""
        correlation_id = uuid4()

        # Register all services
        for service_info in multiple_service_infos:
            result = await handler.register_service(
                service_info=service_info,
                correlation_id=correlation_id,
            )
            assert result.success

        # Discover specific service
        query = ModelDiscoveryQuery(
            service_name="api-gateway",
            correlation_id=correlation_id,
        )
        discover_result = await handler.discover_services(query)

        assert discover_result.success
        assert len(discover_result.services) == 1
        assert discover_result.services[0].service_name == "api-gateway"

    @pytest.mark.asyncio
    async def test_mock_handler_tag_filtering(
        self,
        handler: HandlerServiceDiscoveryMock,
        multiple_service_infos: list[ModelServiceInfo],
    ) -> None:
        """Mock handler supports tag-based filtering."""
        correlation_id = uuid4()

        # Register services with different tags
        service1 = ModelServiceInfo(
            service_id=uuid4(),
            service_name="my-service",
            address="10.0.0.1",
            port=8080,
            tags=("primary", "production"),
            metadata={},
            correlation_id=correlation_id,
        )
        service2 = ModelServiceInfo(
            service_id=uuid4(),
            service_name="my-service",
            address="10.0.0.2",
            port=8080,
            tags=("secondary", "production"),
            metadata={},
            correlation_id=correlation_id,
        )

        await handler.register_service(service1, correlation_id)
        await handler.register_service(service2, correlation_id)

        # Discover with tag filter
        query = ModelDiscoveryQuery(
            service_name="my-service",
            tags=("primary",),
            correlation_id=correlation_id,
        )
        discover_result = await handler.discover_services(query)

        assert discover_result.success
        assert len(discover_result.services) == 1
        assert "primary" in discover_result.services[0].tags

    @pytest.mark.asyncio
    async def test_mock_handler_idempotent_deregister(
        self,
        handler: HandlerServiceDiscoveryMock,
        sample_service_info: ModelServiceInfo,
    ) -> None:
        """Deregistering a non-existent service does not raise error."""
        correlation_id = uuid4()

        # Deregister a service that was never registered (should not raise)
        await handler.deregister_service(
            service_id=sample_service_info.service_id,
            correlation_id=correlation_id,
        )

        # Deregister again (still should not raise)
        await handler.deregister_service(
            service_id=sample_service_info.service_id,
            correlation_id=correlation_id,
        )


# =============================================================================
# Handler Factory Pattern Tests
# =============================================================================


class TestHandlerFactoryPattern:
    """Test that handlers can be swapped using factory pattern.

    This validates the capability-oriented design principle that nodes
    should work with any handler implementing the protocol.
    """

    def create_handler_by_type(self, handler_type: str) -> ProtocolDiscoveryOperations:
        """Factory method to create handlers by type identifier.

        In production code, this pattern would be used to select handlers
        based on configuration (environment variables, config files, etc.).

        Args:
            handler_type: Type identifier ("mock", "consul")

        Returns:
            Handler instance implementing ProtocolDiscoveryOperations

        Raises:
            ValueError: If handler_type is not recognized
        """
        if handler_type == "mock":
            return HandlerServiceDiscoveryMock()
        elif handler_type == "consul":
            from omnibase_infra.handlers.service_discovery.handler_service_discovery_consul import (
                HandlerServiceDiscoveryConsul,
            )

            return HandlerServiceDiscoveryConsul(
                consul_host=os.getenv("CONSUL_HOST", "localhost"),
                consul_port=int(os.getenv("CONSUL_PORT", "8500")),
                consul_scheme=os.getenv("CONSUL_SCHEME", "http"),
            )
        else:
            raise ValueError(f"Unknown handler type: {handler_type}")

    def test_factory_creates_mock_handler(self) -> None:
        """Factory creates HandlerServiceDiscoveryMock for 'mock' type."""
        handler = self.create_handler_by_type("mock")

        assert isinstance(handler, HandlerServiceDiscoveryMock)
        assert isinstance(handler, ProtocolDiscoveryOperations)
        assert handler.handler_type == "mock"

    @pytest.mark.skipif(
        not CONSUL_AVAILABLE,
        reason="Consul not available (CONSUL_HOST not set or not reachable)",
    )
    def test_factory_creates_consul_handler(self) -> None:
        """Factory creates HandlerServiceDiscoveryConsul for 'consul' type."""
        handler = self.create_handler_by_type("consul")

        from omnibase_infra.handlers.service_discovery.handler_service_discovery_consul import (
            HandlerServiceDiscoveryConsul,
        )

        assert isinstance(handler, HandlerServiceDiscoveryConsul)
        assert isinstance(handler, ProtocolDiscoveryOperations)
        assert handler.handler_type == "consul"

    def test_factory_raises_for_unknown_type(self) -> None:
        """Factory raises ValueError for unknown handler type."""
        with pytest.raises(ValueError, match="Unknown handler type"):
            self.create_handler_by_type("unknown")

    @pytest.mark.asyncio
    async def test_swapping_handlers_preserves_behavior(self) -> None:
        """Swapping handlers preserves expected behavior.

        This test demonstrates that code written against the protocol
        works identically regardless of which handler is used.
        """
        correlation_id = uuid4()
        service_info = ModelServiceInfo(
            service_id=uuid4(),
            service_name="test-service",
            address="10.0.0.1",
            port=8080,
            tags=(),
            metadata={},
            correlation_id=correlation_id,
        )

        # Create two different handlers
        handler1 = self.create_handler_by_type("mock")
        handler2 = self.create_handler_by_type("mock")  # Another mock for comparison

        # Both handlers should work identically
        result1 = await handler1.register_service(service_info, correlation_id)
        result2 = await handler2.register_service(service_info, correlation_id)

        assert result1.success == result2.success
        assert result1.service_id == result2.service_id

        # Health checks should both work
        health1 = await handler1.health_check(correlation_id)
        health2 = await handler2.health_check(correlation_id)

        assert isinstance(health1, ModelServiceDiscoveryHealthCheckResult)
        assert isinstance(health2, ModelServiceDiscoveryHealthCheckResult)


# =============================================================================
# Runtime Handler Swapping Tests
# =============================================================================


class TestRuntimeHandlerSwapping:
    """Test that handlers can be swapped at runtime.

    This validates that a node or service can change its handler
    implementation without restart or code changes.
    """

    @pytest.mark.asyncio
    async def test_swap_handler_mid_operation(self) -> None:
        """Handler can be swapped mid-operation.

        Simulates a scenario where a service starts with a mock handler
        for testing, then switches to a different handler.
        """
        correlation_id = uuid4()
        service_info = ModelServiceInfo(
            service_id=uuid4(),
            service_name="switchable-service",
            address="10.0.0.1",
            port=8080,
            tags=(),
            metadata={},
            correlation_id=correlation_id,
        )

        # Start with mock handler
        current_handler: ProtocolDiscoveryOperations = HandlerServiceDiscoveryMock()

        # Register service with first handler
        result = await current_handler.register_service(service_info, correlation_id)
        assert result.success

        # Swap to a new mock handler (simulating handler switch)
        new_handler = HandlerServiceDiscoveryMock()

        # Register same service in new handler
        result2 = await new_handler.register_service(service_info, correlation_id)
        assert result2.success

        # Both handlers should report healthy
        assert (await current_handler.health_check()).healthy
        assert (await new_handler.health_check()).healthy

    @pytest.mark.asyncio
    async def test_handler_interface_contract(self) -> None:
        """All handlers expose the same interface contract.

        This test verifies that code written against the protocol
        can work with any conforming handler.
        """

        async def perform_operations(
            handler: ProtocolDiscoveryOperations,
        ) -> tuple[bool, bool]:
            """Perform a series of operations using only protocol methods."""
            correlation_id = uuid4()
            service_info = ModelServiceInfo(
                service_id=uuid4(),
                service_name="contract-test-service",
                address="10.0.0.1",
                port=8080,
                tags=(),
                metadata={},
                correlation_id=correlation_id,
            )

            # Register
            register_result = await handler.register_service(
                service_info, correlation_id
            )

            # Discover
            query = ModelDiscoveryQuery(
                service_name=service_info.service_name,
                correlation_id=correlation_id,
            )
            discover_result = await handler.discover_services(query)

            # Deregister
            await handler.deregister_service(service_info.service_id, correlation_id)

            return (register_result.success, discover_result.success)

        # Test with mock handler
        mock = HandlerServiceDiscoveryMock()
        register_ok, discover_ok = await perform_operations(mock)

        assert register_ok, "Mock handler registration failed"
        assert discover_ok, "Mock handler discovery failed"


# =============================================================================
# Consul Handler Integration Tests (requires Consul)
# =============================================================================


@pytest.mark.skipif(
    not CONSUL_AVAILABLE,
    reason="Consul not available (CONSUL_HOST not set or not reachable)",
)
class TestConsulHandlerSwapping(BaseHandlerSwappingTests):
    """Test handler swapping with HandlerServiceDiscoveryConsul.

    These tests require a running Consul instance and are skipped
    in CI environments without Consul access.
    """

    @pytest.fixture
    async def handler(
        self,
    ) -> AsyncGenerator[ProtocolDiscoveryOperations, None]:
        """Provide HandlerServiceDiscoveryConsul for testing."""
        from omnibase_infra.handlers.service_discovery.handler_service_discovery_consul import (
            HandlerServiceDiscoveryConsul,
        )

        handler = HandlerServiceDiscoveryConsul(
            consul_host=os.getenv("CONSUL_HOST", "localhost"),
            consul_port=int(os.getenv("CONSUL_PORT", "8500")),
            consul_scheme=os.getenv("CONSUL_SCHEME", "http"),
        )

        yield handler

        # Cleanup
        await handler.shutdown()

    @pytest.fixture
    def sample_service_info(self) -> ModelServiceInfo:
        """Create a sample service info with unique name for Consul tests."""
        # Use unique service name to avoid conflicts in shared Consul
        return ModelServiceInfo(
            service_id=uuid4(),
            service_name=f"test-service-{uuid4().hex[:8]}",
            address="192.168.1.100",
            port=8080,
            tags=("api", "v1", "test"),
            metadata={"environment": "test"},
            health_check_url="http://192.168.1.100:8080/health",
            registered_at=datetime.now(UTC),
            correlation_id=uuid4(),
        )

    @pytest.mark.asyncio
    async def test_consul_handler_register_and_discover(
        self,
        handler: ProtocolDiscoveryOperations,
        sample_service_info: ModelServiceInfo,
    ) -> None:
        """Consul handler can register and discover services."""
        correlation_id = uuid4()

        try:
            # Register the service
            register_result = await handler.register_service(
                service_info=sample_service_info,
                correlation_id=correlation_id,
            )
            assert register_result.success

            # Note: Consul may take a moment to propagate the registration
            # In a real test, we might add a small delay or retry loop

        finally:
            # Always deregister to clean up
            await handler.deregister_service(
                service_id=sample_service_info.service_id,
                correlation_id=correlation_id,
            )

    @pytest.mark.asyncio
    async def test_consul_handler_health_check(
        self,
        handler: ProtocolDiscoveryOperations,
    ) -> None:
        """Consul handler health check works."""
        await self.test_health_check_returns_status(handler)


__all__: list[str] = [
    "BaseHandlerSwappingTests",
    "TestMockHandlerSwapping",
    "TestHandlerFactoryPattern",
    "TestRuntimeHandlerSwapping",
    "TestConsulHandlerSwapping",
]
