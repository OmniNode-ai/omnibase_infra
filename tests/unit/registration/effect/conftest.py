# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Shared pytest fixtures for Registry Effect Node tests.

Provides common fixtures for OMN-954 (Registry Effect Node tests) including:
- Idempotency store instances for deduplication testing
- Mock infrastructure clients (Consul, PostgreSQL)
- Sample registration models and events
- Correlation ID fixtures for tracing

Usage:
    Fixtures are automatically available to all tests in this package
    and its subpackages. Import models directly in test files as needed.

Example:
    >>> def test_registration_idempotency(
    ...     inmemory_idempotency_store,
    ...     sample_introspection_event,
    ...     correlation_id,
    ... ):
    ...     # Use fixtures directly
    ...     store = inmemory_idempotency_store
    ...     event = sample_introspection_event
    ...     ...

Related Tickets:
    - OMN-954: Registry Effect Node tests
    - OMN-890: Registry Effect Node implementation
    - OMN-945: Idempotency system
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Literal
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

from omnibase_infra.idempotency import InMemoryIdempotencyStore
from omnibase_infra.models.registration import (
    ModelNodeCapabilities,
    ModelNodeIntrospectionEvent,
    ModelNodeMetadata,
    ModelNodeRegistration,
)
from tests.helpers import create_introspection_event

# -----------------------------------------------------------------------------
# Idempotency Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def inmemory_idempotency_store() -> InMemoryIdempotencyStore:
    """Create InMemoryIdempotencyStore for testing.

    Returns a fresh in-memory idempotency store instance suitable for
    testing registration deduplication without external dependencies.

    Returns:
        InMemoryIdempotencyStore: A new in-memory idempotency store.

    Example:
        >>> async def test_dedup(inmemory_idempotency_store):
        ...     store = inmemory_idempotency_store
        ...     msg_id = uuid4()
        ...     assert await store.check_and_record(msg_id) is True
        ...     assert await store.check_and_record(msg_id) is False
    """
    return InMemoryIdempotencyStore()


@pytest.fixture
async def initialized_idempotency_store() -> AsyncIterator[InMemoryIdempotencyStore]:
    """Create and initialize InMemoryIdempotencyStore for async tests.

    Yields an initialized store instance and ensures cleanup after test.

    Yields:
        InMemoryIdempotencyStore: An initialized in-memory store.
    """
    store = InMemoryIdempotencyStore()
    yield store
    await store.clear()


# -----------------------------------------------------------------------------
# Mock Infrastructure Client Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def mock_consul_client() -> MagicMock:
    """Create mock Consul client for service registration testing.

    Provides a fully mocked consul.Consul client with common operations
    pre-configured for success. Customize return values in individual
    tests as needed.

    Returns:
        MagicMock: Mocked Consul client with agent, catalog, health,
            and status operations.

    Example:
        >>> def test_registration(mock_consul_client):
        ...     # Override for specific test
        ...     mock_consul_client.agent.service.register.side_effect = Exception("fail")
        ...     ...
    """
    client = MagicMock()

    # Mock Agent operations (service registration)
    client.agent = MagicMock()
    client.agent.service = MagicMock()
    client.agent.service.register = AsyncMock(return_value=True)
    client.agent.service.deregister = AsyncMock(return_value=True)
    client.agent.services = MagicMock(
        return_value={
            "node-service-1": {
                "ID": "node-service-1",
                "Service": "registry-effect",
                "Address": "192.168.1.100",
                "Port": 8080,
            }
        }
    )

    # Mock Catalog operations
    client.catalog = MagicMock()
    client.catalog.services = MagicMock(
        return_value=(0, {"registry-effect": [], "other-service": []})
    )
    client.catalog.service = MagicMock(
        return_value=(
            0,
            [
                {
                    "ServiceID": "node-service-1",
                    "ServiceName": "registry-effect",
                    "ServiceAddress": "192.168.1.100",
                    "ServicePort": 8080,
                }
            ],
        )
    )

    # Mock Health operations
    client.health = MagicMock()
    client.health.service = MagicMock(
        return_value=(
            0,
            [
                {
                    "Service": {
                        "ID": "node-service-1",
                        "Service": "registry-effect",
                        "Address": "192.168.1.100",
                        "Port": 8080,
                    },
                    "Checks": [
                        {"Status": "passing", "Name": "Service check"},
                    ],
                }
            ],
        )
    )

    # Mock Status operations (for health check)
    client.status = MagicMock()
    client.status.leader = MagicMock(return_value="192.168.1.1:8300")

    return client


@pytest.fixture
def mock_postgres_handler() -> AsyncMock:
    """Create mock PostgreSQL handler for registration persistence.

    Provides a fully mocked async database handler with common operations
    pre-configured for success. Customize return values in individual
    tests as needed.

    Returns:
        AsyncMock: Mocked PostgreSQL handler with execute, query,
            and health check operations.

    Example:
        >>> async def test_persistence(mock_postgres_handler):
        ...     mock_postgres_handler.execute.return_value = {"rows_affected": 1}
        ...     ...
    """
    handler = AsyncMock()

    # Initialize state
    handler._initialized = True
    handler.handler_type = "database"

    # Mock execute operation (INSERT, UPDATE, DELETE)
    handler.execute = AsyncMock(
        return_value=MagicMock(
            result=MagicMock(
                status="success",
                payload=MagicMock(data=MagicMock(rows_affected=1)),
            )
        )
    )

    # Mock query operation (SELECT)
    handler.query = AsyncMock(
        return_value=MagicMock(
            result=MagicMock(
                status="success",
                payload=MagicMock(data=MagicMock(rows=[], row_count=0)),
            )
        )
    )

    # Mock health check
    handler.health_check = AsyncMock(
        return_value={
            "healthy": True,
            "initialized": True,
            "handler_type": "database",
        }
    )

    # Mock initialization and shutdown
    handler.initialize = AsyncMock()
    handler.shutdown = AsyncMock()

    return handler


# -----------------------------------------------------------------------------
# Sample Model Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def correlation_id() -> UUID:
    """Create a UUID correlation ID for request tracing.

    Returns:
        UUID: A fresh UUID4 for correlation tracking.
    """
    return uuid4()


@pytest.fixture
def sample_node_id() -> UUID:
    """Create a sample node ID for testing.

    Returns:
        UUID: A fresh UUID4 representing a node ID.
    """
    return uuid4()


@pytest.fixture
def sample_introspection_event(
    sample_node_id: UUID,
    correlation_id: UUID,
) -> ModelNodeIntrospectionEvent:
    """Create a sample introspection event for testing.

    Provides a fully populated ModelNodeIntrospectionEvent with all
    required and optional fields set to sensible test values.

    Args:
        sample_node_id: Node ID fixture (auto-injected).
        correlation_id: Correlation ID fixture (auto-injected).

    Returns:
        ModelNodeIntrospectionEvent: A valid introspection event for testing.

    Example:
        >>> def test_process_event(sample_introspection_event):
        ...     event = sample_introspection_event
        ...     assert event.node_type == "effect"
        ...     assert event.node_version == "1.0.0"
    """
    return ModelNodeIntrospectionEvent(
        node_id=sample_node_id,
        node_type="effect",
        node_version="1.0.0",
        correlation_id=correlation_id,
        endpoints={
            "health": "http://localhost:8080/health",
            "api": "http://localhost:8080/api",
        },
        capabilities=ModelNodeCapabilities(
            postgres=True,
            read=True,
            write=True,
            database=True,
        ),
        metadata=ModelNodeMetadata(
            environment="test",
            region="us-west-2",
            cluster="test-cluster",
        ),
        node_role="adapter",
        network_id="test-network",
        deployment_id="test-deployment",
        epoch=1,
        timestamp=datetime.now(UTC),  # Required: time injection pattern
    )


@pytest.fixture
def sample_node_registration(
    sample_node_id: UUID,
) -> ModelNodeRegistration:
    """Create a sample node registration for testing.

    Provides a fully populated ModelNodeRegistration with all
    required and optional fields set to sensible test values.

    Args:
        sample_node_id: Node ID fixture (auto-injected).

    Returns:
        ModelNodeRegistration: A valid registration model for testing.
    """
    now = datetime.now(UTC)
    return ModelNodeRegistration(
        node_id=sample_node_id,
        node_type="effect",
        node_version="1.0.0",
        capabilities=ModelNodeCapabilities(
            postgres=True,
            read=True,
            write=True,
        ),
        endpoints={
            "health": "http://localhost:8080/health",
        },
        metadata=ModelNodeMetadata(
            environment="test",
        ),
        health_endpoint="http://localhost:8080/health",
        registered_at=now,
        updated_at=now,
    )


# -----------------------------------------------------------------------------
# Factory Fixtures
# -----------------------------------------------------------------------------


def create_node_registration(
    node_type: Literal["effect", "compute", "reducer", "orchestrator"] = "effect",
    node_id: UUID | None = None,
    node_version: str = "1.0.0",
    registered_at: datetime | None = None,
    updated_at: datetime | None = None,
) -> ModelNodeRegistration:
    """Factory function for creating node registrations.

    Provides flexibility for creating custom registrations in tests
    while maintaining sensible defaults.

    Args:
        node_type: Type of node (default: "effect").
        node_id: Optional node ID (generates if not provided).
        node_version: Semantic version string (default: "1.0.0").
        registered_at: Registration timestamp (default: now).
        updated_at: Last update timestamp (default: now).

    Returns:
        ModelNodeRegistration: Configured registration instance.
    """
    now = datetime.now(UTC)
    return ModelNodeRegistration(
        node_id=node_id or uuid4(),
        node_type=node_type,
        node_version=node_version,
        capabilities=ModelNodeCapabilities(postgres=True, read=True),
        endpoints={"health": "http://localhost:8080/health"},
        metadata=ModelNodeMetadata(environment="test"),
        health_endpoint="http://localhost:8080/health",
        registered_at=registered_at or now,
        updated_at=updated_at or now,
    )


# -----------------------------------------------------------------------------
# Circuit Breaker Test Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def circuit_breaker_config() -> dict[str, object]:
    """Create circuit breaker configuration for testing.

    Returns:
        dict: Configuration with failure threshold and reset timeout.
    """
    return {
        "failure_threshold": 3,
        "reset_timeout_seconds": 10.0,
        "service_name": "test-registry-effect",
    }


# -----------------------------------------------------------------------------
# Retry Test Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def retry_config() -> dict[str, object]:
    """Create retry configuration for testing.

    Returns:
        dict: Configuration with retry limits and backoff settings.
    """
    return {
        "max_attempts": 3,
        "initial_delay_seconds": 0.1,
        "max_delay_seconds": 1.0,
        "exponential_base": 2.0,
    }
