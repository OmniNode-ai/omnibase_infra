# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Shared pytest fixtures for registration effect performance tests.

Provides fixtures for performance testing including:
- Pre-configured idempotency stores with various cache sizes
- Mock clients with minimal latency for high-volume testing
- Simulated effect executors for load testing
- Latency measurement utilities

Usage:
    Fixtures are automatically available to all tests in this package.

Related:
    - OMN-954: Effect idempotency testing requirements
    - InMemoryEffectIdempotencyStore: Primary store under test
"""

from __future__ import annotations

from datetime import UTC, datetime
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
from omnibase_infra.nodes.effects.models.model_effect_idempotency_config import (
    ModelEffectIdempotencyConfig,
)
from omnibase_infra.nodes.effects.store_effect_idempotency_inmemory import (
    InMemoryEffectIdempotencyStore,
)

# -----------------------------------------------------------------------------
# Module-Level Markers
# -----------------------------------------------------------------------------

pytestmark = [
    pytest.mark.performance,
]

# -----------------------------------------------------------------------------
# Idempotency Store Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def default_effect_store() -> InMemoryEffectIdempotencyStore:
    """Create InMemoryEffectIdempotencyStore with default config for testing.

    Returns:
        InMemoryEffectIdempotencyStore with default 10k max size.
    """
    return InMemoryEffectIdempotencyStore()


@pytest.fixture
def small_cache_effect_store() -> InMemoryEffectIdempotencyStore:
    """Create InMemoryEffectIdempotencyStore with small cache for LRU testing.

    Returns:
        InMemoryEffectIdempotencyStore with 100 entry max size.
    """
    config = ModelEffectIdempotencyConfig(
        max_cache_size=100,
        cache_ttl_seconds=3600.0,
    )
    return InMemoryEffectIdempotencyStore(config=config)


@pytest.fixture
def large_cache_effect_store() -> InMemoryEffectIdempotencyStore:
    """Create InMemoryEffectIdempotencyStore with large cache for stress testing.

    Returns:
        InMemoryEffectIdempotencyStore with 100k entry max size.
    """
    config = ModelEffectIdempotencyConfig(
        max_cache_size=100000,
        cache_ttl_seconds=3600.0,
    )
    return InMemoryEffectIdempotencyStore(config=config)


@pytest.fixture
def inmemory_idempotency_store() -> InMemoryIdempotencyStore:
    """Create InMemoryIdempotencyStore for domain-based idempotency testing.

    Returns:
        InMemoryIdempotencyStore for general idempotency tests.
    """
    return InMemoryIdempotencyStore()


# -----------------------------------------------------------------------------
# Mock Infrastructure Client Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def fast_mock_consul_client() -> MagicMock:
    """Create mock Consul client with immediate returns for perf testing.

    Returns:
        MagicMock configured for zero-latency Consul operations.
    """
    client = MagicMock()
    client.agent = MagicMock()
    client.agent.service = MagicMock()
    client.agent.service.register = AsyncMock(return_value=True)
    client.agent.service.deregister = AsyncMock(return_value=True)
    return client


@pytest.fixture
def fast_mock_postgres_client() -> MagicMock:
    """Create mock PostgreSQL client with immediate returns for perf testing.

    Returns:
        MagicMock configured for zero-latency PostgreSQL operations.
    """
    client = MagicMock()
    client.execute = AsyncMock(return_value=None)
    client.fetchone = AsyncMock(return_value=None)
    return client


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


def create_sample_introspection_event(
    node_id: UUID | None = None,
    correlation_id: UUID | None = None,
) -> ModelNodeIntrospectionEvent:
    """Factory function for creating introspection events.

    Args:
        node_id: Optional node ID (generates if not provided).
        correlation_id: Optional correlation ID (generates if not provided).

    Returns:
        ModelNodeIntrospectionEvent configured for testing.
    """
    return ModelNodeIntrospectionEvent(
        node_id=node_id or uuid4(),
        node_type="effect",
        node_version="1.0.0",
        correlation_id=correlation_id or uuid4(),
        endpoints={"health": "http://localhost:8080/health"},
        capabilities=ModelNodeCapabilities(postgres=True, read=True, write=True),
        metadata=ModelNodeMetadata(environment="test"),
        timestamp=datetime.now(UTC),  # Required: time injection pattern
    )


def create_sample_registration(
    node_id: UUID | None = None,
) -> ModelNodeRegistration:
    """Factory function for creating node registrations.

    Args:
        node_id: Optional node ID (generates if not provided).

    Returns:
        ModelNodeRegistration configured for testing.
    """
    now = datetime.now(UTC)
    return ModelNodeRegistration(
        node_id=node_id or uuid4(),
        node_type="effect",
        node_version="1.0.0",
        capabilities=ModelNodeCapabilities(postgres=True, read=True),
        endpoints={"health": "http://localhost:8080/health"},
        metadata=ModelNodeMetadata(environment="test"),
        health_endpoint="http://localhost:8080/health",
        registered_at=now,
        updated_at=now,
    )
