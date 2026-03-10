# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
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
    - StoreEffectIdempotencyInmemory: Primary store under test
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

from omnibase_core.models.primitives.model_semver import ModelSemVer
from omnibase_infra.idempotency import StoreIdempotencyInmemory
from omnibase_infra.models.registration import (
    ModelNodeCapabilities,
    ModelNodeIntrospectionEvent,
    ModelNodeMetadata,
    ModelNodeRegistration,
)
from omnibase_infra.nodes.node_registry_effect.models.model_effect_idempotency_config import (
    ModelEffectIdempotencyConfig,
)
from omnibase_infra.nodes.node_registry_effect.store_effect_idempotency_inmemory import (
    StoreEffectIdempotencyInmemory,
)

# -----------------------------------------------------------------------------
# Idempotency Store Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def default_effect_store() -> StoreEffectIdempotencyInmemory:
    """Create StoreEffectIdempotencyInmemory with default config for testing.

    Returns:
        StoreEffectIdempotencyInmemory with default 10k max size.
    """
    return StoreEffectIdempotencyInmemory()


@pytest.fixture
def small_cache_effect_store() -> StoreEffectIdempotencyInmemory:
    """Create StoreEffectIdempotencyInmemory with small cache for LRU testing.

    Returns:
        StoreEffectIdempotencyInmemory with 100 entry max size.
    """
    config = ModelEffectIdempotencyConfig(
        max_cache_size=100,
        cache_ttl_seconds=3600.0,
    )
    return StoreEffectIdempotencyInmemory(config=config)


@pytest.fixture
def large_cache_effect_store() -> StoreEffectIdempotencyInmemory:
    """Create StoreEffectIdempotencyInmemory with large cache for stress testing.

    Returns:
        StoreEffectIdempotencyInmemory with 100k entry max size.
    """
    config = ModelEffectIdempotencyConfig(
        max_cache_size=100000,
        cache_ttl_seconds=3600.0,
    )
    return StoreEffectIdempotencyInmemory(config=config)


@pytest.fixture
def inmemory_idempotency_store() -> StoreIdempotencyInmemory:
    """Create StoreIdempotencyInmemory for domain-based idempotency testing.

    Returns:
        StoreIdempotencyInmemory for general idempotency tests.
    """
    return StoreIdempotencyInmemory()


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
        node_version=ModelSemVer.parse("1.0.0"),
        correlation_id=correlation_id or uuid4(),
        endpoints={"health": "http://localhost:8080/health"},
        declared_capabilities=ModelNodeCapabilities(
            postgres=True, read=True, write=True
        ),
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
        node_version=ModelSemVer.parse("1.0.0"),
        capabilities=ModelNodeCapabilities(postgres=True, read=True),
        endpoints={"health": "http://localhost:8080/health"},
        metadata=ModelNodeMetadata(environment="test"),
        health_endpoint="http://localhost:8080/health",
        registered_at=now,
        updated_at=now,
    )
