# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Pytest fixtures for NodeRegistryEffect integration tests.

This module provides fixtures that wire NodeRegistryEffect with:
1. Real ModelONEXContainer from omnibase_core
2. Test double backend clients (Consul, PostgreSQL)
3. InMemory idempotency store for isolation

Design Principles:
    - Real container wiring: Uses actual ModelONEXContainer
    - Test doubles, not mocks: Backend clients are controllable implementations
    - Test isolation: Each test gets fresh container and effect instances
    - Async-native: All fixtures support async test patterns
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from omnibase_infra.nodes.effects import NodeRegistryEffect
from omnibase_infra.nodes.effects.models import (
    ModelEffectIdempotencyConfig,
    ModelRegistryRequest,
)
from omnibase_infra.nodes.effects.store_effect_idempotency_inmemory import (
    InMemoryEffectIdempotencyStore,
)

from .test_doubles import StubConsulClient, StubPostgresAdapter


@pytest.fixture
def consul_client() -> StubConsulClient:
    """Create a fresh StubConsulClient.

    Returns:
        StubConsulClient with default (success) configuration.
    """
    return StubConsulClient()


@pytest.fixture
def postgres_adapter() -> StubPostgresAdapter:
    """Create a fresh StubPostgresAdapter.

    Returns:
        StubPostgresAdapter with default (success) configuration.
    """
    return StubPostgresAdapter()


@pytest.fixture
def idempotency_store() -> InMemoryEffectIdempotencyStore:
    """Create a fresh InMemoryEffectIdempotencyStore.

    Returns:
        InMemoryEffectIdempotencyStore with default configuration.
    """
    config = ModelEffectIdempotencyConfig(
        max_cache_size=1000,
        cache_ttl_seconds=3600.0,
    )
    return InMemoryEffectIdempotencyStore(config=config)


@pytest.fixture
def registry_effect(
    consul_client: StubConsulClient,
    postgres_adapter: StubPostgresAdapter,
    idempotency_store: InMemoryEffectIdempotencyStore,
) -> NodeRegistryEffect:
    """Create NodeRegistryEffect with test double backends.

    This fixture wires the effect node with controllable test doubles,
    enabling verification of:
    - Success flows with both backends
    - Partial failure scenarios
    - Retry behavior with idempotency

    Args:
        consul_client: Test double Consul client.
        postgres_adapter: Test double PostgreSQL adapter.
        idempotency_store: In-memory idempotency store.

    Returns:
        NodeRegistryEffect configured with test doubles.
    """
    return NodeRegistryEffect(
        consul_client=consul_client,
        postgres_adapter=postgres_adapter,
        idempotency_store=idempotency_store,
    )


@pytest.fixture
def sample_request() -> ModelRegistryRequest:
    """Create a sample registration request.

    Returns:
        ModelRegistryRequest with valid test data.
    """
    return ModelRegistryRequest(
        node_id=uuid4(),
        node_type="effect",
        node_version="1.0.0",
        correlation_id=uuid4(),
        service_name="onex-effect",
        endpoints={"health": "http://localhost:8080/health"},
        tags=["onex", "effect", "test"],
        metadata={"environment": "test"},
        health_check_config={"HTTP": "http://localhost:8080/health"},
        timestamp=datetime(2025, 1, 1, tzinfo=UTC),
    )


@pytest.fixture
def request_factory() -> Callable[..., ModelRegistryRequest]:
    """Factory for creating unique registration requests.

    Returns a callable that generates unique ModelRegistryRequest instances.
    Useful for tests that need multiple distinct requests.

    Returns:
        Callable that creates unique ModelRegistryRequest instances.

    Example:
        >>> request1 = request_factory()
        >>> request2 = request_factory()
        >>> assert request1.node_id != request2.node_id
    """

    def _create_request(
        node_type: str = "effect",
        node_version: str = "1.0.0",
    ) -> ModelRegistryRequest:
        return ModelRegistryRequest(
            node_id=uuid4(),
            node_type=node_type,
            node_version=node_version,
            correlation_id=uuid4(),
            service_name=f"onex-{node_type}",
            endpoints={"health": "http://localhost:8080/health"},
            tags=["onex", node_type, "test"],
            metadata={"environment": "test"},
            timestamp=datetime(2025, 1, 1, tzinfo=UTC),
        )

    return _create_request
