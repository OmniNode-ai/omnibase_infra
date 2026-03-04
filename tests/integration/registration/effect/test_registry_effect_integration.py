# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Integration tests for NodeRegistryEffect with container wiring.

These tests verify NodeRegistryEffect works correctly with the full dependency
injection container and real async behavior, using test doubles instead of mocks.

Test Scenarios:
    1. Full Success Flow: PostgreSQL succeeds
    2. PostgreSQL Failure Flow: PostgreSQL fails
    3. Idempotency Verification: Same request returns same result
    4. Async Behavior: Concurrent registration isolation

Design Principles:
    - Uses test doubles implementing protocol interfaces (not mocks)
    - Tests real async behavior with asyncio
    - Verifies state in the effect and backend test doubles
    - Covers failure and retry semantics
    - Tests idempotency guarantees

Related:
    - NodeRegistryEffect: Effect node under test
    - ProtocolPostgresAdapter: Protocol for PostgreSQL backend
    - OMN-954: Effect idempotency and retry behavior
    - OMN-3540: Consul removed entirely
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from omnibase_core.models.primitives.model_semver import ModelSemVer
from omnibase_infra.nodes.effects import NodeRegistryEffect
from omnibase_infra.nodes.effects.models import ModelRegistryRequest
from omnibase_infra.nodes.effects.store_effect_idempotency_inmemory import (
    InMemoryEffectIdempotencyStore,
)

from .test_doubles import StubPostgresAdapter


@pytest.mark.integration
class TestFullSuccessFlow:
    """Test Scenario 1: PostgreSQL succeeds."""

    @pytest.mark.asyncio
    async def test_full_registration_success(
        self,
        registry_effect: NodeRegistryEffect,
        postgres_adapter: StubPostgresAdapter,
        sample_request: ModelRegistryRequest,
    ) -> None:
        """Test full registration flow with postgres backend succeeding.

        Verifies:
            1. Response status is "success"
            2. Postgres backend result shows success=True
            3. Backend test double recorded the registration
            4. Processing time is positive
        """
        # Act
        response = await registry_effect.register_node(sample_request)

        # Assert - Response status
        assert response.status == "success"
        assert response.is_complete_success() is True
        assert response.is_complete_failure() is False

        # Assert - Backend results
        assert response.postgres_result.success is True
        assert response.postgres_result.error is None

        # Assert - Backend state (test double tracked registrations)
        assert len(postgres_adapter.registrations) == 1

        # Verify PostgreSQL registration details
        pg_reg = postgres_adapter.registrations[0]
        assert pg_reg.node_id == sample_request.node_id
        assert pg_reg.node_type == sample_request.node_type
        assert pg_reg.node_version == sample_request.node_version

        # Assert - Processing time
        assert response.processing_time_ms > 0

        # Assert - Correlation ID propagation
        assert response.correlation_id == sample_request.correlation_id

    @pytest.mark.asyncio
    async def test_multiple_successful_registrations(
        self,
        registry_effect: NodeRegistryEffect,
        postgres_adapter: StubPostgresAdapter,
        request_factory: Callable[..., ModelRegistryRequest],
    ) -> None:
        """Test multiple independent registrations all succeed.

        Verifies that multiple unique requests are processed independently.
        """
        # Arrange - Create multiple unique requests
        request1 = request_factory(node_type="effect")
        request2 = request_factory(node_type="compute")
        request3 = request_factory(node_type="reducer")

        # Act
        response1 = await registry_effect.register_node(request1)
        response2 = await registry_effect.register_node(request2)
        response3 = await registry_effect.register_node(request3)

        # Assert - All succeeded
        assert response1.status == "success"
        assert response2.status == "success"
        assert response3.status == "success"

        # Assert - All registrations recorded
        assert len(postgres_adapter.registrations) == 3

        # Assert - Distinct node IDs
        registered_node_ids = {reg.node_id for reg in postgres_adapter.registrations}
        assert len(registered_node_ids) == 3


@pytest.mark.integration
class TestPostgresFailureFlow:
    """Test Scenario 2: PostgreSQL fails."""

    @pytest.mark.asyncio
    async def test_postgres_failure_result(
        self,
        postgres_adapter: StubPostgresAdapter,
        idempotency_store: InMemoryEffectIdempotencyStore,
        sample_request: ModelRegistryRequest,
    ) -> None:
        """Test failure when PostgreSQL fails.

        Verifies:
            1. Response status is "failed"
            2. Postgres result shows failure
            3. Error captured in response
        """
        # Arrange - Configure postgres to fail
        postgres_adapter.should_fail = True
        postgres_adapter.failure_error = "Connection refused"

        effect = NodeRegistryEffect(
            postgres_adapter=postgres_adapter,
            idempotency_store=idempotency_store,
        )

        # Act
        response = await effect.register_node(sample_request)

        # Assert - Complete failure status
        assert response.status == "failed"
        assert response.is_complete_failure() is True
        assert response.is_complete_success() is False

        # Assert - Individual backend results
        assert response.postgres_result.success is False
        assert response.postgres_result.error is not None

        # Assert - Error summary populated
        assert response.error_summary is not None

        # Assert - No registrations recorded
        assert len(postgres_adapter.registrations) == 0

    @pytest.mark.asyncio
    async def test_postgres_exception_handled(
        self,
        postgres_adapter: StubPostgresAdapter,
        idempotency_store: InMemoryEffectIdempotencyStore,
        sample_request: ModelRegistryRequest,
    ) -> None:
        """Test that postgres exceptions are handled gracefully.

        Verifies exception handling doesn't crash the effect.
        """
        # Arrange - Configure postgres to raise exception
        postgres_adapter.set_exception(ConnectionError("Connection refused"))

        effect = NodeRegistryEffect(
            postgres_adapter=postgres_adapter,
            idempotency_store=idempotency_store,
        )

        # Act
        response = await effect.register_node(sample_request)

        # Assert - Failure with error captured
        assert response.status == "failed"
        assert response.postgres_result.success is False
        assert response.postgres_result.error is not None


@pytest.mark.integration
class TestIdempotencyVerification:
    """Test Scenario 3: Same request returns same result (idempotency)."""

    @pytest.mark.asyncio
    async def test_duplicate_request_same_result(
        self,
        registry_effect: NodeRegistryEffect,
        postgres_adapter: StubPostgresAdapter,
        sample_request: ModelRegistryRequest,
    ) -> None:
        """Test that duplicate requests return consistent results.

        Verifies:
            1. First request succeeds and records registrations
            2. Second identical request succeeds immediately
            3. Backends are NOT called again (idempotency)
            4. Same result returned
        """
        # Act - First request
        response1 = await registry_effect.register_node(sample_request)

        # Record state after first request
        postgres_calls_after_first = postgres_adapter.call_count

        # Act - Duplicate request (same correlation_id)
        response2 = await registry_effect.register_node(sample_request)

        # Assert - Both succeeded
        assert response1.status == "success"
        assert response2.status == "success"

        # Assert - Same correlation ID in responses
        assert response1.correlation_id == response2.correlation_id

        # Assert - Backend NOT called again (idempotency)
        assert postgres_adapter.call_count == postgres_calls_after_first

        # Assert - Only one registration per backend
        assert len(postgres_adapter.registrations) == 1

    @pytest.mark.asyncio
    async def test_different_correlation_ids_independent(
        self,
        registry_effect: NodeRegistryEffect,
        postgres_adapter: StubPostgresAdapter,
        request_factory: Callable[..., ModelRegistryRequest],
    ) -> None:
        """Test that different correlation IDs are processed independently.

        Verifies that idempotency is keyed by correlation_id, not node_id.
        """
        _ = request_factory

        # Arrange - Same node_id but different correlation_ids
        base_node_id = uuid4()
        request1 = ModelRegistryRequest(
            node_id=base_node_id,
            node_type="effect",
            node_version=ModelSemVer.parse("1.0.0"),
            correlation_id=uuid4(),  # Different correlation_id
            service_name="onex-effect",
            endpoints={"health": "http://localhost:8080/health"},
            tags=["onex"],
            metadata={},
            timestamp=datetime(2025, 1, 1, tzinfo=UTC),
        )
        request2 = ModelRegistryRequest(
            node_id=base_node_id,  # Same node_id
            node_type="effect",
            node_version=ModelSemVer.parse("1.0.0"),
            correlation_id=uuid4(),  # Different correlation_id
            service_name="onex-effect",
            endpoints={"health": "http://localhost:8080/health"},
            tags=["onex"],
            metadata={},
            timestamp=datetime(2025, 1, 1, tzinfo=UTC),
        )

        # Act
        response1 = await registry_effect.register_node(request1)
        response2 = await registry_effect.register_node(request2)

        # Assert - Both processed independently
        assert response1.status == "success"
        assert response2.status == "success"

        # Assert - Both backend calls made (no idempotency cross-talk)
        assert postgres_adapter.call_count == 2

    @pytest.mark.asyncio
    async def test_completed_backends_tracked_correctly(
        self,
        registry_effect: NodeRegistryEffect,
        postgres_adapter: StubPostgresAdapter,
        sample_request: ModelRegistryRequest,
    ) -> None:
        """Test that completed backends are tracked in idempotency store.

        Verifies the internal state of the idempotency store.
        """
        _ = postgres_adapter

        # Act - Complete registration
        await registry_effect.register_node(sample_request)

        # Assert - Check completed backends via effect method
        completed = await registry_effect.get_completed_backends(
            sample_request.correlation_id
        )
        assert "postgres" in completed

    @pytest.mark.asyncio
    async def test_clear_completed_backends_allows_reprocessing(
        self,
        registry_effect: NodeRegistryEffect,
        postgres_adapter: StubPostgresAdapter,
        sample_request: ModelRegistryRequest,
    ) -> None:
        """Test that clearing completed backends allows reprocessing.

        Verifies the clear_completed_backends method enables force re-registration.
        """
        # Act - First registration
        await registry_effect.register_node(sample_request)
        first_postgres_count = postgres_adapter.call_count

        # Clear completed backends
        await registry_effect.clear_completed_backends(sample_request.correlation_id)

        # Verify cleared
        completed = await registry_effect.get_completed_backends(
            sample_request.correlation_id
        )
        assert len(completed) == 0

        # Act - Re-register (should call backends again)
        await registry_effect.register_node(sample_request)

        # Assert - Backend called again
        assert postgres_adapter.call_count > first_postgres_count


@pytest.mark.integration
class TestAsyncBehavior:
    """Additional tests for async behavior patterns."""

    @pytest.mark.asyncio
    async def test_concurrent_registrations_isolated(
        self,
        postgres_adapter: StubPostgresAdapter,
        idempotency_store: InMemoryEffectIdempotencyStore,
        request_factory: Callable[..., ModelRegistryRequest],
    ) -> None:
        """Test that concurrent registrations don't interfere.

        Verifies isolation between concurrent registration attempts.
        """
        effect = NodeRegistryEffect(
            postgres_adapter=postgres_adapter,
            idempotency_store=idempotency_store,
        )

        requests = [request_factory() for _ in range(3)]

        # Act - Register concurrently
        results = await asyncio.gather(*[effect.register_node(r) for r in requests])

        # Assert - All succeeded
        for result in results:
            assert result.status == "success"

        # Assert - All registered
        assert len(postgres_adapter.registrations) == 3
