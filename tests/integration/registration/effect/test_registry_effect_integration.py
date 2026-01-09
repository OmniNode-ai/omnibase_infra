# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Integration tests for NodeRegistryEffect with container wiring.

These tests verify NodeRegistryEffect works correctly with the full dependency
injection container and real async behavior, using test doubles instead of mocks.

Test Scenarios:
    1. Full Success Flow: Both Consul and PostgreSQL succeed
    2. Consul Failure Flow: PostgreSQL succeeds, Consul fails
    3. PostgreSQL Failure Flow: Consul succeeds, PostgreSQL fails
    4. Both Fail Flow: Both backends fail
    5. Retry Success Flow: Retry after partial failure succeeds
    6. Idempotency Verification: Same request returns same result

Design Principles:
    - Uses test doubles implementing protocol interfaces (not mocks)
    - Tests real async behavior with asyncio
    - Verifies state in both the effect and backend test doubles
    - Covers partial failure and retry semantics
    - Tests idempotency guarantees

Related:
    - NodeRegistryEffect: Effect node under test
    - ProtocolConsulClient: Protocol for Consul backend
    - ProtocolPostgresAdapter: Protocol for PostgreSQL backend
    - OMN-954: Effect idempotency and retry behavior
    - PR #78: Integration test requirements
"""

from __future__ import annotations

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

from .test_doubles import StubConsulClient, StubPostgresAdapter


@pytest.mark.integration
class TestFullSuccessFlow:
    """Test Scenario 1: Both Consul and PostgreSQL succeed."""

    @pytest.mark.asyncio
    async def test_full_registration_success(
        self,
        registry_effect: NodeRegistryEffect,
        consul_client: StubConsulClient,
        postgres_adapter: StubPostgresAdapter,
        sample_request: ModelRegistryRequest,
    ) -> None:
        """Test full registration flow with both backends succeeding.

        Verifies:
            1. Response status is "success"
            2. Both backend results show success=True
            3. Backend test doubles recorded the registrations
            4. Processing time is positive
        """
        # Act
        response = await registry_effect.register_node(sample_request)

        # Assert - Response status
        assert response.status == "success"
        assert response.is_complete_success() is True
        assert response.is_partial_failure() is False
        assert response.is_complete_failure() is False

        # Assert - Backend results
        assert response.consul_result.success is True
        assert response.postgres_result.success is True
        assert response.consul_result.error is None
        assert response.postgres_result.error is None

        # Assert - Backend state (test doubles tracked registrations)
        assert len(consul_client.registrations) == 1
        assert len(postgres_adapter.registrations) == 1

        # Verify Consul registration details
        consul_reg = consul_client.registrations[0]
        assert sample_request.node_id is not None
        # Service ID follows ONEX convention: onex-{node_type}-{node_id}
        assert (
            f"onex-{sample_request.node_type}-{sample_request.node_id}"
            == consul_reg.service_id
        )
        assert consul_reg.service_name == sample_request.service_name

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
        consul_client: StubConsulClient,
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
        assert len(consul_client.registrations) == 3
        assert len(postgres_adapter.registrations) == 3

        # Assert - Distinct node IDs
        registered_node_ids = {reg.node_id for reg in postgres_adapter.registrations}
        assert len(registered_node_ids) == 3


@pytest.mark.integration
class TestConsulFailureFlow:
    """Test Scenario 2: PostgreSQL succeeds, Consul fails."""

    @pytest.mark.asyncio
    async def test_consul_failure_partial_result(
        self,
        consul_client: StubConsulClient,
        postgres_adapter: StubPostgresAdapter,
        idempotency_store: InMemoryEffectIdempotencyStore,
        sample_request: ModelRegistryRequest,
    ) -> None:
        """Test partial failure when Consul fails but PostgreSQL succeeds.

        Verifies:
            1. Response status is "partial"
            2. Consul result shows failure, PostgreSQL shows success
            3. Only PostgreSQL registration was recorded
            4. Error summary contains Consul error
        """
        # Arrange - Configure Consul to fail
        consul_client.should_fail = True
        consul_client.failure_error = "Service unavailable"

        effect = NodeRegistryEffect(
            consul_client=consul_client,
            postgres_adapter=postgres_adapter,
            idempotency_store=idempotency_store,
        )

        # Act
        response = await effect.register_node(sample_request)

        # Assert - Partial failure status
        assert response.status == "partial"
        assert response.is_partial_failure() is True
        assert response.is_complete_success() is False

        # Assert - Individual backend results
        assert response.consul_result.success is False
        assert response.postgres_result.success is True

        # Assert - Error captured (sanitized to prevent secret exposure)
        # Raw error "Service unavailable" is sanitized to "service unavailable" safe prefix
        # Backend name is lowercase as passed to sanitize_backend_error()
        assert response.consul_result.error is not None
        assert "consul operation failed" in response.consul_result.error
        assert "service unavailable" in response.consul_result.error
        assert response.error_summary is not None
        assert "consul" in response.error_summary.lower()

        # Assert - Only PostgreSQL registration recorded
        assert len(consul_client.registrations) == 0
        assert len(postgres_adapter.registrations) == 1

        # Assert - Failed backends helper
        assert response.get_failed_backends() == ["consul"]
        assert response.get_successful_backends() == ["postgres"]

    @pytest.mark.asyncio
    async def test_consul_exception_handled(
        self,
        consul_client: StubConsulClient,
        postgres_adapter: StubPostgresAdapter,
        idempotency_store: InMemoryEffectIdempotencyStore,
        sample_request: ModelRegistryRequest,
    ) -> None:
        """Test that Consul exceptions are handled gracefully.

        Verifies exception handling doesn't crash the effect and produces
        a proper partial failure response.
        """
        # Arrange - Configure Consul to raise exception
        consul_client.set_exception(ConnectionError("Connection refused"))

        effect = NodeRegistryEffect(
            consul_client=consul_client,
            postgres_adapter=postgres_adapter,
            idempotency_store=idempotency_store,
        )

        # Act
        response = await effect.register_node(sample_request)

        # Assert - Partial failure with sanitized error
        # Note: Standard Python ConnectionError falls into the generic Exception
        # handler because only InfraConnectionError maps to CONSUL_CONNECTION_ERROR
        assert response.status == "partial"
        assert response.consul_result.success is False
        assert response.consul_result.error is not None
        assert "ConnectionError" in response.consul_result.error
        assert response.consul_result.error_code == "CONSUL_UNKNOWN_ERROR"

        # Assert - PostgreSQL still succeeded
        assert response.postgres_result.success is True
        assert len(postgres_adapter.registrations) == 1


@pytest.mark.integration
class TestPostgresFailureFlow:
    """Test Scenario 3: Consul succeeds, PostgreSQL fails."""

    @pytest.mark.asyncio
    async def test_postgres_failure_partial_result(
        self,
        consul_client: StubConsulClient,
        postgres_adapter: StubPostgresAdapter,
        idempotency_store: InMemoryEffectIdempotencyStore,
        sample_request: ModelRegistryRequest,
    ) -> None:
        """Test partial failure when PostgreSQL fails but Consul succeeds.

        Verifies:
            1. Response status is "partial"
            2. PostgreSQL result shows failure, Consul shows success
            3. Only Consul registration was recorded
            4. Error summary contains PostgreSQL error
        """
        # Arrange - Configure PostgreSQL to fail
        postgres_adapter.should_fail = True
        postgres_adapter.failure_error = "Connection timeout"

        effect = NodeRegistryEffect(
            consul_client=consul_client,
            postgres_adapter=postgres_adapter,
            idempotency_store=idempotency_store,
        )

        # Act
        response = await effect.register_node(sample_request)

        # Assert - Partial failure status
        assert response.status == "partial"
        assert response.is_partial_failure() is True

        # Assert - Individual backend results
        assert response.consul_result.success is True
        assert response.postgres_result.success is False

        # Assert - Error captured (sanitized to prevent secret exposure)
        # Raw error "Connection timeout" is sanitized to "timeout" safe prefix
        # Backend name is lowercase as passed to sanitize_backend_error()
        assert response.postgres_result.error is not None
        assert "postgres operation failed" in response.postgres_result.error
        assert "timeout" in response.postgres_result.error
        assert response.error_summary is not None
        assert "postgres" in response.error_summary.lower()

        # Assert - Only Consul registration recorded
        assert len(consul_client.registrations) == 1
        assert len(postgres_adapter.registrations) == 0

        # Assert - Failed backends helper
        assert response.get_failed_backends() == ["postgres"]
        assert response.get_successful_backends() == ["consul"]

    @pytest.mark.asyncio
    async def test_postgres_exception_handled(
        self,
        consul_client: StubConsulClient,
        postgres_adapter: StubPostgresAdapter,
        idempotency_store: InMemoryEffectIdempotencyStore,
        sample_request: ModelRegistryRequest,
    ) -> None:
        """Test that PostgreSQL exceptions are handled gracefully."""
        # Arrange - Configure PostgreSQL to raise exception
        postgres_adapter.set_exception(TimeoutError("Query timeout"))

        effect = NodeRegistryEffect(
            consul_client=consul_client,
            postgres_adapter=postgres_adapter,
            idempotency_store=idempotency_store,
        )

        # Act
        response = await effect.register_node(sample_request)

        # Assert - Partial failure with sanitized error
        # Note: TimeoutError is correctly caught and mapped to POSTGRES_TIMEOUT_ERROR
        assert response.status == "partial"
        assert response.postgres_result.success is False
        assert response.postgres_result.error is not None
        assert "TimeoutError" in response.postgres_result.error
        assert response.postgres_result.error_code == "POSTGRES_TIMEOUT_ERROR"

        # Assert - Consul still succeeded
        assert response.consul_result.success is True
        assert len(consul_client.registrations) == 1


@pytest.mark.integration
class TestBothFailFlow:
    """Test Scenario 4: Both backends fail."""

    @pytest.mark.asyncio
    async def test_both_backends_fail(
        self,
        consul_client: StubConsulClient,
        postgres_adapter: StubPostgresAdapter,
        idempotency_store: InMemoryEffectIdempotencyStore,
        sample_request: ModelRegistryRequest,
    ) -> None:
        """Test complete failure when both backends fail.

        Verifies:
            1. Response status is "failed"
            2. Both backend results show failure
            3. No registrations recorded
            4. Error summary contains both errors
        """
        # Arrange - Configure both to fail
        consul_client.should_fail = True
        consul_client.failure_error = "Consul unavailable"
        postgres_adapter.should_fail = True
        postgres_adapter.failure_error = "Database down"

        effect = NodeRegistryEffect(
            consul_client=consul_client,
            postgres_adapter=postgres_adapter,
            idempotency_store=idempotency_store,
        )

        # Act
        response = await effect.register_node(sample_request)

        # Assert - Complete failure status
        assert response.status == "failed"
        assert response.is_complete_failure() is True
        assert response.is_partial_failure() is False
        assert response.is_complete_success() is False

        # Assert - Both backend results show failure
        assert response.consul_result.success is False
        assert response.postgres_result.success is False

        # Assert - Errors captured (sanitized to prevent secret exposure)
        # Raw errors without safe prefix patterns are sanitized to generic message
        # Backend names are lowercase as passed to sanitize_backend_error()
        assert response.consul_result.error is not None
        assert response.postgres_result.error is not None
        assert "consul operation failed" in response.consul_result.error
        assert "postgres operation failed" in response.postgres_result.error

        # Assert - Error summary contains both
        assert response.error_summary is not None
        assert "consul" in response.error_summary.lower()
        assert "postgres" in response.error_summary.lower()

        # Assert - No registrations recorded
        assert len(consul_client.registrations) == 0
        assert len(postgres_adapter.registrations) == 0

        # Assert - Both in failed backends
        assert set(response.get_failed_backends()) == {"consul", "postgres"}
        assert response.get_successful_backends() == []

    @pytest.mark.asyncio
    async def test_both_backends_raise_exceptions(
        self,
        consul_client: StubConsulClient,
        postgres_adapter: StubPostgresAdapter,
        idempotency_store: InMemoryEffectIdempotencyStore,
        sample_request: ModelRegistryRequest,
    ) -> None:
        """Test complete failure when both backends raise exceptions."""
        # Arrange - Configure both to raise exceptions
        consul_client.set_exception(ConnectionError("Network unreachable"))
        postgres_adapter.set_exception(RuntimeError("Connection pool exhausted"))

        effect = NodeRegistryEffect(
            consul_client=consul_client,
            postgres_adapter=postgres_adapter,
            idempotency_store=idempotency_store,
        )

        # Act
        response = await effect.register_node(sample_request)

        # Assert - Complete failure
        assert response.status == "failed"
        assert response.consul_result.success is False
        assert response.postgres_result.success is False

        # Assert - Sanitized errors (no raw exception messages)
        assert response.consul_result.error is not None
        assert response.postgres_result.error is not None
        assert "ConnectionError" in response.consul_result.error
        assert "RuntimeError" in response.postgres_result.error


@pytest.mark.integration
class TestRetrySuccessFlow:
    """Test Scenario 5: Retry after partial failure succeeds."""

    @pytest.mark.asyncio
    async def test_retry_after_consul_failure_succeeds(
        self,
        consul_client: StubConsulClient,
        postgres_adapter: StubPostgresAdapter,
        idempotency_store: InMemoryEffectIdempotencyStore,
        sample_request: ModelRegistryRequest,
    ) -> None:
        """Test successful retry after Consul failure.

        Verifies:
            1. First attempt results in partial failure (Consul fails)
            2. Retry with recovered Consul succeeds
            3. PostgreSQL is NOT called again (idempotency)
            4. Final response shows full success
        """
        # Arrange - Configure Consul to fail initially
        consul_client.should_fail = True
        consul_client.failure_error = "Temporary unavailable"

        effect = NodeRegistryEffect(
            consul_client=consul_client,
            postgres_adapter=postgres_adapter,
            idempotency_store=idempotency_store,
        )

        # Act - First attempt (partial failure expected)
        response1 = await effect.register_node(sample_request)

        # Assert - Partial failure
        assert response1.status == "partial"
        assert response1.consul_result.success is False
        assert response1.postgres_result.success is True

        # Record call counts before retry
        consul_calls_before = consul_client.call_count
        postgres_calls_before = postgres_adapter.call_count

        # Arrange - Fix Consul for retry
        consul_client.should_fail = False

        # Act - Retry with same correlation_id (idempotency key)
        response2 = await effect.register_node(sample_request)

        # Assert - Full success on retry
        assert response2.status == "success"
        assert response2.consul_result.success is True
        assert response2.postgres_result.success is True

        # Assert - Consul was called again (retry)
        assert consul_client.call_count > consul_calls_before

        # Assert - PostgreSQL was NOT called again (idempotency)
        # The idempotency store tracked postgres as completed
        assert postgres_adapter.call_count == postgres_calls_before

        # Assert - Only one PostgreSQL registration
        assert len(postgres_adapter.registrations) == 1

    @pytest.mark.asyncio
    async def test_retry_after_postgres_failure_succeeds(
        self,
        consul_client: StubConsulClient,
        postgres_adapter: StubPostgresAdapter,
        idempotency_store: InMemoryEffectIdempotencyStore,
        sample_request: ModelRegistryRequest,
    ) -> None:
        """Test successful retry after PostgreSQL failure.

        Verifies:
            1. First attempt results in partial failure (PostgreSQL fails)
            2. Retry with recovered PostgreSQL succeeds
            3. Consul is NOT called again (idempotency)
            4. Final response shows full success
        """
        # Arrange - Configure PostgreSQL to fail initially
        postgres_adapter.should_fail = True
        postgres_adapter.failure_error = "Temporary unavailable"

        effect = NodeRegistryEffect(
            consul_client=consul_client,
            postgres_adapter=postgres_adapter,
            idempotency_store=idempotency_store,
        )

        # Act - First attempt (partial failure expected)
        response1 = await effect.register_node(sample_request)

        # Assert - Partial failure
        assert response1.status == "partial"
        assert response1.consul_result.success is True
        assert response1.postgres_result.success is False

        # Record call counts before retry
        consul_calls_before = consul_client.call_count
        postgres_calls_before = postgres_adapter.call_count

        # Arrange - Fix PostgreSQL for retry
        postgres_adapter.should_fail = False

        # Act - Retry with same correlation_id
        response2 = await effect.register_node(sample_request)

        # Assert - Full success on retry
        assert response2.status == "success"
        assert response2.consul_result.success is True
        assert response2.postgres_result.success is True

        # Assert - PostgreSQL was called again (retry)
        assert postgres_adapter.call_count > postgres_calls_before

        # Assert - Consul was NOT called again (idempotency)
        assert consul_client.call_count == consul_calls_before

        # Assert - Only one Consul registration
        assert len(consul_client.registrations) == 1


@pytest.mark.integration
class TestIdempotencyVerification:
    """Test Scenario 6: Same request returns same result (idempotency)."""

    @pytest.mark.asyncio
    async def test_duplicate_request_same_result(
        self,
        registry_effect: NodeRegistryEffect,
        consul_client: StubConsulClient,
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
        consul_calls_after_first = consul_client.call_count
        postgres_calls_after_first = postgres_adapter.call_count

        # Act - Duplicate request (same correlation_id)
        response2 = await registry_effect.register_node(sample_request)

        # Assert - Both succeeded
        assert response1.status == "success"
        assert response2.status == "success"

        # Assert - Same correlation ID in responses
        assert response1.correlation_id == response2.correlation_id

        # Assert - Backends NOT called again
        assert consul_client.call_count == consul_calls_after_first
        assert postgres_adapter.call_count == postgres_calls_after_first

        # Assert - Only one registration per backend
        assert len(consul_client.registrations) == 1
        assert len(postgres_adapter.registrations) == 1

    @pytest.mark.asyncio
    async def test_different_correlation_ids_independent(
        self,
        registry_effect: NodeRegistryEffect,
        consul_client: StubConsulClient,
        postgres_adapter: StubPostgresAdapter,
        request_factory: Callable[..., ModelRegistryRequest],
    ) -> None:
        """Test that different correlation IDs are processed independently.

        Verifies that idempotency is keyed by correlation_id, not node_id.
        """
        # Suppress unused fixture warning - fixture is available for debugging
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
        assert consul_client.call_count == 2
        assert postgres_adapter.call_count == 2

    @pytest.mark.asyncio
    async def test_completed_backends_tracked_correctly(
        self,
        registry_effect: NodeRegistryEffect,
        consul_client: StubConsulClient,
        postgres_adapter: StubPostgresAdapter,
        sample_request: ModelRegistryRequest,
    ) -> None:
        """Test that completed backends are tracked in idempotency store.

        Verifies the internal state of the idempotency store.
        """
        # Act - Complete registration
        await registry_effect.register_node(sample_request)

        # Assert - Check completed backends via effect method
        completed = await registry_effect.get_completed_backends(
            sample_request.correlation_id
        )
        assert "consul" in completed
        assert "postgres" in completed

    @pytest.mark.asyncio
    async def test_clear_completed_backends_allows_reprocessing(
        self,
        registry_effect: NodeRegistryEffect,
        consul_client: StubConsulClient,
        postgres_adapter: StubPostgresAdapter,
        sample_request: ModelRegistryRequest,
    ) -> None:
        """Test that clearing completed backends allows reprocessing.

        Verifies the clear_completed_backends method enables force re-registration.
        """
        # Act - First registration
        await registry_effect.register_node(sample_request)
        first_consul_count = consul_client.call_count
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

        # Assert - Backends called again
        assert consul_client.call_count > first_consul_count
        assert postgres_adapter.call_count > first_postgres_count


@pytest.mark.integration
class TestAsyncBehavior:
    """Additional tests for async behavior patterns."""

    @pytest.mark.asyncio
    async def test_concurrent_registrations_isolated(
        self,
        consul_client: StubConsulClient,
        postgres_adapter: StubPostgresAdapter,
        idempotency_store: InMemoryEffectIdempotencyStore,
        request_factory: Callable[..., ModelRegistryRequest],
    ) -> None:
        """Test that concurrent registrations are properly isolated.

        Verifies that concurrent requests with different correlation IDs
        don't interfere with each other.
        """
        import asyncio

        effect = NodeRegistryEffect(
            consul_client=consul_client,
            postgres_adapter=postgres_adapter,
            idempotency_store=idempotency_store,
        )

        # Create multiple unique requests
        requests = [request_factory() for _ in range(5)]

        # Act - Execute concurrently
        responses = await asyncio.gather(
            *[effect.register_node(req) for req in requests]
        )

        # Assert - All succeeded
        assert all(r.status == "success" for r in responses)

        # Assert - All registrations recorded
        assert len(consul_client.registrations) == 5
        assert len(postgres_adapter.registrations) == 5

        # Assert - All have unique node IDs
        node_ids = {reg.node_id for reg in postgres_adapter.registrations}
        assert len(node_ids) == 5

    @pytest.mark.asyncio
    async def test_simulated_network_latency(
        self,
        consul_client: StubConsulClient,
        postgres_adapter: StubPostgresAdapter,
        idempotency_store: InMemoryEffectIdempotencyStore,
        sample_request: ModelRegistryRequest,
    ) -> None:
        """Test behavior with simulated network latency.

        Verifies the effect handles delays correctly.
        """
        # Configure delays
        consul_client.delay_seconds = 0.01  # 10ms
        postgres_adapter.delay_seconds = 0.01  # 10ms

        effect = NodeRegistryEffect(
            consul_client=consul_client,
            postgres_adapter=postgres_adapter,
            idempotency_store=idempotency_store,
        )

        # Act
        response = await effect.register_node(sample_request)

        # Assert - Success despite delays
        assert response.status == "success"

        # Assert - Processing time reflects delays
        assert response.processing_time_ms >= 20.0  # At least 20ms total
