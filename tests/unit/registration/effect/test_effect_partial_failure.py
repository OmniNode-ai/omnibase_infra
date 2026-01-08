# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for Registry Effect partial failure scenarios.

This test suite validates the partial failure handling of the NodeRegistryEffect node,
which operates on TWO backends (Consul + PostgreSQL) and must handle scenarios
where one backend succeeds while the other fails.

Test Coverage (G4 Acceptance Criteria):
    1. test_consul_success_postgres_failure - Consul succeeds, PostgreSQL fails
    2. test_consul_failure_postgres_success - Consul fails, PostgreSQL succeeds
    3. test_both_backends_fail - Both backends fail
    4. test_partial_failure_idempotency - Retry only failed backend
    5. test_partial_failure_error_aggregation - Error context preservation
    6. test_partial_failure_processing_time - Timing reflects actual duration

Response Status Semantics:
    - "success": Both backends succeeded
    - "partial": Exactly one backend succeeded
    - "failed": Both backends failed

Related:
    - NodeRegistryEffect: Effect node under test
    - ModelRegistryResponse: Response model with partial failure support
    - ModelBackendResult: Individual backend result model
    - OMN-954: Partial failure scenario testing ticket
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest

from omnibase_infra.nodes.effects import (
    ModelBackendResult,
    ModelRegistryRequest,
    ModelRegistryResponse,
    NodeRegistryEffect,
)

# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def mock_consul_client() -> AsyncMock:
    """Create a mock Consul client for testing.

    Returns:
        AsyncMock implementing ProtocolConsulClient interface.
    """
    mock = AsyncMock()
    mock.register_service = AsyncMock(return_value=ModelBackendResult(success=True))
    return mock


@pytest.fixture
def mock_postgres_handler() -> AsyncMock:
    """Create a mock PostgreSQL handler for testing.

    Returns:
        AsyncMock implementing ProtocolPostgresAdapter interface
        (adapter protocol for database operations).
    """
    mock = AsyncMock()
    mock.upsert = AsyncMock(return_value=ModelBackendResult(success=True))
    return mock


@pytest.fixture
def registry_effect(
    mock_consul_client: AsyncMock,
    mock_postgres_handler: AsyncMock,
) -> NodeRegistryEffect:
    """Create a NodeRegistryEffect with mock backends.

    Args:
        mock_consul_client: Mock Consul client.
        mock_postgres_handler: Mock PostgreSQL handler.

    Returns:
        NodeRegistryEffect instance with mocked backends.
    """
    return NodeRegistryEffect(mock_consul_client, mock_postgres_handler)


@pytest.fixture
def sample_registry_request() -> ModelRegistryRequest:
    """Create a sample registry request for testing.

    Returns:
        ModelRegistryRequest with valid test data.
    """
    return ModelRegistryRequest(
        node_id=uuid4(),
        node_type="effect",
        node_version="1.0.0",
        correlation_id=uuid4(),
        service_name="test-service",
        endpoints={"health": "http://localhost:8080/health"},
        tags=["test", "effect"],
        metadata={"environment": "test"},
        timestamp=datetime(2025, 1, 1, tzinfo=UTC),
    )


@pytest.fixture
def correlation_id() -> UUID:
    """Create a correlation ID for testing.

    Returns:
        UUID for request correlation.
    """
    return uuid4()


# -----------------------------------------------------------------------------
# Test Class: Partial Failure Scenarios
# -----------------------------------------------------------------------------


@pytest.mark.unit
class TestEffectPartialFailure:
    """Test suite for partial failure scenarios (G4 acceptance criteria).

    These tests validate that the NodeRegistryEffect correctly handles scenarios
    where one backend succeeds and the other fails, preserving appropriate
    context and enabling targeted retries.
    """

    @pytest.mark.asyncio
    async def test_consul_success_postgres_failure(
        self,
        registry_effect: NodeRegistryEffect,
        mock_consul_client: AsyncMock,
        mock_postgres_handler: AsyncMock,
        sample_registry_request: ModelRegistryRequest,
    ) -> None:
        """Test partial failure when Consul succeeds but PostgreSQL fails.

        Scenario:
            - Consul registration succeeds
            - PostgreSQL upsert fails with connection error

        Expected:
            - response.status == "partial"
            - response.consul_result.success == True
            - response.postgres_result.success == False
            - correlation_id is preserved in both results
        """
        # Arrange
        mock_consul_client.register_service.return_value = ModelBackendResult(
            success=True
        )
        mock_postgres_handler.upsert.side_effect = Exception("DB connection failed")

        # Act
        response = await registry_effect.register_node(sample_registry_request)

        # Assert
        assert response.status == "partial"
        assert response.consul_result.success is True
        assert response.postgres_result.success is False
        # Error message is sanitized to avoid exposing secrets (connection strings, etc.)
        # Format: "{ExceptionType}: {original_message}" (sanitize_error_message preserves the message)
        assert "Exception: DB connection failed" in (
            response.postgres_result.error or ""
        )
        assert response.correlation_id == sample_registry_request.correlation_id
        assert response.node_id == sample_registry_request.node_id

        # Verify consul was called successfully
        mock_consul_client.register_service.assert_called_once()

        # Verify postgres was attempted
        mock_postgres_handler.upsert.assert_called_once()

    @pytest.mark.asyncio
    async def test_consul_failure_postgres_success(
        self,
        registry_effect: NodeRegistryEffect,
        mock_consul_client: AsyncMock,
        mock_postgres_handler: AsyncMock,
        sample_registry_request: ModelRegistryRequest,
    ) -> None:
        """Test partial failure when Consul fails but PostgreSQL succeeds.

        Scenario:
            - Consul registration fails with service unavailable
            - PostgreSQL upsert succeeds

        Expected:
            - response.status == "partial"
            - response.consul_result.success == False
            - response.postgres_result.success == True
            - Error context preserved for Consul failure
        """
        # Arrange
        mock_consul_client.register_service.side_effect = Exception(
            "Consul service unavailable"
        )
        mock_postgres_handler.upsert.return_value = ModelBackendResult(success=True)

        # Act
        response = await registry_effect.register_node(sample_registry_request)

        # Assert
        assert response.status == "partial"
        assert response.consul_result.success is False
        assert response.postgres_result.success is True
        # Error message is sanitized to avoid exposing secrets (connection strings, etc.)
        # Format: "{ExceptionType}: {original_message}" (sanitize_error_message preserves the message)
        assert "Exception: Consul service unavailable" in (
            response.consul_result.error or ""
        )
        assert response.consul_result.error_code == "CONSUL_UNKNOWN_ERROR"
        assert response.correlation_id == sample_registry_request.correlation_id

        # Verify appropriate error context
        assert response.error_summary is not None
        assert "Consul" in response.error_summary

    @pytest.mark.asyncio
    async def test_both_backends_fail(
        self,
        registry_effect: NodeRegistryEffect,
        mock_consul_client: AsyncMock,
        mock_postgres_handler: AsyncMock,
        sample_registry_request: ModelRegistryRequest,
    ) -> None:
        """Test complete failure when both backends fail.

        Scenario:
            - Consul registration fails
            - PostgreSQL upsert fails

        Expected:
            - response.status == "failed"
            - Both results show success == False
            - Both error contexts preserved
            - No partial state left
        """
        # Arrange
        mock_consul_client.register_service.side_effect = Exception(
            "Consul connection refused"
        )
        mock_postgres_handler.upsert.side_effect = Exception("PostgreSQL timeout")

        # Act
        response = await registry_effect.register_node(sample_registry_request)

        # Assert
        assert response.status == "failed"
        assert response.consul_result.success is False
        assert response.postgres_result.success is False

        # Verify error messages are sanitized (no raw exception messages that may contain secrets)
        # Format: "{ExceptionType}: {original_message}" (sanitize_error_message preserves the message)
        assert "Exception: Consul connection refused" in (
            response.consul_result.error or ""
        )
        assert "Exception: PostgreSQL timeout" in (response.postgres_result.error or "")

        # Verify error summary aggregates both errors
        assert response.error_summary is not None
        assert "Consul" in response.error_summary
        assert "PostgreSQL" in response.error_summary

        # Verify no partial state left (completed backends cache should be empty)
        completed = await registry_effect.get_completed_backends(
            sample_registry_request.correlation_id
        )
        assert len(completed) == 0

        # Verify both backends report errors
        failed_backends = response.get_failed_backends()
        assert "consul" in failed_backends
        assert "postgres" in failed_backends

    @pytest.mark.asyncio
    async def test_partial_failure_idempotency(
        self,
        registry_effect: NodeRegistryEffect,
        mock_consul_client: AsyncMock,
        mock_postgres_handler: AsyncMock,
    ) -> None:
        """Test idempotent retry after partial failure.

        Scenario:
            - First attempt: Consul succeeds, PostgreSQL fails
            - Retry same intent
            - Consul should NOT be called again (already succeeded)
            - Only PostgreSQL should be retried

        Expected:
            - Second attempt only calls PostgreSQL
            - Consul call count remains 1
            - Final response is success if PostgreSQL retry succeeds
        """
        # Create request with specific correlation_id for tracking
        correlation_id = uuid4()
        request = ModelRegistryRequest(
            node_id=uuid4(),
            node_type="effect",
            node_version="1.0.0",
            correlation_id=correlation_id,
            service_name="test-service",
            timestamp=datetime(2025, 1, 1, tzinfo=UTC),
        )

        # Arrange - First attempt: Consul succeeds, PostgreSQL fails
        mock_consul_client.register_service.return_value = ModelBackendResult(
            success=True
        )
        mock_postgres_handler.upsert.side_effect = Exception("DB connection failed")

        # Act - First attempt
        response1 = await registry_effect.register_node(request)

        # Verify first attempt result
        assert response1.status == "partial"
        assert response1.consul_result.success is True
        assert response1.postgres_result.success is False
        assert mock_consul_client.register_service.call_count == 1

        # Verify Consul is marked as completed
        completed = await registry_effect.get_completed_backends(correlation_id)
        assert "consul" in completed
        assert "postgres" not in completed

        # Arrange - Second attempt: PostgreSQL now succeeds
        mock_postgres_handler.upsert.side_effect = None
        mock_postgres_handler.upsert.return_value = ModelBackendResult(success=True)

        # Act - Second attempt (retry)
        response2 = await registry_effect.register_node(request)

        # Assert - Consul NOT called again, only PostgreSQL retried
        assert response2.status == "success"
        assert response2.consul_result.success is True
        assert response2.postgres_result.success is True

        # CRITICAL: Consul should NOT be called again
        assert mock_consul_client.register_service.call_count == 1

        # PostgreSQL should be called twice (initial failure + retry)
        assert mock_postgres_handler.upsert.call_count == 2

    @pytest.mark.asyncio
    async def test_partial_failure_error_aggregation(
        self,
        registry_effect: NodeRegistryEffect,
        mock_consul_client: AsyncMock,
        mock_postgres_handler: AsyncMock,
        sample_registry_request: ModelRegistryRequest,
    ) -> None:
        """Test error aggregation when both backends fail with different errors.

        Scenario:
            - Consul fails with "service unavailable" error
            - PostgreSQL fails with "connection timeout" error

        Expected:
            - error_summary contains both sanitized error messages
            - Each backend's error context is preserved
            - correlation_id present in all error contexts

        Note:
            Error messages are sanitized to prevent credential/secret leakage.
            Raw error messages like "Connection pool exhausted" are sanitized
            to generic messages. Use safe patterns like "unavailable" for
            error messages that should be preserved.
        """
        # Arrange - Both backends fail with distinct errors
        # Note: "unavailable" is a safe pattern that passes through sanitization
        mock_consul_client.register_service.return_value = ModelBackendResult(
            success=False,
            error="Service unavailable",
        )
        mock_postgres_handler.upsert.return_value = ModelBackendResult(
            success=False,
            error="Connection timeout",
        )

        # Act
        response = await registry_effect.register_node(sample_registry_request)

        # Assert - Status is failed
        assert response.status == "failed"

        # Assert - Each backend has sanitized error (safe patterns preserved)
        # "unavailable" and "connection timeout" are safe patterns that pass through
        assert "unavailable" in (response.consul_result.error or "").lower()
        assert "timeout" in (response.postgres_result.error or "").lower()

        # Assert - Aggregated error summary contains both backend names
        assert response.error_summary is not None
        assert "Consul" in response.error_summary
        assert "PostgreSQL" in response.error_summary

        # Assert - Correlation ID preserved in results
        assert (
            response.consul_result.correlation_id
            == sample_registry_request.correlation_id
        )
        assert (
            response.postgres_result.correlation_id
            == sample_registry_request.correlation_id
        )

        # Assert - Error codes set for programmatic handling
        assert response.consul_result.error_code == "CONSUL_REGISTRATION_ERROR"
        assert response.postgres_result.error_code == "POSTGRES_UPSERT_ERROR"

    @pytest.mark.asyncio
    async def test_partial_failure_processing_time(
        self,
        registry_effect: NodeRegistryEffect,
        mock_consul_client: AsyncMock,
        mock_postgres_handler: AsyncMock,
        sample_registry_request: ModelRegistryRequest,
    ) -> None:
        """Test that processing_time_ms reflects actual duration with timeout.

        Scenario:
            - Consul succeeds quickly (~10ms simulated)
            - PostgreSQL times out (~100ms simulated)

        Expected:
            - processing_time_ms reflects actual total duration
            - Timeout backend marked as failed
            - Individual backend durations tracked
        """

        # Arrange - Consul succeeds quickly
        async def quick_consul_success(
            *args: object, **kwargs: object
        ) -> ModelBackendResult:
            await asyncio.sleep(0.01)  # 10ms
            return ModelBackendResult(success=True)

        # Arrange - PostgreSQL times out (simulated with slower operation + failure)
        async def slow_postgres_timeout(
            *args: object, **kwargs: object
        ) -> ModelBackendResult:
            await asyncio.sleep(0.1)  # 100ms
            raise TimeoutError("PostgreSQL operation timed out")

        mock_consul_client.register_service.side_effect = quick_consul_success
        mock_postgres_handler.upsert.side_effect = slow_postgres_timeout

        # Act
        response = await registry_effect.register_node(sample_registry_request)

        # Assert - Status is partial (Consul succeeded, PostgreSQL failed)
        assert response.status == "partial"
        assert response.consul_result.success is True
        assert response.postgres_result.success is False

        # Assert - Processing time reflects actual duration (at least 100ms total)
        assert response.processing_time_ms >= 100.0  # At least PostgreSQL's 100ms

        # Assert - Individual backend durations tracked
        assert response.consul_result.duration_ms >= 10.0  # Consul's 10ms
        assert response.postgres_result.duration_ms >= 100.0  # PostgreSQL's 100ms

        # Assert - Timeout exception type is captured in sanitized error message
        # Format: "{ExceptionType}: {original_message}" (exception type and message preserved)
        assert "TimeoutError: PostgreSQL operation timed out" in (
            response.postgres_result.error or ""
        )


# -----------------------------------------------------------------------------
# Additional Edge Case Tests
# -----------------------------------------------------------------------------


@pytest.mark.unit
class TestPartialFailureEdgeCases:
    """Additional edge case tests for partial failure handling."""

    @pytest.mark.asyncio
    async def test_success_both_backends(
        self,
        registry_effect: NodeRegistryEffect,
        mock_consul_client: AsyncMock,
        mock_postgres_handler: AsyncMock,
        sample_registry_request: ModelRegistryRequest,
    ) -> None:
        """Test that both backends succeeding returns success status.

        This is the baseline test to ensure normal operation works.
        """
        # Arrange
        mock_consul_client.register_service.return_value = ModelBackendResult(
            success=True
        )
        mock_postgres_handler.upsert.return_value = ModelBackendResult(success=True)

        # Act
        response = await registry_effect.register_node(sample_registry_request)

        # Assert
        assert response.status == "success"
        assert response.consul_result.success is True
        assert response.postgres_result.success is True
        assert response.error_summary is None
        assert len(response.get_failed_backends()) == 0
        assert set(response.get_successful_backends()) == {"consul", "postgres"}

    @pytest.mark.asyncio
    async def test_clear_completed_backends_enables_retry(
        self,
        registry_effect: NodeRegistryEffect,
        mock_consul_client: AsyncMock,
        mock_postgres_handler: AsyncMock,
    ) -> None:
        """Test that clearing completed backends allows full re-registration."""
        # Create request
        correlation_id = uuid4()
        request = ModelRegistryRequest(
            node_id=uuid4(),
            node_type="effect",
            node_version="1.0.0",
            correlation_id=correlation_id,
            timestamp=datetime(2025, 1, 1, tzinfo=UTC),
        )

        # First registration - both succeed
        mock_consul_client.register_service.return_value = ModelBackendResult(
            success=True
        )
        mock_postgres_handler.upsert.return_value = ModelBackendResult(success=True)

        response1 = await registry_effect.register_node(request)
        assert response1.status == "success"
        assert mock_consul_client.register_service.call_count == 1

        # Clear completed backends
        await registry_effect.clear_completed_backends(correlation_id)

        # Second registration - should call both backends again
        response2 = await registry_effect.register_node(request)
        assert response2.status == "success"
        assert mock_consul_client.register_service.call_count == 2  # Called again

    @pytest.mark.asyncio
    async def test_response_helper_methods(
        self,
        registry_effect: NodeRegistryEffect,
        mock_consul_client: AsyncMock,
        mock_postgres_handler: AsyncMock,
        sample_registry_request: ModelRegistryRequest,
    ) -> None:
        """Test ModelRegistryResponse helper methods work correctly."""
        # Arrange - Partial failure
        mock_consul_client.register_service.return_value = ModelBackendResult(
            success=True
        )
        mock_postgres_handler.upsert.side_effect = Exception("DB error")

        # Act
        response = await registry_effect.register_node(sample_registry_request)

        # Assert helper methods
        assert response.is_partial_failure() is True
        assert response.is_complete_success() is False
        assert response.is_complete_failure() is False
        assert response.get_failed_backends() == ["postgres"]
        assert response.get_successful_backends() == ["consul"]

    @pytest.mark.asyncio
    async def test_skip_backend_flags(
        self,
        registry_effect: NodeRegistryEffect,
        mock_consul_client: AsyncMock,
        mock_postgres_handler: AsyncMock,
        sample_registry_request: ModelRegistryRequest,
    ) -> None:
        """Test that skip_consul and skip_postgres flags work correctly."""
        # Arrange
        mock_consul_client.register_service.return_value = ModelBackendResult(
            success=True
        )
        mock_postgres_handler.upsert.return_value = ModelBackendResult(success=True)

        # Act - Skip Consul
        response = await registry_effect.register_node(
            sample_registry_request,
            skip_consul=True,
        )

        # Assert - Consul not called, PostgreSQL called
        assert response.status == "success"
        mock_consul_client.register_service.assert_not_called()
        mock_postgres_handler.upsert.assert_called_once()

        # Reset mocks
        mock_consul_client.reset_mock()
        mock_postgres_handler.reset_mock()
        await registry_effect.clear_completed_backends(
            sample_registry_request.correlation_id
        )

        # Act - Skip PostgreSQL
        response = await registry_effect.register_node(
            sample_registry_request,
            skip_postgres=True,
        )

        # Assert - Consul called, PostgreSQL not called
        assert response.status == "success"
        mock_consul_client.register_service.assert_called_once()
        mock_postgres_handler.upsert.assert_not_called()


# -----------------------------------------------------------------------------
# Test ModelBackendResult directly
# -----------------------------------------------------------------------------


@pytest.mark.unit
class TestModelBackendResult:
    """Unit tests for ModelBackendResult model."""

    def test_success_result(self) -> None:
        """Test creating a successful backend result."""
        result = ModelBackendResult(
            success=True,
            duration_ms=45.2,
            backend_id="consul",
            correlation_id=uuid4(),
        )
        assert result.success is True
        assert result.error is None
        assert result.duration_ms == 45.2

    def test_failure_result(self) -> None:
        """Test creating a failed backend result."""
        correlation_id = uuid4()
        result = ModelBackendResult(
            success=False,
            error="Connection refused",
            error_code="DATABASE_CONNECTION_ERROR",
            duration_ms=5000.0,
            backend_id="postgres",
            correlation_id=correlation_id,
        )
        assert result.success is False
        assert result.error == "Connection refused"
        assert result.error_code == "DATABASE_CONNECTION_ERROR"
        assert result.correlation_id == correlation_id


# -----------------------------------------------------------------------------
# Test ModelRegistryResponse factory method
# -----------------------------------------------------------------------------


@pytest.mark.unit
class TestModelRegistryResponseFactory:
    """Unit tests for ModelRegistryResponse.from_backend_results factory."""

    def test_from_backend_results_success(self) -> None:
        """Test factory creates success status when both backends succeed."""
        node_id = uuid4()
        correlation_id = uuid4()
        consul = ModelBackendResult(success=True, duration_ms=10.0)
        postgres = ModelBackendResult(success=True, duration_ms=20.0)

        response = ModelRegistryResponse.from_backend_results(
            node_id=node_id,
            correlation_id=correlation_id,
            consul_result=consul,
            postgres_result=postgres,
            timestamp=datetime(2025, 1, 1, tzinfo=UTC),
        )

        assert response.status == "success"
        assert response.error_summary is None
        # Processing time is sum of backend durations
        assert response.processing_time_ms == 30.0

    def test_from_backend_results_partial(self) -> None:
        """Test factory creates partial status when one backend fails."""
        node_id = uuid4()
        correlation_id = uuid4()
        consul = ModelBackendResult(success=True, duration_ms=10.0)
        postgres = ModelBackendResult(
            success=False,
            error="Connection failed",
            duration_ms=5000.0,
        )

        response = ModelRegistryResponse.from_backend_results(
            node_id=node_id,
            correlation_id=correlation_id,
            consul_result=consul,
            postgres_result=postgres,
            timestamp=datetime(2025, 1, 1, tzinfo=UTC),
        )

        assert response.status == "partial"
        assert "PostgreSQL" in (response.error_summary or "")

    def test_from_backend_results_failed(self) -> None:
        """Test factory creates failed status when both backends fail."""
        node_id = uuid4()
        correlation_id = uuid4()
        consul = ModelBackendResult(
            success=False,
            error="Consul error",
            duration_ms=1000.0,
        )
        postgres = ModelBackendResult(
            success=False,
            error="Postgres error",
            duration_ms=2000.0,
        )

        response = ModelRegistryResponse.from_backend_results(
            node_id=node_id,
            correlation_id=correlation_id,
            consul_result=consul,
            postgres_result=postgres,
            timestamp=datetime(2025, 1, 1, tzinfo=UTC),
        )

        assert response.status == "failed"
        assert "Consul" in (response.error_summary or "")
        assert "PostgreSQL" in (response.error_summary or "")
