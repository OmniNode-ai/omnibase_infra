# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Unit tests for Registry Effect failure scenarios.

This test suite validates the failure handling of the NodeRegistryEffect node,
which operates on a single backend (PostgreSQL) and must handle scenarios
where the backend fails.

Test Coverage (G4 Acceptance Criteria):
    1. test_postgres_failure - PostgreSQL fails
    2. test_both_backends_fail - retained as postgres-only failure for compat
    3. test_partial_failure_idempotency - Retry only failed backend
    4. test_partial_failure_error_aggregation - Error context preservation
    5. test_partial_failure_processing_time - Timing reflects actual duration

Response Status Semantics:
    - "success": PostgreSQL backend succeeded
    - "failed": PostgreSQL backend failed

Related:
    - NodeRegistryEffect: Effect node under test
    - ModelRegistryResponse: Response model with failure support
    - ModelBackendResult: Individual backend result model
    - OMN-954: Partial failure scenario testing ticket
    - OMN-3540: Remove Consul entirely from omnibase_infra runtime
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest

from omnibase_core.enums import EnumNodeKind
from omnibase_core.models.primitives.model_semver import ModelSemVer
from omnibase_infra.models.model_backend_result import ModelBackendResult
from omnibase_infra.nodes.node_registry_effect import NodeRegistryEffect
from omnibase_infra.nodes.node_registry_effect.models import (
    ModelRegistryRequest,
    ModelRegistryResponse,
)

# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def mock_postgres_handler() -> AsyncMock:
    """Create a mock PostgreSQL handler for testing.

    Returns:
        AsyncMock implementing ProtocolPostgresAdapter interface
        (adapter protocol for database operations).
    """
    mock = AsyncMock()
    mock.upsert = AsyncMock(
        return_value=ModelBackendResult(success=True, backend_id="postgres")
    )
    return mock


@pytest.fixture
def registry_effect(
    mock_postgres_handler: AsyncMock,
) -> NodeRegistryEffect:
    """Create a NodeRegistryEffect with mock backend.

    Args:
        mock_postgres_handler: Mock PostgreSQL handler.

    Returns:
        NodeRegistryEffect instance with mocked backend.
    """
    return NodeRegistryEffect(mock_postgres_handler)


@pytest.fixture
def sample_registry_request() -> ModelRegistryRequest:
    """Create a sample registry request for testing.

    Returns:
        ModelRegistryRequest with valid test data.
    """
    return ModelRegistryRequest(
        node_id=uuid4(),
        node_type=EnumNodeKind.EFFECT,  # ModelRegistryRequest uses EnumNodeKind
        node_version=ModelSemVer.parse("1.0.0"),
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
# Test Class: Failure Scenarios
# -----------------------------------------------------------------------------


@pytest.mark.unit
class TestEffectPartialFailure:
    """Test suite for failure scenarios (G4 acceptance criteria).

    These tests validate that the NodeRegistryEffect correctly handles scenarios
    where the PostgreSQL backend fails, preserving appropriate context and
    enabling targeted retries.
    """

    @pytest.mark.asyncio
    async def test_consul_success_postgres_failure(
        self,
        registry_effect: NodeRegistryEffect,
        mock_postgres_handler: AsyncMock,
        sample_registry_request: ModelRegistryRequest,
    ) -> None:
        """Test failure when PostgreSQL fails.

        Scenario:
            - PostgreSQL upsert fails with connection error

        Expected:
            - response.status == "failed"
            - response.postgres_result.success == False
            - correlation_id is preserved in results
        """
        # Arrange
        mock_postgres_handler.upsert.side_effect = Exception("DB connection failed")

        # Act
        response = await registry_effect.register_node(sample_registry_request)

        # Assert
        assert response.status == "failed"
        assert response.postgres_result.success is False
        # Error message is sanitized to avoid exposing secrets (connection strings, etc.)
        # Format: "{ExceptionType}: {original_message}" (sanitize_error_message preserves the message)
        assert "Exception: DB connection failed" in (
            response.postgres_result.error or ""
        )
        assert response.correlation_id == sample_registry_request.correlation_id
        assert response.node_id == sample_registry_request.node_id

        # Verify postgres was attempted
        mock_postgres_handler.upsert.assert_called_once()

    @pytest.mark.asyncio
    async def test_consul_failure_postgres_success(
        self,
        registry_effect: NodeRegistryEffect,
        mock_postgres_handler: AsyncMock,
        sample_registry_request: ModelRegistryRequest,
    ) -> None:
        """Test success when PostgreSQL succeeds.

        Scenario:
            - PostgreSQL upsert succeeds

        Expected:
            - response.status == "success"
            - response.postgres_result.success == True
        """
        # Arrange
        mock_postgres_handler.upsert.return_value = ModelBackendResult(
            success=True, backend_id="postgres"
        )

        # Act
        response = await registry_effect.register_node(sample_registry_request)

        # Assert
        assert response.status == "success"
        assert response.postgres_result.success is True
        assert response.correlation_id == sample_registry_request.correlation_id

    @pytest.mark.asyncio
    async def test_both_backends_fail(
        self,
        registry_effect: NodeRegistryEffect,
        mock_postgres_handler: AsyncMock,
        sample_registry_request: ModelRegistryRequest,
    ) -> None:
        """Test complete failure when PostgreSQL fails.

        Scenario:
            - PostgreSQL upsert fails

        Expected:
            - response.status == "failed"
            - postgres_result shows success == False
            - Error context preserved
        """
        # Arrange
        mock_postgres_handler.upsert.side_effect = Exception("PostgreSQL timeout")

        # Act
        response = await registry_effect.register_node(sample_registry_request)

        # Assert
        assert response.status == "failed"
        assert response.postgres_result.success is False

        # Verify error messages are sanitized (no raw exception messages that may contain secrets)
        # Format: "{ExceptionType}: {original_message}" (sanitize_error_message preserves the message)
        assert "Exception: PostgreSQL timeout" in (response.postgres_result.error or "")

        # Verify error summary captures the error
        assert response.error_summary is not None
        assert "PostgreSQL" in response.error_summary

        # Verify no partial state left (completed backends cache should be empty)
        completed = await registry_effect.get_completed_backends(
            sample_registry_request.correlation_id
        )
        assert len(completed) == 0

        # Verify backend reports error
        failed_backends = response.get_failed_backends()
        assert "postgres" in failed_backends

    @pytest.mark.asyncio
    async def test_partial_failure_idempotency(
        self,
        registry_effect: NodeRegistryEffect,
        mock_postgres_handler: AsyncMock,
    ) -> None:
        """Test idempotent retry after failure.

        Scenario:
            - First attempt: PostgreSQL fails
            - Retry same intent
            - PostgreSQL is retried

        Expected:
            - Second attempt calls PostgreSQL again
            - Final response is success if PostgreSQL retry succeeds
        """
        # Create request with specific correlation_id for tracking
        correlation_id = uuid4()
        request = ModelRegistryRequest(
            node_id=uuid4(),
            node_type=EnumNodeKind.EFFECT,
            node_version=ModelSemVer.parse("1.0.0"),
            correlation_id=correlation_id,
            service_name="test-service",
            timestamp=datetime(2025, 1, 1, tzinfo=UTC),
        )

        # Arrange - First attempt: PostgreSQL fails
        mock_postgres_handler.upsert.side_effect = Exception("DB connection failed")

        # Act - First attempt
        response1 = await registry_effect.register_node(request)

        # Verify first attempt result
        assert response1.status == "failed"
        assert response1.postgres_result.success is False
        assert mock_postgres_handler.upsert.call_count == 1

        # Verify postgres is NOT marked as completed (it failed)
        completed = await registry_effect.get_completed_backends(correlation_id)
        assert "postgres" not in completed

        # Arrange - Second attempt: PostgreSQL now succeeds
        mock_postgres_handler.upsert.side_effect = None
        mock_postgres_handler.upsert.return_value = ModelBackendResult(
            success=True, backend_id="postgres"
        )

        # Act - Second attempt (retry)
        response2 = await registry_effect.register_node(request)

        # Assert - PostgreSQL retried and succeeded
        assert response2.status == "success"
        assert response2.postgres_result.success is True

        # PostgreSQL should be called twice (initial failure + retry)
        assert mock_postgres_handler.upsert.call_count == 2

    @pytest.mark.asyncio
    async def test_partial_failure_error_aggregation(
        self,
        registry_effect: NodeRegistryEffect,
        mock_postgres_handler: AsyncMock,
        sample_registry_request: ModelRegistryRequest,
    ) -> None:
        """Test error aggregation when backend fails.

        Scenario:
            - PostgreSQL fails with "connection timeout" error

        Expected:
            - error_summary contains sanitized error message
            - Backend's error context is preserved
            - correlation_id present in error context

        Note:
            Error messages are sanitized to prevent credential/secret leakage.
            Raw error messages like "Connection pool exhausted" are sanitized
            to generic messages. Use safe patterns like "unavailable" for
            error messages that should be preserved.
        """
        # Arrange - Backend fails with distinct error
        # Note: "timeout" is a safe pattern that passes through sanitization
        mock_postgres_handler.upsert.return_value = ModelBackendResult(
            success=False,
            error="Connection timeout",
        )

        # Act
        response = await registry_effect.register_node(sample_registry_request)

        # Assert - Status is failed
        assert response.status == "failed"

        # Assert - Backend has sanitized error (safe patterns preserved)
        # "connection timeout" is a safe pattern that passes through
        assert "timeout" in (response.postgres_result.error or "").lower()

        # Assert - Aggregated error summary contains backend name
        assert response.error_summary is not None
        assert "PostgreSQL" in response.error_summary

        # Assert - Correlation ID preserved in results
        assert (
            response.postgres_result.correlation_id
            == sample_registry_request.correlation_id
        )

        # Assert - Error codes set for programmatic handling
        assert response.postgres_result.error_code == "POSTGRES_UPSERT_ERROR"

    @pytest.mark.asyncio
    async def test_partial_failure_processing_time(
        self,
        registry_effect: NodeRegistryEffect,
        mock_postgres_handler: AsyncMock,
        sample_registry_request: ModelRegistryRequest,
    ) -> None:
        """Test that processing_time_ms reflects actual duration with timeout.

        Scenario:
            - PostgreSQL times out (~100ms simulated)

        Expected:
            - processing_time_ms reflects actual total duration
            - Timeout backend marked as failed
            - Individual backend durations tracked
        """

        # Arrange - PostgreSQL times out (simulated with slower operation + failure)
        async def slow_postgres_timeout(
            *args: object, **kwargs: object
        ) -> ModelBackendResult:
            await asyncio.sleep(0.1)  # 100ms
            raise TimeoutError("PostgreSQL operation timed out")

        mock_postgres_handler.upsert.side_effect = slow_postgres_timeout

        # Act
        response = await registry_effect.register_node(sample_registry_request)

        # Assert - Status is failed (PostgreSQL failed)
        assert response.status == "failed"
        assert response.postgres_result.success is False

        # Assert - Processing time reflects actual duration (at least 100ms total)
        assert response.processing_time_ms >= 100.0  # At least PostgreSQL's 100ms

        # Assert - Individual backend duration tracked
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
    """Additional edge case tests for failure handling."""

    @pytest.mark.asyncio
    async def test_success_both_backends(
        self,
        registry_effect: NodeRegistryEffect,
        mock_postgres_handler: AsyncMock,
        sample_registry_request: ModelRegistryRequest,
    ) -> None:
        """Test that PostgreSQL succeeding returns success status.

        This is the baseline test to ensure normal operation works.
        """
        # Arrange
        mock_postgres_handler.upsert.return_value = ModelBackendResult(
            success=True, backend_id="postgres"
        )

        # Act
        response = await registry_effect.register_node(sample_registry_request)

        # Assert
        assert response.status == "success"
        assert response.postgres_result.success is True
        assert response.error_summary is None
        assert len(response.get_failed_backends()) == 0
        assert set(response.get_successful_backends()) == {"postgres"}

    @pytest.mark.asyncio
    async def test_clear_completed_backends_enables_retry(
        self,
        registry_effect: NodeRegistryEffect,
        mock_postgres_handler: AsyncMock,
    ) -> None:
        """Test that clearing completed backends allows full re-registration."""
        # Create request
        correlation_id = uuid4()
        request = ModelRegistryRequest(
            node_id=uuid4(),
            node_type=EnumNodeKind.EFFECT,
            node_version=ModelSemVer.parse("1.0.0"),
            correlation_id=correlation_id,
            timestamp=datetime(2025, 1, 1, tzinfo=UTC),
        )

        # First registration - succeeds
        mock_postgres_handler.upsert.return_value = ModelBackendResult(
            success=True, backend_id="postgres"
        )

        response1 = await registry_effect.register_node(request)
        assert response1.status == "success"
        assert mock_postgres_handler.upsert.call_count == 1

        # Clear completed backends
        await registry_effect.clear_completed_backends(correlation_id)

        # Second registration - should call backend again
        response2 = await registry_effect.register_node(request)
        assert response2.status == "success"
        assert mock_postgres_handler.upsert.call_count == 2  # Called again

    @pytest.mark.asyncio
    async def test_response_helper_methods(
        self,
        registry_effect: NodeRegistryEffect,
        mock_postgres_handler: AsyncMock,
        sample_registry_request: ModelRegistryRequest,
    ) -> None:
        """Test ModelRegistryResponse helper methods work correctly."""
        # Arrange - Failure
        mock_postgres_handler.upsert.side_effect = Exception("DB error")

        # Act
        response = await registry_effect.register_node(sample_registry_request)

        # Assert helper methods
        assert response.is_complete_success() is False
        assert response.is_complete_failure() is True
        assert response.get_failed_backends() == ["postgres"]
        assert response.get_successful_backends() == []

    @pytest.mark.asyncio
    async def test_skip_backend_flags(
        self,
        registry_effect: NodeRegistryEffect,
        mock_postgres_handler: AsyncMock,
        sample_registry_request: ModelRegistryRequest,
    ) -> None:
        """Test that skip_postgres flag works correctly."""
        # Arrange
        mock_postgres_handler.upsert.return_value = ModelBackendResult(
            success=True, backend_id="postgres"
        )

        # Act - Skip PostgreSQL
        response = await registry_effect.register_node(
            sample_registry_request,
            skip_postgres=True,
        )

        # Assert - PostgreSQL not called, response is success (skipped = treated as success)
        assert response.status == "success"
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
            backend_id="postgres",
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
        """Test factory creates success status when backend succeeds."""
        node_id = uuid4()
        correlation_id = uuid4()
        postgres = ModelBackendResult(
            success=True, duration_ms=20.0, backend_id="postgres"
        )

        response = ModelRegistryResponse.from_backend_results(
            node_id=node_id,
            correlation_id=correlation_id,
            postgres_result=postgres,
            timestamp=datetime(2025, 1, 1, tzinfo=UTC),
        )

        assert response.status == "success"
        assert response.error_summary is None
        # Processing time is taken from backend duration
        assert response.processing_time_ms == 20.0

    def test_from_backend_results_partial(self) -> None:
        """Test factory creates failed status when backend fails."""
        node_id = uuid4()
        correlation_id = uuid4()
        postgres = ModelBackendResult(
            success=False,
            error="Connection failed",
            duration_ms=5000.0,
            backend_id="postgres",
        )

        response = ModelRegistryResponse.from_backend_results(
            node_id=node_id,
            correlation_id=correlation_id,
            postgres_result=postgres,
            timestamp=datetime(2025, 1, 1, tzinfo=UTC),
        )

        assert response.status == "failed"
        assert "PostgreSQL" in (response.error_summary or "")

    def test_from_backend_results_failed(self) -> None:
        """Test factory creates failed status when backend fails."""
        node_id = uuid4()
        correlation_id = uuid4()
        postgres = ModelBackendResult(
            success=False,
            error="Postgres error",
            duration_ms=2000.0,
            backend_id="postgres",
        )

        response = ModelRegistryResponse.from_backend_results(
            node_id=node_id,
            correlation_id=correlation_id,
            postgres_result=postgres,
            timestamp=datetime(2025, 1, 1, tzinfo=UTC),
        )

        assert response.status == "failed"
        assert "PostgreSQL" in (response.error_summary or "")
