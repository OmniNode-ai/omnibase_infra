# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Integration tests for NodeRegistryEffect end-to-end flow.

MIGRATION STATUS (OMN-1103):
    These tests were written for the imperative NodeRegistryEffect which had an
    execute_operation() method. After the declarative refactoring (OMN-1103):

    1. The new declarative node at omnibase_infra.nodes.node_registry_effect.node
       is now a contract-driven shell with NO custom methods.

    2. The legacy module at omnibase_infra.nodes.effects.registry_effect uses a
       different API (register_node() instead of execute_operation()).

    3. Handler behavior is tested in:
       tests/unit/nodes/node_registry_effect/handlers/

    4. These integration tests are SKIPPED pending migration to:
       - Test handlers directly
       - Test via NodeRegistrationOrchestrator workflow coordination

Test Categories (SKIPPED):
    - TestRegisterNodeSuccess: Both backends succeed scenarios
    - TestParallelExecution: Verify parallel handler execution
    - TestPartialFailureHandling: One backend fails scenarios
    - TestCompleteFailure: Both backends fail scenario
    - TestDeregisterNode: Deregistration operation tests
    - TestRetryPartialFailure: Targeted backend retry tests

Related Tickets:
    - OMN-1103: NodeRegistryEffect refactoring to declarative pattern
"""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from omnibase_core.enums.enum_node_kind import EnumNodeKind
from omnibase_core.models.primitives.model_semver import ModelSemVer

from omnibase_infra.nodes.effects import NodeRegistryEffect
from omnibase_infra.nodes.effects.models import (
    ModelBackendResult,
    ModelRegistryRequest,
)

# Skip all tests in this module - pending migration after declarative refactoring
pytestmark = pytest.mark.skip(
    reason="OMN-1103: Tests pending migration after declarative refactoring. "
    "Handler behavior is tested in tests/unit/nodes/node_registry_effect/handlers/. "
    "These integration tests will be updated to use NodeRegistrationOrchestrator."
)

# Fixed timestamp for deterministic tests
TEST_TIMESTAMP = datetime(2025, 1, 15, 12, 0, 0, tzinfo=UTC)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_consul_client() -> AsyncMock:
    """Create a mock ProtocolConsulClient with success responses.

    Returns:
        AsyncMock configured for successful Consul operations.
    """
    client = AsyncMock()
    client.register_service = AsyncMock(
        return_value=ModelBackendResult(
            success=True,
            duration_ms=10.0,
            backend_id="consul",
        )
    )
    client.deregister_service = AsyncMock(
        return_value=ModelBackendResult(
            success=True,
            duration_ms=10.0,
            backend_id="consul",
        )
    )
    return client


@pytest.fixture
def mock_postgres_adapter() -> AsyncMock:
    """Create a mock ProtocolPostgresAdapter with success responses.

    Returns:
        AsyncMock configured for successful PostgreSQL operations.
    """
    adapter = AsyncMock()
    # Note: handlers call upsert() and deactivate(), not upsert_registration()
    adapter.upsert = AsyncMock(
        return_value=ModelBackendResult(
            success=True,
            duration_ms=15.0,
            backend_id="postgres",
        )
    )
    adapter.deactivate = AsyncMock(
        return_value=ModelBackendResult(
            success=True,
            duration_ms=12.0,
            backend_id="postgres",
        )
    )
    return adapter


@pytest.fixture
def simple_mock_container() -> MagicMock:
    """Create a simple mock ONEX container for node tests.

    Returns:
        MagicMock configured with minimal container.config attribute.
    """
    container = MagicMock()
    container.config = MagicMock()
    return container


@pytest.fixture
def registry_effect_node(
    mock_consul_client: AsyncMock,
    mock_postgres_adapter: AsyncMock,
) -> NodeRegistryEffect:
    """Create a NodeRegistryEffect with mock backends configured.

    Note: Uses legacy NodeRegistryEffect from omnibase_infra.nodes.effects which
    takes consul_client and postgres_adapter as constructor arguments.

    Args:
        mock_consul_client: Mock Consul client fixture.
        mock_postgres_adapter: Mock PostgreSQL adapter fixture.

    Returns:
        NodeRegistryEffect configured with mock backends.
    """
    return NodeRegistryEffect(mock_consul_client, mock_postgres_adapter)


@pytest.fixture
def registry_request() -> ModelRegistryRequest:
    """Create a standard registry request for testing.

    Returns:
        ModelRegistryRequest with test data.
    """
    return ModelRegistryRequest(
        node_id=uuid4(),
        node_type=EnumNodeKind.EFFECT,
        node_version=ModelSemVer.parse("1.0.0"),
        correlation_id=uuid4(),
        service_name="test-service",
        endpoints={"health": "http://localhost:8080/health"},
        tags=["test", "integration"],
        metadata={"environment": "test"},
        timestamp=TEST_TIMESTAMP,
    )


# =============================================================================
# TestRegisterNodeSuccess
# =============================================================================


class TestRegisterNodeSuccess:
    """Tests for successful registration with both backends."""

    @pytest.mark.asyncio
    async def test_register_node_success_both_backends(
        self,
        registry_effect_node: NodeRegistryEffect,
        registry_request: ModelRegistryRequest,
        mock_consul_client: AsyncMock,
        mock_postgres_adapter: AsyncMock,
    ) -> None:
        """Test that successful registration returns success status.

        Verifies:
            - status is "success" when both backends succeed
            - consul_result.success is True
            - postgres_result.success is True
            - Both handlers are called
        """
        # Act
        response = await registry_effect_node.execute_operation(
            registry_request, "register_node"
        )

        # Assert
        assert response.status == "success"
        assert response.consul_result.success is True
        assert response.postgres_result.success is True
        assert response.node_id == registry_request.node_id
        assert response.correlation_id == registry_request.correlation_id
        assert response.error_summary is None

    @pytest.mark.asyncio
    async def test_register_node_calls_both_handlers(
        self,
        registry_effect_node: NodeRegistryEffect,
        registry_request: ModelRegistryRequest,
        mock_consul_client: AsyncMock,
        mock_postgres_adapter: AsyncMock,
    ) -> None:
        """Test that both Consul and PostgreSQL handlers are called."""
        # Act
        await registry_effect_node.execute_operation(registry_request, "register_node")

        # Assert - both handlers should be called
        mock_consul_client.register_service.assert_called_once()
        mock_postgres_adapter.upsert.assert_called_once()


# =============================================================================
# TestParallelExecution
# =============================================================================


class TestParallelExecution:
    """Tests verifying handlers execute in parallel."""

    @pytest.mark.asyncio
    async def test_register_node_parallel_execution(
        self,
        simple_mock_container: MagicMock,
        registry_request: ModelRegistryRequest,
    ) -> None:
        """Test that handlers are executed in parallel, not sequentially.

        Uses timing to verify parallel execution: if both handlers take 100ms
        and run in parallel, total time should be ~100ms. If sequential, ~200ms.
        """
        # Arrange - Create handlers with deliberate delays
        call_times: dict[str, list[float]] = {"consul": [], "postgres": []}

        async def slow_consul_register(*args, **kwargs) -> ModelBackendResult:
            call_times["consul"].append(time.perf_counter())
            await asyncio.sleep(0.1)  # 100ms delay
            call_times["consul"].append(time.perf_counter())
            return ModelBackendResult(
                success=True, duration_ms=100.0, backend_id="consul"
            )

        async def slow_postgres_upsert(*args, **kwargs) -> ModelBackendResult:
            call_times["postgres"].append(time.perf_counter())
            await asyncio.sleep(0.1)  # 100ms delay
            call_times["postgres"].append(time.perf_counter())
            return ModelBackendResult(
                success=True, duration_ms=100.0, backend_id="postgres"
            )

        mock_consul = AsyncMock()
        mock_consul.register_service = AsyncMock(side_effect=slow_consul_register)

        mock_postgres = AsyncMock()
        mock_postgres.upsert = AsyncMock(side_effect=slow_postgres_upsert)

        node = NodeRegistryEffect(mock_consul, mock_postgres)

        # Act
        start_time = time.perf_counter()
        await node.execute_operation(registry_request, "register_node")
        total_time = time.perf_counter() - start_time

        # Assert - Both handlers should have been called
        assert len(call_times["consul"]) == 2, "Consul handler should start and end"
        assert len(call_times["postgres"]) == 2, "Postgres handler should start and end"

        # If parallel, total time should be ~100ms (max of both), not ~200ms (sum)
        # Allow some margin for test overhead
        assert total_time < 0.18, (
            f"Total time {total_time:.3f}s suggests sequential execution. "
            "Expected parallel execution with total time < 180ms."
        )

        # Verify overlap: both handlers should start before either ends
        consul_start = call_times["consul"][0]
        postgres_start = call_times["postgres"][0]
        consul_end = call_times["consul"][1]
        postgres_end = call_times["postgres"][1]

        # Both should start before either ends (indicates parallel)
        assert consul_start < postgres_end, "Consul should start before Postgres ends"
        assert postgres_start < consul_end, "Postgres should start before Consul ends"


# =============================================================================
# TestPartialFailureHandling
# =============================================================================


class TestPartialFailureHandling:
    """Tests for partial failure scenarios (one backend fails)."""

    @pytest.mark.asyncio
    async def test_register_node_partial_failure_consul_fails(
        self,
        simple_mock_container: MagicMock,
        registry_request: ModelRegistryRequest,
    ) -> None:
        """Test partial failure when Consul fails but PostgreSQL succeeds.

        Verifies:
            - status is "partial"
            - consul_result.success is False
            - postgres_result.success is True
            - error_summary contains Consul error
        """
        # Arrange - Consul fails, Postgres succeeds
        mock_consul = AsyncMock()
        mock_consul.register_service = AsyncMock(
            return_value=ModelBackendResult(
                success=False,
                error="Connection refused to Consul",
                error_code="CONSUL_CONNECTION_ERROR",
                duration_ms=5.0,
                backend_id="consul",
            )
        )

        mock_postgres = AsyncMock()
        mock_postgres.upsert = AsyncMock(
            return_value=ModelBackendResult(
                success=True,
                duration_ms=10.0,
                backend_id="postgres",
            )
        )

        node = NodeRegistryEffect(mock_consul, mock_postgres)

        # Act
        response = await node.execute_operation(registry_request, "register_node")

        # Assert
        assert response.status == "partial"
        assert response.consul_result.success is False
        assert response.postgres_result.success is True
        assert response.error_summary is not None
        assert "Consul" in response.error_summary

    @pytest.mark.asyncio
    async def test_register_node_partial_failure_postgres_fails(
        self,
        simple_mock_container: MagicMock,
        registry_request: ModelRegistryRequest,
    ) -> None:
        """Test partial failure when PostgreSQL fails but Consul succeeds.

        Verifies:
            - status is "partial"
            - consul_result.success is True
            - postgres_result.success is False
            - error_summary contains PostgreSQL error
        """
        # Arrange - Consul succeeds, Postgres fails
        mock_consul = AsyncMock()
        mock_consul.register_service = AsyncMock(
            return_value=ModelBackendResult(
                success=True,
                duration_ms=10.0,
                backend_id="consul",
            )
        )

        mock_postgres = AsyncMock()
        mock_postgres.upsert = AsyncMock(
            return_value=ModelBackendResult(
                success=False,
                error="Database connection timeout",
                error_code="POSTGRES_TIMEOUT_ERROR",
                duration_ms=5000.0,
                backend_id="postgres",
            )
        )

        node = NodeRegistryEffect(mock_consul, mock_postgres)

        # Act
        response = await node.execute_operation(registry_request, "register_node")

        # Assert
        assert response.status == "partial"
        assert response.consul_result.success is True
        assert response.postgres_result.success is False
        assert response.error_summary is not None
        assert "PostgreSQL" in response.error_summary

    @pytest.mark.asyncio
    async def test_partial_failure_get_failed_backends(
        self,
        simple_mock_container: MagicMock,
        registry_request: ModelRegistryRequest,
    ) -> None:
        """Test get_failed_backends() helper on partial failure."""
        # Arrange - Consul fails
        mock_consul = AsyncMock()
        mock_consul.register_service = AsyncMock(
            return_value=ModelBackendResult(
                success=False,
                error="Consul unavailable",
                duration_ms=5.0,
                backend_id="consul",
            )
        )

        mock_postgres = AsyncMock()
        mock_postgres.upsert = AsyncMock(
            return_value=ModelBackendResult(success=True, duration_ms=10.0)
        )

        node = NodeRegistryEffect(mock_consul, mock_postgres)

        # Act
        response = await node.execute_operation(registry_request, "register_node")

        # Assert
        assert response.is_partial_failure()
        failed_backends = response.get_failed_backends()
        assert "consul" in failed_backends
        assert "postgres" not in failed_backends


# =============================================================================
# TestCompleteFailure
# =============================================================================


class TestCompleteFailure:
    """Tests for complete failure scenario (both backends fail)."""

    @pytest.mark.asyncio
    async def test_register_node_both_fail(
        self,
        simple_mock_container: MagicMock,
        registry_request: ModelRegistryRequest,
    ) -> None:
        """Test complete failure when both backends fail.

        Verifies:
            - status is "failed"
            - consul_result.success is False
            - postgres_result.success is False
            - error_summary contains both errors
        """
        # Arrange - Both backends fail
        mock_consul = AsyncMock()
        mock_consul.register_service = AsyncMock(
            return_value=ModelBackendResult(
                success=False,
                error="Consul connection refused",
                error_code="CONSUL_CONNECTION_ERROR",
                duration_ms=5.0,
                backend_id="consul",
            )
        )

        mock_postgres = AsyncMock()
        mock_postgres.upsert = AsyncMock(
            return_value=ModelBackendResult(
                success=False,
                error="Database unavailable",
                error_code="POSTGRES_UNAVAILABLE_ERROR",
                duration_ms=5.0,
                backend_id="postgres",
            )
        )

        node = NodeRegistryEffect(mock_consul, mock_postgres)

        # Act
        response = await node.execute_operation(registry_request, "register_node")

        # Assert
        assert response.status == "failed"
        assert response.consul_result.success is False
        assert response.postgres_result.success is False
        assert response.is_complete_failure()
        assert response.error_summary is not None
        assert "Consul" in response.error_summary
        assert "PostgreSQL" in response.error_summary


# =============================================================================
# TestDeregisterNode
# =============================================================================


class TestDeregisterNode:
    """Tests for deregistration operation."""

    @pytest.mark.asyncio
    async def test_deregister_node_success(
        self,
        registry_effect_node: NodeRegistryEffect,
        registry_request: ModelRegistryRequest,
        mock_consul_client: AsyncMock,
        mock_postgres_adapter: AsyncMock,
    ) -> None:
        """Test successful deregistration from both backends.

        Verifies:
            - status is "success"
            - Both deregistration handlers are called
            - Both backend results show success
        """
        # Act
        response = await registry_effect_node.execute_operation(
            registry_request, "deregister_node"
        )

        # Assert
        assert response.status == "success"
        assert response.consul_result.success is True
        assert response.postgres_result.success is True
        mock_consul_client.deregister_service.assert_called_once()
        mock_postgres_adapter.deactivate.assert_called_once()

    @pytest.mark.asyncio
    async def test_deregister_node_partial_failure(
        self,
        simple_mock_container: MagicMock,
        registry_request: ModelRegistryRequest,
    ) -> None:
        """Test deregistration with partial failure."""
        # Arrange - Consul fails deregistration
        mock_consul = AsyncMock()
        mock_consul.deregister_service = AsyncMock(
            return_value=ModelBackendResult(
                success=False,
                error="Service not found in Consul",
                duration_ms=5.0,
                backend_id="consul",
            )
        )

        mock_postgres = AsyncMock()
        mock_postgres.deactivate = AsyncMock(
            return_value=ModelBackendResult(
                success=True,
                duration_ms=10.0,
                backend_id="postgres",
            )
        )

        node = NodeRegistryEffect(mock_consul, mock_postgres)

        # Act
        response = await node.execute_operation(registry_request, "deregister_node")

        # Assert
        assert response.status == "partial"
        assert response.consul_result.success is False
        assert response.postgres_result.success is True


# =============================================================================
# TestRetryPartialFailure
# =============================================================================


class TestRetryPartialFailure:
    """Tests for retry_partial_failure operation."""

    @pytest.mark.asyncio
    async def test_retry_partial_failure_consul(
        self,
        simple_mock_container: MagicMock,
        registry_request: ModelRegistryRequest,
    ) -> None:
        """Test retry targeting only Consul backend.

        Verifies:
            - Only Consul handler is called during retry
            - PostgreSQL handler is NOT called
            - Successful retry returns success for targeted backend
        """
        # Arrange
        mock_consul = AsyncMock()
        mock_consul.register_service = AsyncMock(
            return_value=ModelBackendResult(
                success=True,
                duration_ms=10.0,
                backend_id="consul",
            )
        )

        mock_postgres = AsyncMock()
        mock_postgres.upsert = AsyncMock(
            return_value=ModelBackendResult(
                success=True,
                duration_ms=10.0,
                backend_id="postgres",
            )
        )

        node = NodeRegistryEffect(mock_consul, mock_postgres)

        # Act - Retry only Consul
        response = await node.execute_operation(
            registry_request, "retry_partial_failure", target_backend="consul"
        )

        # Assert
        assert response.consul_result.success is True
        # Consul registration should be called via the partial retry handler
        mock_consul.register_service.assert_called()

    @pytest.mark.asyncio
    async def test_retry_partial_failure_postgres(
        self,
        simple_mock_container: MagicMock,
        registry_request: ModelRegistryRequest,
    ) -> None:
        """Test retry targeting only PostgreSQL backend.

        Verifies:
            - Only PostgreSQL handler is called during retry
            - Consul handler is NOT called
            - Successful retry returns success for targeted backend
        """
        # Arrange
        mock_consul = AsyncMock()
        mock_consul.register_service = AsyncMock(
            return_value=ModelBackendResult(
                success=True,
                duration_ms=10.0,
                backend_id="consul",
            )
        )

        mock_postgres = AsyncMock()
        mock_postgres.upsert = AsyncMock(
            return_value=ModelBackendResult(
                success=True,
                duration_ms=10.0,
                backend_id="postgres",
            )
        )

        node = NodeRegistryEffect(mock_consul, mock_postgres)

        # Act - Retry only PostgreSQL
        response = await node.execute_operation(
            registry_request, "retry_partial_failure", target_backend="postgres"
        )

        # Assert
        assert response.postgres_result.success is True
        # PostgreSQL registration should be called via the partial retry handler
        mock_postgres.upsert.assert_called()

    @pytest.mark.asyncio
    async def test_retry_partial_failure_missing_target_raises(
        self,
        registry_effect_node: NodeRegistryEffect,
        registry_request: ModelRegistryRequest,
    ) -> None:
        """Test that retry_partial_failure without target_backend raises ValueError."""
        # Act & Assert
        with pytest.raises(ValueError, match="target_backend is required"):
            await registry_effect_node.execute_operation(
                registry_request, "retry_partial_failure"
            )


# =============================================================================
# TestExceptionHandling
# =============================================================================


class TestExceptionHandling:
    """Tests for exception handling during handler execution."""

    @pytest.mark.asyncio
    async def test_handler_exception_captured_as_failure(
        self,
        simple_mock_container: MagicMock,
        registry_request: ModelRegistryRequest,
    ) -> None:
        """Test that exceptions from handlers are captured as failures.

        Verifies:
            - Handler exceptions do not propagate
            - Result shows failure with error message
        """
        # Arrange - Consul handler raises exception
        mock_consul = AsyncMock()
        mock_consul.register_service = AsyncMock(
            side_effect=RuntimeError("Unexpected handler error")
        )

        mock_postgres = AsyncMock()
        mock_postgres.upsert = AsyncMock(
            return_value=ModelBackendResult(success=True, duration_ms=10.0)
        )

        node = NodeRegistryEffect(mock_consul, mock_postgres)

        # Act - Should not raise
        response = await node.execute_operation(registry_request, "register_node")

        # Assert
        assert response.status == "partial"
        assert response.consul_result.success is False
        assert "RuntimeError" in (response.consul_result.error or "")
        assert response.postgres_result.success is True


# =============================================================================
# TestNodeIdAndCorrelationIdPropagation
# =============================================================================


class TestNodeIdAndCorrelationIdPropagation:
    """Tests for node_id and correlation_id propagation."""

    @pytest.mark.asyncio
    async def test_node_id_propagated_to_response(
        self,
        registry_effect_node: NodeRegistryEffect,
        registry_request: ModelRegistryRequest,
    ) -> None:
        """Test that node_id is propagated from request to response."""
        # Act
        response = await registry_effect_node.execute_operation(
            registry_request, "register_node"
        )

        # Assert
        assert response.node_id == registry_request.node_id

    @pytest.mark.asyncio
    async def test_correlation_id_propagated_to_response(
        self,
        registry_effect_node: NodeRegistryEffect,
        registry_request: ModelRegistryRequest,
    ) -> None:
        """Test that correlation_id is propagated from request to response."""
        # Act
        response = await registry_effect_node.execute_operation(
            registry_request, "register_node"
        )

        # Assert
        assert response.correlation_id == registry_request.correlation_id

    @pytest.mark.asyncio
    async def test_correlation_id_in_backend_results(
        self,
        simple_mock_container: MagicMock,
        registry_request: ModelRegistryRequest,
    ) -> None:
        """Test that correlation_id appears in backend results."""
        # Arrange
        correlation_id = registry_request.correlation_id

        mock_consul = AsyncMock()
        mock_consul.register_service = AsyncMock(
            return_value=ModelBackendResult(
                success=True,
                duration_ms=10.0,
                backend_id="consul",
                correlation_id=correlation_id,
            )
        )

        mock_postgres = AsyncMock()
        mock_postgres.upsert = AsyncMock(
            return_value=ModelBackendResult(
                success=True,
                duration_ms=10.0,
                backend_id="postgres",
                correlation_id=correlation_id,
            )
        )

        node = NodeRegistryEffect(mock_consul, mock_postgres)

        # Act
        response = await node.execute_operation(registry_request, "register_node")

        # Assert
        assert response.consul_result.correlation_id == correlation_id
        assert response.postgres_result.correlation_id == correlation_id


# =============================================================================
# Module Exports
# =============================================================================

__all__: list[str] = [
    "TestRegisterNodeSuccess",
    "TestParallelExecution",
    "TestPartialFailureHandling",
    "TestCompleteFailure",
    "TestDeregisterNode",
    "TestRetryPartialFailure",
    "TestExceptionHandling",
    "TestNodeIdAndCorrelationIdPropagation",
]
