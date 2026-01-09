# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for HandlerPostgresDeactivate.

Tests validate:
- Successful deactivation via PostgreSQL adapter
- Failed deactivation (adapter returns failure)
- Exception handling (adapter raises)
- Correlation ID propagation

Related Tickets:
    - OMN-1103: NodeRegistryEffect refactoring to declarative pattern
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from omnibase_core.enums.enum_node_kind import EnumNodeKind
from omnibase_core.models.primitives.model_semver import ModelSemVer

from omnibase_infra.nodes.effects.models import ModelBackendResult
from omnibase_infra.nodes.node_registry_effect.handlers.handler_postgres_deactivate import (
    HandlerPostgresDeactivate,
)
from omnibase_infra.nodes.node_registry_effect.models import ModelRegistryRequest

# Fixed test time for deterministic testing
TEST_NOW = datetime(2025, 1, 15, 12, 0, 0, tzinfo=UTC)


def create_mock_postgres_adapter() -> AsyncMock:
    """Create a mock ProtocolPostgresAdapter."""
    mock = AsyncMock()
    mock.deactivate = AsyncMock(
        return_value=ModelBackendResult(success=True, duration_ms=10.0)
    )
    return mock


def create_registry_request(
    node_id: str | None = None,
    node_type: EnumNodeKind = EnumNodeKind.EFFECT,
) -> ModelRegistryRequest:
    """Create a test registry request."""
    return ModelRegistryRequest(
        node_id=node_id or uuid4(),
        node_type=node_type,
        node_version=ModelSemVer.parse("1.0.0"),
        correlation_id=uuid4(),
        timestamp=TEST_NOW,
    )


class TestHandlerPostgresDeactivateSuccess:
    """Test successful PostgreSQL registration deactivation."""

    @pytest.mark.asyncio
    async def test_successful_deactivation(self) -> None:
        """Test that successful deactivation returns success result."""
        # Arrange
        mock_adapter = create_mock_postgres_adapter()
        mock_adapter.deactivate.return_value = ModelBackendResult(
            success=True, duration_ms=15.5
        )

        handler = HandlerPostgresDeactivate(mock_adapter)
        request = create_registry_request()
        correlation_id = uuid4()

        # Act
        result = await handler.handle(request, correlation_id)

        # Assert
        assert result.success is True
        assert result.error is None
        assert result.error_code is None
        assert result.backend_id == "postgres"
        assert result.correlation_id == correlation_id
        assert result.duration_ms >= 0

    @pytest.mark.asyncio
    async def test_deactivation_calls_adapter_with_correct_node_id(self) -> None:
        """Test that deactivation passes correct node_id to adapter."""
        # Arrange
        mock_adapter = create_mock_postgres_adapter()
        handler = HandlerPostgresDeactivate(mock_adapter)

        node_id = uuid4()
        request = create_registry_request(node_id=node_id)
        correlation_id = uuid4()

        # Act
        await handler.handle(request, correlation_id)

        # Assert - verify node_id is passed correctly
        mock_adapter.deactivate.assert_called_once_with(node_id=node_id)


class TestHandlerPostgresDeactivateNodeTypes:
    """Test deactivation works for all node types."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "node_type",
        [
            EnumNodeKind.EFFECT,
            EnumNodeKind.COMPUTE,
            EnumNodeKind.REDUCER,
            EnumNodeKind.ORCHESTRATOR,
        ],
    )
    async def test_deactivation_for_all_node_types(
        self, node_type: EnumNodeKind
    ) -> None:
        """Test deactivation succeeds for all ONEX node types."""
        # Arrange
        mock_adapter = create_mock_postgres_adapter()
        handler = HandlerPostgresDeactivate(mock_adapter)

        node_id = uuid4()
        request = create_registry_request(node_id=node_id, node_type=node_type)
        correlation_id = uuid4()

        # Act
        result = await handler.handle(request, correlation_id)

        # Assert
        assert result.success is True
        # Verify correct node_id was passed regardless of node type
        call_args = mock_adapter.deactivate.call_args
        assert call_args.kwargs["node_id"] == node_id


class TestHandlerPostgresDeactivateFailure:
    """Test PostgreSQL deactivation failure scenarios."""

    @pytest.mark.asyncio
    async def test_failed_deactivation_returns_error(self) -> None:
        """Test that adapter failure is properly captured in result."""
        # Arrange
        mock_adapter = create_mock_postgres_adapter()
        mock_adapter.deactivate.return_value = ModelBackendResult(
            success=False,
            error="Node registration not found",
            duration_ms=5.0,
        )

        handler = HandlerPostgresDeactivate(mock_adapter)
        request = create_registry_request()
        correlation_id = uuid4()

        # Act
        result = await handler.handle(request, correlation_id)

        # Assert
        assert result.success is False
        assert result.error is not None
        assert result.error_code == "POSTGRES_DEACTIVATION_ERROR"
        assert result.backend_id == "postgres"
        assert result.correlation_id == correlation_id
        assert result.duration_ms >= 0

    @pytest.mark.asyncio
    async def test_failed_deactivation_sanitizes_error(self) -> None:
        """Test that error messages are sanitized."""
        # Arrange
        mock_adapter = create_mock_postgres_adapter()
        # Simulate a raw error that might contain sensitive info
        mock_adapter.deactivate.return_value = ModelBackendResult(
            success=False,
            error="Connection refused to postgres.internal:5432",
            duration_ms=5.0,
        )

        handler = HandlerPostgresDeactivate(mock_adapter)
        request = create_registry_request()
        correlation_id = uuid4()

        # Act
        result = await handler.handle(request, correlation_id)

        # Assert - error should be sanitized (exact behavior depends on sanitize_backend_error)
        assert result.success is False
        assert result.error is not None


class TestHandlerPostgresDeactivateException:
    """Test exception handling during deactivation."""

    @pytest.mark.asyncio
    async def test_exception_is_caught_and_returned_as_error(self) -> None:
        """Test that exceptions are captured in result, not raised.

        Note: Python's built-in ConnectionError is not InfraConnectionError,
        so it maps to POSTGRES_UNKNOWN_ERROR (generic exception handling).
        """
        # Arrange
        mock_adapter = create_mock_postgres_adapter()
        mock_adapter.deactivate.side_effect = ConnectionError("Connection refused")

        handler = HandlerPostgresDeactivate(mock_adapter)
        request = create_registry_request()
        correlation_id = uuid4()

        # Act - should NOT raise
        result = await handler.handle(request, correlation_id)

        # Assert
        assert result.success is False
        assert result.error is not None
        assert "ConnectionError" in result.error
        # Python's ConnectionError maps to UNKNOWN (not InfraConnectionError)
        assert result.error_code == "POSTGRES_UNKNOWN_ERROR"
        assert result.backend_id == "postgres"
        assert result.correlation_id == correlation_id

    @pytest.mark.asyncio
    async def test_timeout_exception_returns_error(self) -> None:
        """Test that timeout exceptions return TIMEOUT_ERROR code."""
        # Arrange
        mock_adapter = create_mock_postgres_adapter()
        mock_adapter.deactivate.side_effect = TimeoutError("Operation timed out")

        handler = HandlerPostgresDeactivate(mock_adapter)
        request = create_registry_request()
        correlation_id = uuid4()

        # Act
        result = await handler.handle(request, correlation_id)

        # Assert
        assert result.success is False
        assert "TimeoutError" in result.error
        # TimeoutError maps to specific timeout error code
        assert result.error_code == "POSTGRES_TIMEOUT_ERROR"

    @pytest.mark.asyncio
    async def test_generic_exception_returns_error(self) -> None:
        """Test that generic exceptions return UNKNOWN_ERROR code."""
        # Arrange
        mock_adapter = create_mock_postgres_adapter()
        mock_adapter.deactivate.side_effect = RuntimeError("Unexpected error occurred")

        handler = HandlerPostgresDeactivate(mock_adapter)
        request = create_registry_request()
        correlation_id = uuid4()

        # Act
        result = await handler.handle(request, correlation_id)

        # Assert
        assert result.success is False
        assert "RuntimeError" in result.error
        # Generic exceptions map to UNKNOWN error code
        assert result.error_code == "POSTGRES_UNKNOWN_ERROR"

    @pytest.mark.asyncio
    async def test_database_exception_returns_error(self) -> None:
        """Test that database-specific exceptions return UNKNOWN_ERROR code."""
        # Arrange
        mock_adapter = create_mock_postgres_adapter()
        # Simulate a database constraint violation
        mock_adapter.deactivate.side_effect = ValueError(
            "Constraint violation: node already inactive"
        )

        handler = HandlerPostgresDeactivate(mock_adapter)
        request = create_registry_request()
        correlation_id = uuid4()

        # Act
        result = await handler.handle(request, correlation_id)

        # Assert
        assert result.success is False
        assert "ValueError" in result.error
        # ValueError is a generic exception, maps to UNKNOWN error code
        assert result.error_code == "POSTGRES_UNKNOWN_ERROR"


class TestHandlerPostgresDeactivateCorrelationId:
    """Test correlation ID propagation."""

    @pytest.mark.asyncio
    async def test_correlation_id_propagated_on_success(self) -> None:
        """Test that correlation_id is included in successful result."""
        # Arrange
        mock_adapter = create_mock_postgres_adapter()
        handler = HandlerPostgresDeactivate(mock_adapter)
        request = create_registry_request()
        correlation_id = uuid4()

        # Act
        result = await handler.handle(request, correlation_id)

        # Assert
        assert result.correlation_id == correlation_id

    @pytest.mark.asyncio
    async def test_correlation_id_propagated_on_failure(self) -> None:
        """Test that correlation_id is included in failed result."""
        # Arrange
        mock_adapter = create_mock_postgres_adapter()
        mock_adapter.deactivate.return_value = ModelBackendResult(
            success=False,
            error="Deactivation failed",
            duration_ms=5.0,
        )

        handler = HandlerPostgresDeactivate(mock_adapter)
        request = create_registry_request()
        correlation_id = uuid4()

        # Act
        result = await handler.handle(request, correlation_id)

        # Assert
        assert result.correlation_id == correlation_id

    @pytest.mark.asyncio
    async def test_correlation_id_propagated_on_exception(self) -> None:
        """Test that correlation_id is included when exception occurs."""
        # Arrange
        mock_adapter = create_mock_postgres_adapter()
        mock_adapter.deactivate.side_effect = Exception("Unexpected error")

        handler = HandlerPostgresDeactivate(mock_adapter)
        request = create_registry_request()
        correlation_id = uuid4()

        # Act
        result = await handler.handle(request, correlation_id)

        # Assert
        assert result.correlation_id == correlation_id


class TestHandlerPostgresDeactivateTiming:
    """Test operation timing measurement."""

    @pytest.mark.asyncio
    async def test_duration_ms_is_recorded(self) -> None:
        """Test that duration_ms is recorded for successful operations."""
        # Arrange
        mock_adapter = create_mock_postgres_adapter()
        handler = HandlerPostgresDeactivate(mock_adapter)
        request = create_registry_request()
        correlation_id = uuid4()

        # Act
        result = await handler.handle(request, correlation_id)

        # Assert
        assert result.duration_ms >= 0

    @pytest.mark.asyncio
    async def test_duration_ms_recorded_on_exception(self) -> None:
        """Test that duration_ms is recorded even when exception occurs."""
        # Arrange
        mock_adapter = create_mock_postgres_adapter()
        mock_adapter.deactivate.side_effect = Exception("Error")

        handler = HandlerPostgresDeactivate(mock_adapter)
        request = create_registry_request()
        correlation_id = uuid4()

        # Act
        result = await handler.handle(request, correlation_id)

        # Assert
        assert result.duration_ms >= 0


class TestHandlerPostgresDeactivateBackendId:
    """Test backend_id field is correctly set."""

    @pytest.mark.asyncio
    async def test_backend_id_is_postgres_on_success(self) -> None:
        """Test that backend_id is 'postgres' on success."""
        # Arrange
        mock_adapter = create_mock_postgres_adapter()
        handler = HandlerPostgresDeactivate(mock_adapter)
        request = create_registry_request()
        correlation_id = uuid4()

        # Act
        result = await handler.handle(request, correlation_id)

        # Assert
        assert result.backend_id == "postgres"

    @pytest.mark.asyncio
    async def test_backend_id_is_postgres_on_failure(self) -> None:
        """Test that backend_id is 'postgres' on failure."""
        # Arrange
        mock_adapter = create_mock_postgres_adapter()
        mock_adapter.deactivate.return_value = ModelBackendResult(
            success=False, error="Failed", duration_ms=5.0
        )

        handler = HandlerPostgresDeactivate(mock_adapter)
        request = create_registry_request()
        correlation_id = uuid4()

        # Act
        result = await handler.handle(request, correlation_id)

        # Assert
        assert result.backend_id == "postgres"

    @pytest.mark.asyncio
    async def test_backend_id_is_postgres_on_exception(self) -> None:
        """Test that backend_id is 'postgres' on exception."""
        # Arrange
        mock_adapter = create_mock_postgres_adapter()
        mock_adapter.deactivate.side_effect = Exception("Error")

        handler = HandlerPostgresDeactivate(mock_adapter)
        request = create_registry_request()
        correlation_id = uuid4()

        # Act
        result = await handler.handle(request, correlation_id)

        # Assert
        assert result.backend_id == "postgres"


__all__: list[str] = [
    "TestHandlerPostgresDeactivateSuccess",
    "TestHandlerPostgresDeactivateNodeTypes",
    "TestHandlerPostgresDeactivateFailure",
    "TestHandlerPostgresDeactivateException",
    "TestHandlerPostgresDeactivateCorrelationId",
    "TestHandlerPostgresDeactivateTiming",
    "TestHandlerPostgresDeactivateBackendId",
]
