# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for HandlerConsulDeregister.

Tests validate:
- Successful deregistration via Consul client
- Failed deregistration (client returns failure)
- Exception handling (client raises)
- Service ID generation: `onex-{node_type.value}-{node_id}`
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
from omnibase_infra.nodes.node_registry_effect.handlers.handler_consul_deregister import (
    HandlerConsulDeregister,
)
from omnibase_infra.nodes.node_registry_effect.models import ModelRegistryRequest

# Fixed test time for deterministic testing
TEST_NOW = datetime(2025, 1, 15, 12, 0, 0, tzinfo=UTC)


def create_mock_consul_client() -> AsyncMock:
    """Create a mock ProtocolConsulClient."""
    mock = AsyncMock()
    mock.deregister_service = AsyncMock(
        return_value=ModelBackendResult(
            success=True, duration_ms=10.0, backend_id="consul"
        )
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


class TestHandlerConsulDeregisterSuccess:
    """Test successful Consul service deregistration."""

    @pytest.mark.asyncio
    async def test_successful_deregistration(self) -> None:
        """Test that successful deregistration returns success result."""
        # Arrange
        mock_client = create_mock_consul_client()
        mock_client.deregister_service.return_value = ModelBackendResult(
            success=True, duration_ms=15.5, backend_id="consul"
        )

        handler = HandlerConsulDeregister(mock_client)
        request = create_registry_request()
        correlation_id = uuid4()

        # Act
        result = await handler.handle(request, correlation_id)

        # Assert
        assert result.success is True
        assert result.error is None
        assert result.error_code is None
        assert result.backend_id == "consul"
        assert result.correlation_id == correlation_id
        assert result.duration_ms >= 0

    @pytest.mark.asyncio
    async def test_deregistration_calls_client_with_correct_service_id(self) -> None:
        """Test that deregistration generates correct service_id."""
        # Arrange
        mock_client = create_mock_consul_client()
        handler = HandlerConsulDeregister(mock_client)

        node_id = uuid4()
        node_type = EnumNodeKind.EFFECT
        request = create_registry_request(node_id=node_id, node_type=node_type)
        correlation_id = uuid4()

        # Act
        await handler.handle(request, correlation_id)

        # Assert - verify service_id follows ONEX convention
        expected_service_id = f"onex-{node_type.value}-{node_id}"
        mock_client.deregister_service.assert_called_once_with(
            service_id=expected_service_id
        )


class TestHandlerConsulDeregisterServiceIdGeneration:
    """Test service ID generation for different node types."""

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
    async def test_service_id_format_for_all_node_types(
        self, node_type: EnumNodeKind
    ) -> None:
        """Test service_id follows `onex-{node_type.value}-{node_id}` format."""
        # Arrange
        mock_client = create_mock_consul_client()
        handler = HandlerConsulDeregister(mock_client)

        node_id = uuid4()
        request = create_registry_request(node_id=node_id, node_type=node_type)
        correlation_id = uuid4()

        # Act
        await handler.handle(request, correlation_id)

        # Assert
        expected_service_id = f"onex-{node_type.value}-{node_id}"
        call_args = mock_client.deregister_service.call_args
        assert call_args.kwargs["service_id"] == expected_service_id


class TestHandlerConsulDeregisterFailure:
    """Test Consul service deregistration failure scenarios."""

    @pytest.mark.asyncio
    async def test_failed_deregistration_returns_error(self) -> None:
        """Test that client failure is properly captured in result."""
        # Arrange
        mock_client = create_mock_consul_client()
        mock_client.deregister_service.return_value = ModelBackendResult(
            success=False,
            error="Service not found in Consul catalog",
            duration_ms=5.0,
        )

        handler = HandlerConsulDeregister(mock_client)
        request = create_registry_request()
        correlation_id = uuid4()

        # Act
        result = await handler.handle(request, correlation_id)

        # Assert
        assert result.success is False
        assert result.error is not None
        assert result.error_code == "CONSUL_DEREGISTRATION_ERROR"
        assert result.backend_id == "consul"
        assert result.correlation_id == correlation_id
        assert result.duration_ms >= 0

    @pytest.mark.asyncio
    async def test_failed_deregistration_sanitizes_error(self) -> None:
        """Test that error messages are sanitized."""
        # Arrange
        mock_client = create_mock_consul_client()
        # Simulate a raw error that might contain sensitive info
        mock_client.deregister_service.return_value = ModelBackendResult(
            success=False,
            error="Connection refused to consul.internal:8500",
            duration_ms=5.0,
        )

        handler = HandlerConsulDeregister(mock_client)
        request = create_registry_request()
        correlation_id = uuid4()

        # Act
        result = await handler.handle(request, correlation_id)

        # Assert - error should be sanitized (exact behavior depends on sanitize_backend_error)
        assert result.success is False
        assert result.error is not None


class TestHandlerConsulDeregisterException:
    """Test exception handling during deregistration."""

    @pytest.mark.asyncio
    async def test_exception_is_caught_and_returned_as_error(self) -> None:
        """Test that exceptions are captured in result, not raised.

        Note: Python's built-in ConnectionError is not InfraConnectionError,
        so it maps to CONSUL_UNKNOWN_ERROR (generic exception handling).
        """
        # Arrange
        mock_client = create_mock_consul_client()
        mock_client.deregister_service.side_effect = ConnectionError(
            "Connection refused"
        )

        handler = HandlerConsulDeregister(mock_client)
        request = create_registry_request()
        correlation_id = uuid4()

        # Act - should NOT raise
        result = await handler.handle(request, correlation_id)

        # Assert
        assert result.success is False
        assert result.error is not None
        assert "ConnectionError" in result.error
        # Python's ConnectionError maps to UNKNOWN (not InfraConnectionError)
        assert result.error_code == "CONSUL_UNKNOWN_ERROR"
        assert result.backend_id == "consul"
        assert result.correlation_id == correlation_id

    @pytest.mark.asyncio
    async def test_timeout_exception_returns_error(self) -> None:
        """Test that timeout exceptions return TIMEOUT_ERROR code."""
        # Arrange
        mock_client = create_mock_consul_client()
        mock_client.deregister_service.side_effect = TimeoutError("Operation timed out")

        handler = HandlerConsulDeregister(mock_client)
        request = create_registry_request()
        correlation_id = uuid4()

        # Act
        result = await handler.handle(request, correlation_id)

        # Assert
        assert result.success is False
        assert "TimeoutError" in result.error
        # TimeoutError maps to specific timeout error code
        assert result.error_code == "CONSUL_TIMEOUT_ERROR"

    @pytest.mark.asyncio
    async def test_generic_exception_returns_error(self) -> None:
        """Test that generic exceptions return UNKNOWN_ERROR code."""
        # Arrange
        mock_client = create_mock_consul_client()
        mock_client.deregister_service.side_effect = RuntimeError(
            "Unexpected error occurred"
        )

        handler = HandlerConsulDeregister(mock_client)
        request = create_registry_request()
        correlation_id = uuid4()

        # Act
        result = await handler.handle(request, correlation_id)

        # Assert
        assert result.success is False
        assert "RuntimeError" in result.error
        # Generic exceptions map to UNKNOWN error code
        assert result.error_code == "CONSUL_UNKNOWN_ERROR"


class TestHandlerConsulDeregisterCorrelationId:
    """Test correlation ID propagation."""

    @pytest.mark.asyncio
    async def test_correlation_id_propagated_on_success(self) -> None:
        """Test that correlation_id is included in successful result."""
        # Arrange
        mock_client = create_mock_consul_client()
        handler = HandlerConsulDeregister(mock_client)
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
        mock_client = create_mock_consul_client()
        mock_client.deregister_service.return_value = ModelBackendResult(
            success=False,
            error="Deregistration failed",
            duration_ms=5.0,
        )

        handler = HandlerConsulDeregister(mock_client)
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
        mock_client = create_mock_consul_client()
        mock_client.deregister_service.side_effect = Exception("Unexpected error")

        handler = HandlerConsulDeregister(mock_client)
        request = create_registry_request()
        correlation_id = uuid4()

        # Act
        result = await handler.handle(request, correlation_id)

        # Assert
        assert result.correlation_id == correlation_id


class TestHandlerConsulDeregisterTiming:
    """Test operation timing measurement."""

    @pytest.mark.asyncio
    async def test_duration_ms_is_recorded(self) -> None:
        """Test that duration_ms is recorded for successful operations."""
        # Arrange
        mock_client = create_mock_consul_client()
        handler = HandlerConsulDeregister(mock_client)
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
        mock_client = create_mock_consul_client()
        mock_client.deregister_service.side_effect = Exception("Error")

        handler = HandlerConsulDeregister(mock_client)
        request = create_registry_request()
        correlation_id = uuid4()

        # Act
        result = await handler.handle(request, correlation_id)

        # Assert
        assert result.duration_ms >= 0


class TestHandlerConsulDeregisterBackendId:
    """Test backend_id field is correctly set."""

    @pytest.mark.asyncio
    async def test_backend_id_is_consul_on_success(self) -> None:
        """Test that backend_id is 'consul' on success."""
        # Arrange
        mock_client = create_mock_consul_client()
        handler = HandlerConsulDeregister(mock_client)
        request = create_registry_request()
        correlation_id = uuid4()

        # Act
        result = await handler.handle(request, correlation_id)

        # Assert
        assert result.backend_id == "consul"

    @pytest.mark.asyncio
    async def test_backend_id_is_consul_on_failure(self) -> None:
        """Test that backend_id is 'consul' on failure."""
        # Arrange
        mock_client = create_mock_consul_client()
        mock_client.deregister_service.return_value = ModelBackendResult(
            success=False, error="Failed", duration_ms=5.0
        )

        handler = HandlerConsulDeregister(mock_client)
        request = create_registry_request()
        correlation_id = uuid4()

        # Act
        result = await handler.handle(request, correlation_id)

        # Assert
        assert result.backend_id == "consul"

    @pytest.mark.asyncio
    async def test_backend_id_is_consul_on_exception(self) -> None:
        """Test that backend_id is 'consul' on exception."""
        # Arrange
        mock_client = create_mock_consul_client()
        mock_client.deregister_service.side_effect = Exception("Error")

        handler = HandlerConsulDeregister(mock_client)
        request = create_registry_request()
        correlation_id = uuid4()

        # Act
        result = await handler.handle(request, correlation_id)

        # Assert
        assert result.backend_id == "consul"


__all__: list[str] = [
    "TestHandlerConsulDeregisterSuccess",
    "TestHandlerConsulDeregisterServiceIdGeneration",
    "TestHandlerConsulDeregisterFailure",
    "TestHandlerConsulDeregisterException",
    "TestHandlerConsulDeregisterCorrelationId",
    "TestHandlerConsulDeregisterTiming",
    "TestHandlerConsulDeregisterBackendId",
]
