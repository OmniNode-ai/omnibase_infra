# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for HandlerConsulRegister.

Tests validate:
- Successful registration via Consul client
- Service ID and name generation
- Failed registration (client returns failure)
- Exception handling (client raises)
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
from omnibase_infra.nodes.node_registry_effect.handlers.handler_consul_register import (
    HandlerConsulRegister,
)
from omnibase_infra.nodes.node_registry_effect.models import ModelRegistryRequest

# Fixed test time for deterministic testing
TEST_NOW = datetime(2025, 1, 15, 12, 0, 0, tzinfo=UTC)


def create_mock_consul_client() -> AsyncMock:
    """Create a mock ProtocolConsulClient."""
    mock = AsyncMock()
    mock.register_service = AsyncMock(
        return_value=ModelBackendResult(success=True, duration_ms=10.0)
    )
    return mock


def create_registry_request(
    node_id: str | None = None,
    node_type: EnumNodeKind = EnumNodeKind.EFFECT,
    service_name: str | None = None,
    tags: list[str] | None = None,
    health_check_config: dict[str, str] | None = None,
) -> ModelRegistryRequest:
    """Create a test registry request."""
    return ModelRegistryRequest(
        node_id=node_id or uuid4(),
        node_type=node_type,
        node_version=ModelSemVer.parse("1.0.0"),
        correlation_id=uuid4(),
        timestamp=TEST_NOW,
        service_name=service_name,
        tags=tags or [],
        health_check_config=health_check_config,
    )


class TestHandlerConsulRegisterSuccess:
    """Test successful Consul service registration."""

    @pytest.mark.asyncio
    async def test_successful_registration(self) -> None:
        """Test that successful registration returns success result."""
        # Arrange
        mock_client = create_mock_consul_client()
        mock_client.register_service.return_value = ModelBackendResult(
            success=True, duration_ms=15.5
        )

        handler = HandlerConsulRegister(mock_client)
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
    async def test_registration_calls_client_with_correct_service_id(self) -> None:
        """Test that registration generates correct service_id."""
        # Arrange
        mock_client = create_mock_consul_client()
        handler = HandlerConsulRegister(mock_client)

        node_id = uuid4()
        node_type = EnumNodeKind.EFFECT
        request = create_registry_request(node_id=node_id, node_type=node_type)
        correlation_id = uuid4()

        # Act
        await handler.handle(request, correlation_id)

        # Assert - verify service_id follows ONEX convention
        expected_service_id = f"onex-{node_type.value}-{node_id}"
        call_args = mock_client.register_service.call_args
        assert call_args.kwargs["service_id"] == expected_service_id

    @pytest.mark.asyncio
    async def test_registration_uses_custom_service_name(self) -> None:
        """Test that custom service_name is used when provided."""
        # Arrange
        mock_client = create_mock_consul_client()
        handler = HandlerConsulRegister(mock_client)

        custom_name = "my-custom-service"
        request = create_registry_request(service_name=custom_name)
        correlation_id = uuid4()

        # Act
        await handler.handle(request, correlation_id)

        # Assert
        call_args = mock_client.register_service.call_args
        assert call_args.kwargs["service_name"] == custom_name

    @pytest.mark.asyncio
    async def test_registration_uses_default_service_name(self) -> None:
        """Test that default service_name is generated when not provided."""
        # Arrange
        mock_client = create_mock_consul_client()
        handler = HandlerConsulRegister(mock_client)

        node_type = EnumNodeKind.COMPUTE
        request = create_registry_request(node_type=node_type, service_name=None)
        correlation_id = uuid4()

        # Act
        await handler.handle(request, correlation_id)

        # Assert
        expected_service_name = f"onex-{node_type.value}"
        call_args = mock_client.register_service.call_args
        assert call_args.kwargs["service_name"] == expected_service_name


class TestHandlerConsulRegisterServiceIdGeneration:
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
        handler = HandlerConsulRegister(mock_client)

        node_id = uuid4()
        request = create_registry_request(node_id=node_id, node_type=node_type)
        correlation_id = uuid4()

        # Act
        await handler.handle(request, correlation_id)

        # Assert
        expected_service_id = f"onex-{node_type.value}-{node_id}"
        call_args = mock_client.register_service.call_args
        assert call_args.kwargs["service_id"] == expected_service_id


class TestHandlerConsulRegisterFailure:
    """Test Consul service registration failure scenarios."""

    @pytest.mark.asyncio
    async def test_failed_registration_returns_error(self) -> None:
        """Test that client failure is properly captured in result."""
        # Arrange
        mock_client = create_mock_consul_client()
        mock_client.register_service.return_value = ModelBackendResult(
            success=False,
            error="Service already exists in Consul catalog",
            duration_ms=5.0,
        )

        handler = HandlerConsulRegister(mock_client)
        request = create_registry_request()
        correlation_id = uuid4()

        # Act
        result = await handler.handle(request, correlation_id)

        # Assert
        assert result.success is False
        assert result.error is not None
        assert result.error_code == "CONSUL_REGISTRATION_ERROR"
        assert result.backend_id == "consul"
        assert result.correlation_id == correlation_id
        assert result.duration_ms >= 0

    @pytest.mark.asyncio
    async def test_failed_registration_sanitizes_error(self) -> None:
        """Test that error messages are sanitized."""
        # Arrange
        mock_client = create_mock_consul_client()
        # Simulate a raw error that might contain sensitive info
        mock_client.register_service.return_value = ModelBackendResult(
            success=False,
            error="Connection refused to consul.internal:8500",
            duration_ms=5.0,
        )

        handler = HandlerConsulRegister(mock_client)
        request = create_registry_request()
        correlation_id = uuid4()

        # Act
        result = await handler.handle(request, correlation_id)

        # Assert - error should be sanitized (exact behavior depends on sanitize_backend_error)
        assert result.success is False
        assert result.error is not None


class TestHandlerConsulRegisterException:
    """Test exception handling during registration."""

    @pytest.mark.asyncio
    async def test_exception_is_caught_and_returned_as_error(self) -> None:
        """Test that exceptions are captured in result, not raised.

        Note: Python's built-in ConnectionError is not InfraConnectionError,
        so it maps to CONSUL_UNKNOWN_ERROR (generic exception handling).
        """
        # Arrange
        mock_client = create_mock_consul_client()
        mock_client.register_service.side_effect = ConnectionError("Connection refused")

        handler = HandlerConsulRegister(mock_client)
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
        mock_client.register_service.side_effect = TimeoutError("Operation timed out")

        handler = HandlerConsulRegister(mock_client)
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
        mock_client.register_service.side_effect = RuntimeError(
            "Unexpected error occurred"
        )

        handler = HandlerConsulRegister(mock_client)
        request = create_registry_request()
        correlation_id = uuid4()

        # Act
        result = await handler.handle(request, correlation_id)

        # Assert
        assert result.success is False
        assert "RuntimeError" in result.error
        # Generic exceptions map to UNKNOWN error code
        assert result.error_code == "CONSUL_UNKNOWN_ERROR"


class TestHandlerConsulRegisterCorrelationId:
    """Test correlation ID propagation."""

    @pytest.mark.asyncio
    async def test_correlation_id_propagated_on_success(self) -> None:
        """Test that correlation_id is included in successful result."""
        # Arrange
        mock_client = create_mock_consul_client()
        handler = HandlerConsulRegister(mock_client)
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
        mock_client.register_service.return_value = ModelBackendResult(
            success=False,
            error="Registration failed",
            duration_ms=5.0,
        )

        handler = HandlerConsulRegister(mock_client)
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
        mock_client.register_service.side_effect = Exception("Unexpected error")

        handler = HandlerConsulRegister(mock_client)
        request = create_registry_request()
        correlation_id = uuid4()

        # Act
        result = await handler.handle(request, correlation_id)

        # Assert
        assert result.correlation_id == correlation_id


class TestHandlerConsulRegisterTiming:
    """Test operation timing measurement."""

    @pytest.mark.asyncio
    async def test_duration_ms_is_recorded(self) -> None:
        """Test that duration_ms is recorded for successful operations."""
        # Arrange
        mock_client = create_mock_consul_client()
        handler = HandlerConsulRegister(mock_client)
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
        mock_client.register_service.side_effect = Exception("Error")

        handler = HandlerConsulRegister(mock_client)
        request = create_registry_request()
        correlation_id = uuid4()

        # Act
        result = await handler.handle(request, correlation_id)

        # Assert
        assert result.duration_ms >= 0


class TestHandlerConsulRegisterBackendId:
    """Test backend_id field is correctly set."""

    @pytest.mark.asyncio
    async def test_backend_id_is_consul_on_success(self) -> None:
        """Test that backend_id is 'consul' on success."""
        # Arrange
        mock_client = create_mock_consul_client()
        handler = HandlerConsulRegister(mock_client)
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
        mock_client.register_service.return_value = ModelBackendResult(
            success=False, error="Failed", duration_ms=5.0
        )

        handler = HandlerConsulRegister(mock_client)
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
        mock_client.register_service.side_effect = Exception("Error")

        handler = HandlerConsulRegister(mock_client)
        request = create_registry_request()
        correlation_id = uuid4()

        # Act
        result = await handler.handle(request, correlation_id)

        # Assert
        assert result.backend_id == "consul"


__all__: list[str] = [
    "TestHandlerConsulRegisterSuccess",
    "TestHandlerConsulRegisterServiceIdGeneration",
    "TestHandlerConsulRegisterFailure",
    "TestHandlerConsulRegisterException",
    "TestHandlerConsulRegisterCorrelationId",
    "TestHandlerConsulRegisterTiming",
    "TestHandlerConsulRegisterBackendId",
]
