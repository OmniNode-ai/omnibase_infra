# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for HandlerPartialRetry.

Tests validate:
- Consul retry routing and execution
- PostgreSQL retry routing and execution
- Invalid backend handling
- Exception handling for both backends
- Correlation ID propagation
- Idempotency key handling

Related Tickets:
    - OMN-1103: NodeRegistryEffect refactoring to declarative pattern
    - OMN-954: Partial failure scenario testing
"""

from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest

from omnibase_core.enums.enum_node_kind import EnumNodeKind
from omnibase_core.models.primitives import ModelSemVer
from omnibase_infra.enums import EnumBackendType
from omnibase_infra.nodes.effects.models import ModelBackendResult
from omnibase_infra.nodes.node_registry_effect.handlers.handler_partial_retry import (
    HandlerPartialRetry,
)


@dataclass
class MockPartialRetryRequest:
    """Mock request object implementing ProtocolPartialRetryRequest."""

    node_id: UUID
    node_type: EnumNodeKind
    node_version: ModelSemVer
    target_backend: EnumBackendType
    idempotency_key: str | None = None
    service_name: str | None = None
    tags: list[str] = field(default_factory=list)
    health_check_config: dict[str, str] | None = None
    endpoints: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, str] = field(default_factory=dict)


def create_mock_consul_client() -> AsyncMock:
    """Create a mock ProtocolConsulClient."""
    mock = AsyncMock()
    mock.register_service = AsyncMock(
        return_value=ModelBackendResult(
            success=True, duration_ms=10.0, backend_id="consul"
        )
    )
    return mock


def create_mock_postgres_adapter() -> AsyncMock:
    """Create a mock ProtocolPostgresAdapter."""
    mock = AsyncMock()
    mock.upsert = AsyncMock(
        return_value=ModelBackendResult(
            success=True, duration_ms=10.0, backend_id="postgres"
        )
    )
    return mock


def create_retry_request(
    target_backend: EnumBackendType = EnumBackendType.CONSUL,
    node_id: UUID | None = None,
    node_type: EnumNodeKind = EnumNodeKind.EFFECT,
    idempotency_key: str | None = None,
    service_name: str | None = None,
    tags: list[str] | None = None,
    endpoints: dict[str, str] | None = None,
    metadata: dict[str, str] | None = None,
) -> MockPartialRetryRequest:
    """Create a test partial retry request."""
    return MockPartialRetryRequest(
        node_id=node_id or uuid4(),
        node_type=node_type,
        node_version=ModelSemVer(major=1, minor=0, patch=0),
        target_backend=target_backend,
        idempotency_key=idempotency_key,
        service_name=service_name,
        tags=tags or [],
        health_check_config=None,
        endpoints=endpoints or {"http": "http://localhost:8080"},
        metadata=metadata or {"environment": "test"},
    )


class TestHandlerPartialRetryConsulSuccess:
    """Test successful Consul retry operations."""

    @pytest.mark.asyncio
    async def test_consul_retry_success(self) -> None:
        """Test that successful Consul retry returns success result."""
        # Arrange
        mock_consul = create_mock_consul_client()
        mock_postgres = create_mock_postgres_adapter()

        handler = HandlerPartialRetry(mock_consul, mock_postgres)
        request = create_retry_request(target_backend=EnumBackendType.CONSUL)
        correlation_id = uuid4()

        # Act
        result = await handler.handle(request, correlation_id)

        # Assert
        assert result.success is True
        assert result.error is None
        assert result.backend_id == "consul"
        assert result.correlation_id == correlation_id
        mock_consul.register_service.assert_called_once()
        mock_postgres.upsert.assert_not_called()

    @pytest.mark.asyncio
    async def test_consul_retry_generates_correct_service_id(self) -> None:
        """Test that Consul retry generates correct service_id."""
        # Arrange
        mock_consul = create_mock_consul_client()
        mock_postgres = create_mock_postgres_adapter()

        handler = HandlerPartialRetry(mock_consul, mock_postgres)
        node_id = uuid4()
        node_type = EnumNodeKind.EFFECT
        request = create_retry_request(
            target_backend=EnumBackendType.CONSUL, node_id=node_id, node_type=node_type
        )
        correlation_id = uuid4()

        # Act
        await handler.handle(request, correlation_id)

        # Assert
        expected_service_id = f"onex-{node_type.value}-{node_id}"
        call_args = mock_consul.register_service.call_args
        assert call_args.kwargs["service_id"] == expected_service_id


class TestHandlerPartialRetryPostgresSuccess:
    """Test successful PostgreSQL retry operations."""

    @pytest.mark.asyncio
    async def test_postgres_retry_success(self) -> None:
        """Test that successful PostgreSQL retry returns success result."""
        # Arrange
        mock_consul = create_mock_consul_client()
        mock_postgres = create_mock_postgres_adapter()

        handler = HandlerPartialRetry(mock_consul, mock_postgres)
        request = create_retry_request(target_backend=EnumBackendType.POSTGRES)
        correlation_id = uuid4()

        # Act
        result = await handler.handle(request, correlation_id)

        # Assert
        assert result.success is True
        assert result.error is None
        assert result.backend_id == "postgres"
        assert result.correlation_id == correlation_id
        mock_postgres.upsert.assert_called_once()
        mock_consul.register_service.assert_not_called()

    @pytest.mark.asyncio
    async def test_postgres_retry_passes_correct_parameters(self) -> None:
        """Test that PostgreSQL retry passes correct parameters."""
        # Arrange
        mock_consul = create_mock_consul_client()
        mock_postgres = create_mock_postgres_adapter()

        handler = HandlerPartialRetry(mock_consul, mock_postgres)
        node_id = uuid4()
        node_type = EnumNodeKind.COMPUTE
        node_version = ModelSemVer(major=2, minor=0, patch=0)
        endpoints = {"grpc": "grpc://localhost:9090"}
        metadata = {"region": "us-west"}
        request = create_retry_request(
            target_backend=EnumBackendType.POSTGRES,
            node_id=node_id,
            node_type=node_type,
            endpoints=endpoints,
            metadata=metadata,
        )
        request.node_version = node_version
        correlation_id = uuid4()

        # Act
        await handler.handle(request, correlation_id)

        # Assert
        # Handler passes ModelSemVer directly to postgres adapter
        mock_postgres.upsert.assert_called_once_with(
            node_id=node_id,
            node_type=node_type,
            node_version=node_version,
            endpoints=endpoints,
            metadata=metadata,
        )


class TestHandlerPartialRetryInvalidBackend:
    """Test handling of invalid backend values."""

    @pytest.mark.asyncio
    async def test_invalid_backend_returns_error(self) -> None:
        """Test that invalid backend returns appropriate error."""
        # Arrange
        mock_consul = create_mock_consul_client()
        mock_postgres = create_mock_postgres_adapter()

        handler = HandlerPartialRetry(mock_consul, mock_postgres)
        # Create request with invalid backend (type: ignore needed for test)
        request = create_retry_request(target_backend=EnumBackendType.CONSUL)
        request.target_backend = "invalid_backend"  # type: ignore[assignment]
        correlation_id = uuid4()

        # Act
        result = await handler.handle(request, correlation_id)

        # Assert
        assert result.success is False
        assert result.error is not None
        assert "Unknown target backend" in result.error
        assert result.error_code == "INVALID_TARGET_BACKEND"
        assert result.backend_id == "invalid_backend"
        mock_consul.register_service.assert_not_called()
        mock_postgres.upsert.assert_not_called()


class TestHandlerPartialRetryConsulFailure:
    """Test Consul retry failure scenarios."""

    @pytest.mark.asyncio
    async def test_consul_retry_failure_returns_error(self) -> None:
        """Test that Consul client failure is properly captured."""
        # Arrange
        mock_consul = create_mock_consul_client()
        mock_consul.register_service.return_value = ModelBackendResult(
            success=False,
            error="Service registration failed",
            duration_ms=5.0,
        )
        mock_postgres = create_mock_postgres_adapter()

        handler = HandlerPartialRetry(mock_consul, mock_postgres)
        request = create_retry_request(target_backend=EnumBackendType.CONSUL)
        correlation_id = uuid4()

        # Act
        result = await handler.handle(request, correlation_id)

        # Assert
        assert result.success is False
        assert result.error is not None
        assert result.error_code == "CONSUL_REGISTRATION_ERROR"
        assert result.backend_id == "consul"

    @pytest.mark.asyncio
    async def test_consul_retry_timeout_exception(self) -> None:
        """Test that timeout exceptions during Consul retry return TIMEOUT_ERROR."""
        # Arrange
        mock_consul = create_mock_consul_client()
        mock_consul.register_service.side_effect = TimeoutError("Connection timed out")
        mock_postgres = create_mock_postgres_adapter()

        handler = HandlerPartialRetry(mock_consul, mock_postgres)
        request = create_retry_request(target_backend=EnumBackendType.CONSUL)
        correlation_id = uuid4()

        # Act
        result = await handler.handle(request, correlation_id)

        # Assert
        assert result.success is False
        assert "TimeoutError" in result.error
        assert result.error_code == "CONSUL_TIMEOUT_ERROR"
        assert result.backend_id == "consul"


class TestHandlerPartialRetryPostgresFailure:
    """Test PostgreSQL retry failure scenarios."""

    @pytest.mark.asyncio
    async def test_postgres_retry_failure_returns_error(self) -> None:
        """Test that PostgreSQL adapter failure is properly captured."""
        # Arrange
        mock_consul = create_mock_consul_client()
        mock_postgres = create_mock_postgres_adapter()
        mock_postgres.upsert.return_value = ModelBackendResult(
            success=False,
            error="Database constraint violation",
            duration_ms=5.0,
        )

        handler = HandlerPartialRetry(mock_consul, mock_postgres)
        request = create_retry_request(target_backend=EnumBackendType.POSTGRES)
        correlation_id = uuid4()

        # Act
        result = await handler.handle(request, correlation_id)

        # Assert
        assert result.success is False
        assert result.error is not None
        assert result.error_code == "POSTGRES_UPSERT_ERROR"
        assert result.backend_id == "postgres"

    @pytest.mark.asyncio
    async def test_postgres_retry_timeout_exception(self) -> None:
        """Test that timeout exceptions during PostgreSQL retry return TIMEOUT_ERROR."""
        # Arrange
        mock_consul = create_mock_consul_client()
        mock_postgres = create_mock_postgres_adapter()
        mock_postgres.upsert.side_effect = TimeoutError("Query timed out")

        handler = HandlerPartialRetry(mock_consul, mock_postgres)
        request = create_retry_request(target_backend=EnumBackendType.POSTGRES)
        correlation_id = uuid4()

        # Act
        result = await handler.handle(request, correlation_id)

        # Assert
        assert result.success is False
        assert "TimeoutError" in result.error
        assert result.error_code == "POSTGRES_TIMEOUT_ERROR"
        assert result.backend_id == "postgres"


class TestHandlerPartialRetryExceptionHandling:
    """Test generic exception handling for both backends."""

    @pytest.mark.asyncio
    async def test_consul_generic_exception_returns_unknown_error(self) -> None:
        """Test that generic exceptions during Consul retry return UNKNOWN_ERROR."""
        # Arrange
        mock_consul = create_mock_consul_client()
        mock_consul.register_service.side_effect = RuntimeError("Unexpected error")
        mock_postgres = create_mock_postgres_adapter()

        handler = HandlerPartialRetry(mock_consul, mock_postgres)
        request = create_retry_request(target_backend=EnumBackendType.CONSUL)
        correlation_id = uuid4()

        # Act
        result = await handler.handle(request, correlation_id)

        # Assert
        assert result.success is False
        assert "RuntimeError" in result.error
        assert result.error_code == "CONSUL_UNKNOWN_ERROR"

    @pytest.mark.asyncio
    async def test_postgres_generic_exception_returns_unknown_error(self) -> None:
        """Test that generic exceptions during PostgreSQL retry return UNKNOWN_ERROR."""
        # Arrange
        mock_consul = create_mock_consul_client()
        mock_postgres = create_mock_postgres_adapter()
        mock_postgres.upsert.side_effect = RuntimeError("Unexpected error")

        handler = HandlerPartialRetry(mock_consul, mock_postgres)
        request = create_retry_request(target_backend=EnumBackendType.POSTGRES)
        correlation_id = uuid4()

        # Act
        result = await handler.handle(request, correlation_id)

        # Assert
        assert result.success is False
        assert "RuntimeError" in result.error
        assert result.error_code == "POSTGRES_UNKNOWN_ERROR"


class TestHandlerPartialRetryCorrelationId:
    """Test correlation ID propagation."""

    @pytest.mark.asyncio
    async def test_correlation_id_propagated_consul_success(self) -> None:
        """Test correlation_id is included in successful Consul result."""
        # Arrange
        mock_consul = create_mock_consul_client()
        mock_postgres = create_mock_postgres_adapter()

        handler = HandlerPartialRetry(mock_consul, mock_postgres)
        request = create_retry_request(target_backend=EnumBackendType.CONSUL)
        correlation_id = uuid4()

        # Act
        result = await handler.handle(request, correlation_id)

        # Assert
        assert result.correlation_id == correlation_id

    @pytest.mark.asyncio
    async def test_correlation_id_propagated_postgres_success(self) -> None:
        """Test correlation_id is included in successful PostgreSQL result."""
        # Arrange
        mock_consul = create_mock_consul_client()
        mock_postgres = create_mock_postgres_adapter()

        handler = HandlerPartialRetry(mock_consul, mock_postgres)
        request = create_retry_request(target_backend=EnumBackendType.POSTGRES)
        correlation_id = uuid4()

        # Act
        result = await handler.handle(request, correlation_id)

        # Assert
        assert result.correlation_id == correlation_id

    @pytest.mark.asyncio
    async def test_correlation_id_propagated_on_failure(self) -> None:
        """Test correlation_id is included in failed result."""
        # Arrange
        mock_consul = create_mock_consul_client()
        mock_consul.register_service.return_value = ModelBackendResult(
            success=False, error="Failed", duration_ms=5.0
        )
        mock_postgres = create_mock_postgres_adapter()

        handler = HandlerPartialRetry(mock_consul, mock_postgres)
        request = create_retry_request(target_backend=EnumBackendType.CONSUL)
        correlation_id = uuid4()

        # Act
        result = await handler.handle(request, correlation_id)

        # Assert
        assert result.correlation_id == correlation_id

    @pytest.mark.asyncio
    async def test_correlation_id_propagated_on_exception(self) -> None:
        """Test correlation_id is included when exception occurs."""
        # Arrange
        mock_consul = create_mock_consul_client()
        mock_consul.register_service.side_effect = Exception("Error")
        mock_postgres = create_mock_postgres_adapter()

        handler = HandlerPartialRetry(mock_consul, mock_postgres)
        request = create_retry_request(target_backend=EnumBackendType.CONSUL)
        correlation_id = uuid4()

        # Act
        result = await handler.handle(request, correlation_id)

        # Assert
        assert result.correlation_id == correlation_id


class TestHandlerPartialRetryTiming:
    """Test operation timing measurement."""

    @pytest.mark.asyncio
    async def test_duration_ms_recorded_consul(self) -> None:
        """Test duration_ms is recorded for Consul operations."""
        # Arrange
        mock_consul = create_mock_consul_client()
        mock_postgres = create_mock_postgres_adapter()

        handler = HandlerPartialRetry(mock_consul, mock_postgres)
        request = create_retry_request(target_backend=EnumBackendType.CONSUL)
        correlation_id = uuid4()

        # Act
        result = await handler.handle(request, correlation_id)

        # Assert
        assert result.duration_ms >= 0

    @pytest.mark.asyncio
    async def test_duration_ms_recorded_postgres(self) -> None:
        """Test duration_ms is recorded for PostgreSQL operations."""
        # Arrange
        mock_consul = create_mock_consul_client()
        mock_postgres = create_mock_postgres_adapter()

        handler = HandlerPartialRetry(mock_consul, mock_postgres)
        request = create_retry_request(target_backend=EnumBackendType.POSTGRES)
        correlation_id = uuid4()

        # Act
        result = await handler.handle(request, correlation_id)

        # Assert
        assert result.duration_ms >= 0

    @pytest.mark.asyncio
    async def test_duration_ms_recorded_on_exception(self) -> None:
        """Test duration_ms is recorded even when exception occurs."""
        # Arrange
        mock_consul = create_mock_consul_client()
        mock_consul.register_service.side_effect = Exception("Error")
        mock_postgres = create_mock_postgres_adapter()

        handler = HandlerPartialRetry(mock_consul, mock_postgres)
        request = create_retry_request(target_backend=EnumBackendType.CONSUL)
        correlation_id = uuid4()

        # Act
        result = await handler.handle(request, correlation_id)

        # Assert
        assert result.duration_ms >= 0


class TestHandlerPartialRetryNodeTypes:
    """Test partial retry works for all node types."""

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
    async def test_consul_retry_for_all_node_types(
        self, node_type: EnumNodeKind
    ) -> None:
        """Test Consul retry succeeds for all ONEX node types."""
        # Arrange
        mock_consul = create_mock_consul_client()
        mock_postgres = create_mock_postgres_adapter()

        handler = HandlerPartialRetry(mock_consul, mock_postgres)
        node_id = uuid4()
        request = create_retry_request(
            target_backend=EnumBackendType.CONSUL, node_id=node_id, node_type=node_type
        )
        correlation_id = uuid4()

        # Act
        result = await handler.handle(request, correlation_id)

        # Assert
        assert result.success is True
        expected_service_id = f"onex-{node_type.value}-{node_id}"
        call_args = mock_consul.register_service.call_args
        assert call_args.kwargs["service_id"] == expected_service_id

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
    async def test_postgres_retry_for_all_node_types(
        self, node_type: EnumNodeKind
    ) -> None:
        """Test PostgreSQL retry succeeds for all ONEX node types."""
        # Arrange
        mock_consul = create_mock_consul_client()
        mock_postgres = create_mock_postgres_adapter()

        handler = HandlerPartialRetry(mock_consul, mock_postgres)
        node_id = uuid4()
        request = create_retry_request(
            target_backend=EnumBackendType.POSTGRES,
            node_id=node_id,
            node_type=node_type,
        )
        correlation_id = uuid4()

        # Act
        result = await handler.handle(request, correlation_id)

        # Assert
        assert result.success is True
        call_args = mock_postgres.upsert.call_args
        assert call_args.kwargs["node_type"] == node_type


class TestHandlerPartialRetryIdempotencyKey:
    """Test idempotency key handling."""

    @pytest.mark.asyncio
    async def test_idempotency_key_present_in_request(self) -> None:
        """Test that handler accepts requests with idempotency_key."""
        # Arrange
        mock_consul = create_mock_consul_client()
        mock_postgres = create_mock_postgres_adapter()

        handler = HandlerPartialRetry(mock_consul, mock_postgres)
        idempotency_key = f"retry-{uuid4()}"
        request = create_retry_request(
            target_backend=EnumBackendType.CONSUL, idempotency_key=idempotency_key
        )
        correlation_id = uuid4()

        # Act
        result = await handler.handle(request, correlation_id)

        # Assert
        assert result.success is True
        # The idempotency_key is part of the request but enforcement is external

    @pytest.mark.asyncio
    async def test_idempotency_key_none_is_valid(self) -> None:
        """Test that handler accepts requests without idempotency_key."""
        # Arrange
        mock_consul = create_mock_consul_client()
        mock_postgres = create_mock_postgres_adapter()

        handler = HandlerPartialRetry(mock_consul, mock_postgres)
        request = create_retry_request(
            target_backend=EnumBackendType.CONSUL, idempotency_key=None
        )
        correlation_id = uuid4()

        # Act
        result = await handler.handle(request, correlation_id)

        # Assert
        assert result.success is True


__all__: list[str] = [
    "TestHandlerPartialRetryConsulSuccess",
    "TestHandlerPartialRetryPostgresSuccess",
    "TestHandlerPartialRetryInvalidBackend",
    "TestHandlerPartialRetryConsulFailure",
    "TestHandlerPartialRetryPostgresFailure",
    "TestHandlerPartialRetryExceptionHandling",
    "TestHandlerPartialRetryCorrelationId",
    "TestHandlerPartialRetryTiming",
    "TestHandlerPartialRetryNodeTypes",
    "TestHandlerPartialRetryIdempotencyKey",
]
