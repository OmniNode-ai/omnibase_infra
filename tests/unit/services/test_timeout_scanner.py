# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Comprehensive unit tests for ServiceTimeoutScanner.

This test suite validates:
- Processor instantiation with projection reader
- Combined timeout query (find_overdue_entities)
- Separate ack timeout query (find_ack_timeouts)
- Separate liveness expiration query (find_liveness_expirations)
- Result model properties and metadata
- Batch size configuration
- Correlation ID propagation
- Error handling (circuit breaker integration)
- Query timing accuracy

Test Organization:
    - TestServiceTimeoutScannerBasics: Instantiation and configuration
    - TestServiceTimeoutScannerFindOverdue: Combined query tests
    - TestServiceTimeoutScannerAckTimeouts: Ack-specific query tests
    - TestServiceTimeoutScannerLivenessExpirations: Liveness-specific query tests
    - TestModelTimeoutQueryResult: Result model tests
    - TestServiceTimeoutScannerErrorHandling: Error scenarios

Coverage Goals:
    - >90% code coverage for processor
    - All query paths tested
    - Error handling validated
    - Timing metadata verified

Related Tickets:
    - OMN-932 (C2): Durable Timeout Handling
    - OMN-944 (F1): Implement Registration Projection Schema
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from pydantic import ValidationError

from omnibase_core.container import ModelONEXContainer
from omnibase_core.models.primitives.model_semver import ModelSemVer
from omnibase_infra.enums import EnumInfraTransportType, EnumRegistrationState
from omnibase_infra.errors import (
    InfraConnectionError,
    InfraTimeoutError,
    InfraUnavailableError,
    ModelTimeoutErrorContext,
)
from omnibase_infra.models.projection import ModelRegistrationProjection
from omnibase_infra.models.registration.model_node_capabilities import (
    ModelNodeCapabilities,
)
from omnibase_infra.services import (
    ModelTimeoutQueryResult,
    ServiceTimeoutScanner,
)

# =============================================================================
# Test Constants
# =============================================================================

# Time offsets for deadline scenarios
ACK_TIMEOUT_OFFSET = timedelta(minutes=5)
"""Offset for creating past/overdue ack deadlines in tests."""

LIVENESS_TIMEOUT_OFFSET = timedelta(minutes=10)
"""Offset for creating past/overdue liveness deadlines in tests."""

# Query duration bounds for validation
MAX_REASONABLE_QUERY_DURATION_MS = 10000.0
"""Maximum reasonable query duration in milliseconds (10 seconds)."""


def create_mock_projection(
    state: EnumRegistrationState = EnumRegistrationState.ACTIVE,
    ack_deadline: datetime | None = None,
    liveness_deadline: datetime | None = None,
    ack_timeout_emitted_at: datetime | None = None,
    liveness_timeout_emitted_at: datetime | None = None,
) -> ModelRegistrationProjection:
    """Create a mock projection with sensible defaults."""
    now = datetime.now(UTC)
    return ModelRegistrationProjection(
        entity_id=uuid4(),
        domain="registration",
        current_state=state,
        node_type="effect",
        node_version=ModelSemVer.parse("1.0.0"),
        capabilities=ModelNodeCapabilities(),
        ack_deadline=ack_deadline,
        liveness_deadline=liveness_deadline,
        ack_timeout_emitted_at=ack_timeout_emitted_at,
        liveness_timeout_emitted_at=liveness_timeout_emitted_at,
        last_applied_event_id=uuid4(),
        last_applied_offset=100,
        registered_at=now,
        updated_at=now,
    )


@pytest.fixture
def mock_container() -> MagicMock:
    """Create a mock ONEX container."""
    return MagicMock(spec=ModelONEXContainer)


@pytest.fixture
def mock_reader() -> AsyncMock:
    """Create a mock projection reader."""
    reader = AsyncMock()
    reader.get_overdue_ack_registrations = AsyncMock(return_value=[])
    reader.get_overdue_liveness_registrations = AsyncMock(return_value=[])
    return reader


@pytest.fixture
def service(mock_container: MagicMock, mock_reader: AsyncMock) -> ServiceTimeoutScanner:
    """Create a ServiceTimeoutScanner instance with mocked dependencies."""
    return ServiceTimeoutScanner(
        container=mock_container, projection_reader=mock_reader
    )


@pytest.mark.unit
@pytest.mark.asyncio
class TestServiceTimeoutScannerBasics:
    """Test basic service instantiation and configuration."""

    async def test_service_instantiation(
        self, mock_container: MagicMock, mock_reader: AsyncMock
    ) -> None:
        """Test that service initializes correctly with container and reader."""
        service = ServiceTimeoutScanner(
            container=mock_container, projection_reader=mock_reader
        )

        assert service._container is mock_container
        assert service._reader is mock_reader
        assert service.batch_size == ServiceTimeoutScanner.DEFAULT_BATCH_SIZE

    async def test_service_custom_batch_size(
        self, mock_container: MagicMock, mock_reader: AsyncMock
    ) -> None:
        """Test that service accepts custom batch size."""
        custom_batch = 50
        service = ServiceTimeoutScanner(
            container=mock_container,
            projection_reader=mock_reader,
            batch_size=custom_batch,
        )

        assert service.batch_size == custom_batch

    async def test_service_default_batch_size_constant(self) -> None:
        """Test that default batch size constant is 100."""
        assert ServiceTimeoutScanner.DEFAULT_BATCH_SIZE == 100

    async def test_batch_size_property(self, service: ServiceTimeoutScanner) -> None:
        """Test batch_size property returns configured value."""
        assert service.batch_size == ServiceTimeoutScanner.DEFAULT_BATCH_SIZE


@pytest.mark.unit
@pytest.mark.asyncio
class TestServiceTimeoutScannerFindOverdue:
    """Test combined find_overdue_entities query."""

    async def test_find_overdue_entities_empty_results(
        self,
        service: ServiceTimeoutScanner,
        mock_reader: AsyncMock,
    ) -> None:
        """Test find_overdue_entities returns empty result when no overdue."""
        mock_reader.get_overdue_ack_registrations.return_value = []
        mock_reader.get_overdue_liveness_registrations.return_value = []

        now = datetime.now(UTC)
        result = await service.find_overdue_entities(now=now)

        assert isinstance(result, ModelTimeoutQueryResult)
        assert result.ack_timeouts == []
        assert result.liveness_expirations == []
        assert result.query_time == now
        assert result.query_duration_ms >= 0.0

    async def test_find_overdue_entities_with_ack_timeouts(
        self,
        service: ServiceTimeoutScanner,
        mock_reader: AsyncMock,
    ) -> None:
        """Test find_overdue_entities returns ack timeouts."""
        now = datetime.now(UTC)
        past_deadline = now - ACK_TIMEOUT_OFFSET

        ack_projections = [
            create_mock_projection(
                state=EnumRegistrationState.AWAITING_ACK,
                ack_deadline=past_deadline,
            ),
            create_mock_projection(
                state=EnumRegistrationState.ACCEPTED,
                ack_deadline=past_deadline,
            ),
        ]
        mock_reader.get_overdue_ack_registrations.return_value = ack_projections
        mock_reader.get_overdue_liveness_registrations.return_value = []

        result = await service.find_overdue_entities(now=now)

        assert len(result.ack_timeouts) == 2
        assert result.liveness_expirations == []
        assert result.total_overdue_count == 2

    async def test_find_overdue_entities_with_liveness_expirations(
        self,
        service: ServiceTimeoutScanner,
        mock_reader: AsyncMock,
    ) -> None:
        """Test find_overdue_entities returns liveness expirations."""
        now = datetime.now(UTC)
        past_deadline = now - LIVENESS_TIMEOUT_OFFSET

        liveness_projections = [
            create_mock_projection(
                state=EnumRegistrationState.ACTIVE,
                liveness_deadline=past_deadline,
            ),
        ]
        mock_reader.get_overdue_ack_registrations.return_value = []
        mock_reader.get_overdue_liveness_registrations.return_value = (
            liveness_projections
        )

        result = await service.find_overdue_entities(now=now)

        assert result.ack_timeouts == []
        assert len(result.liveness_expirations) == 1
        assert result.total_overdue_count == 1

    async def test_find_overdue_entities_with_both_types(
        self,
        service: ServiceTimeoutScanner,
        mock_reader: AsyncMock,
    ) -> None:
        """Test find_overdue_entities returns both timeout types."""
        now = datetime.now(UTC)
        past_deadline = now - ACK_TIMEOUT_OFFSET

        ack_projections = [
            create_mock_projection(
                state=EnumRegistrationState.AWAITING_ACK,
                ack_deadline=past_deadline,
            ),
        ]
        liveness_projections = [
            create_mock_projection(
                state=EnumRegistrationState.ACTIVE,
                liveness_deadline=past_deadline,
            ),
            create_mock_projection(
                state=EnumRegistrationState.ACTIVE,
                liveness_deadline=past_deadline,
            ),
        ]
        mock_reader.get_overdue_ack_registrations.return_value = ack_projections
        mock_reader.get_overdue_liveness_registrations.return_value = (
            liveness_projections
        )

        result = await service.find_overdue_entities(now=now)

        assert len(result.ack_timeouts) == 1
        assert len(result.liveness_expirations) == 2
        assert result.total_overdue_count == 3
        assert result.has_overdue_entities is True

    async def test_find_overdue_entities_passes_correct_parameters(
        self,
        service: ServiceTimeoutScanner,
        mock_reader: AsyncMock,
    ) -> None:
        """Test find_overdue_entities passes parameters to reader correctly."""
        now = datetime.now(UTC)
        domain = "custom_domain"
        correlation_id = uuid4()

        await service.find_overdue_entities(
            now=now,
            domain=domain,
            correlation_id=correlation_id,
        )

        # Verify ack query parameters
        ack_call = mock_reader.get_overdue_ack_registrations.call_args
        assert ack_call.kwargs["now"] == now
        assert ack_call.kwargs["domain"] == domain
        assert ack_call.kwargs["limit"] == service.batch_size
        assert ack_call.kwargs["correlation_id"] == correlation_id

        # Verify liveness query parameters
        liveness_call = mock_reader.get_overdue_liveness_registrations.call_args
        assert liveness_call.kwargs["now"] == now
        assert liveness_call.kwargs["domain"] == domain
        assert liveness_call.kwargs["limit"] == service.batch_size
        assert liveness_call.kwargs["correlation_id"] == correlation_id

    async def test_find_overdue_entities_generates_correlation_id(
        self,
        service: ServiceTimeoutScanner,
        mock_reader: AsyncMock,
    ) -> None:
        """Test find_overdue_entities generates correlation ID if not provided."""
        now = datetime.now(UTC)

        await service.find_overdue_entities(now=now)

        # Both calls should have a correlation_id
        ack_call = mock_reader.get_overdue_ack_registrations.call_args
        liveness_call = mock_reader.get_overdue_liveness_registrations.call_args

        assert ack_call.kwargs["correlation_id"] is not None
        assert liveness_call.kwargs["correlation_id"] is not None
        # Both should use the same correlation ID
        assert (
            ack_call.kwargs["correlation_id"] == liveness_call.kwargs["correlation_id"]
        )

    async def test_find_overdue_entities_tracks_query_duration(
        self,
        service: ServiceTimeoutScanner,
        mock_reader: AsyncMock,
    ) -> None:
        """Test find_overdue_entities tracks query duration."""
        now = datetime.now(UTC)

        result = await service.find_overdue_entities(now=now)

        # Duration should be a positive number (even if very small)
        assert result.query_duration_ms >= 0.0
        # Should be a reasonable value (less than 10 seconds)
        assert result.query_duration_ms < MAX_REASONABLE_QUERY_DURATION_MS


@pytest.mark.unit
@pytest.mark.asyncio
class TestServiceTimeoutScannerAckTimeouts:
    """Test ack-specific timeout query."""

    async def test_find_ack_timeouts_delegates_to_reader(
        self,
        service: ServiceTimeoutScanner,
        mock_reader: AsyncMock,
    ) -> None:
        """Test find_ack_timeouts delegates to reader correctly."""
        now = datetime.now(UTC)
        past_deadline = now - ACK_TIMEOUT_OFFSET

        ack_projections = [
            create_mock_projection(
                state=EnumRegistrationState.AWAITING_ACK,
                ack_deadline=past_deadline,
            ),
        ]
        mock_reader.get_overdue_ack_registrations.return_value = ack_projections

        result = await service.find_ack_timeouts(now=now)

        assert result == ack_projections
        mock_reader.get_overdue_ack_registrations.assert_called_once()

    async def test_find_ack_timeouts_passes_parameters(
        self,
        service: ServiceTimeoutScanner,
        mock_reader: AsyncMock,
    ) -> None:
        """Test find_ack_timeouts passes all parameters correctly."""
        now = datetime.now(UTC)
        domain = "test_domain"
        correlation_id = uuid4()

        await service.find_ack_timeouts(
            now=now,
            domain=domain,
            correlation_id=correlation_id,
        )

        call_args = mock_reader.get_overdue_ack_registrations.call_args
        assert call_args.kwargs["now"] == now
        assert call_args.kwargs["domain"] == domain
        assert call_args.kwargs["limit"] == service.batch_size
        assert call_args.kwargs["correlation_id"] == correlation_id

    async def test_find_ack_timeouts_empty_result(
        self,
        service: ServiceTimeoutScanner,
        mock_reader: AsyncMock,
    ) -> None:
        """Test find_ack_timeouts returns empty list when no timeouts."""
        mock_reader.get_overdue_ack_registrations.return_value = []

        result = await service.find_ack_timeouts(now=datetime.now(UTC))

        assert result == []

    async def test_find_ack_timeouts_generates_correlation_id(
        self,
        service: ServiceTimeoutScanner,
        mock_reader: AsyncMock,
    ) -> None:
        """Test find_ack_timeouts generates correlation ID if not provided."""
        await service.find_ack_timeouts(now=datetime.now(UTC))

        call_args = mock_reader.get_overdue_ack_registrations.call_args
        assert call_args.kwargs["correlation_id"] is not None


@pytest.mark.unit
@pytest.mark.asyncio
class TestServiceTimeoutScannerLivenessExpirations:
    """Test liveness-specific expiration query."""

    async def test_find_liveness_expirations_delegates_to_reader(
        self,
        service: ServiceTimeoutScanner,
        mock_reader: AsyncMock,
    ) -> None:
        """Test find_liveness_expirations delegates to reader correctly."""
        now = datetime.now(UTC)
        past_deadline = now - LIVENESS_TIMEOUT_OFFSET

        liveness_projections = [
            create_mock_projection(
                state=EnumRegistrationState.ACTIVE,
                liveness_deadline=past_deadline,
            ),
        ]
        mock_reader.get_overdue_liveness_registrations.return_value = (
            liveness_projections
        )

        result = await service.find_liveness_expirations(now=now)

        assert result == liveness_projections
        mock_reader.get_overdue_liveness_registrations.assert_called_once()

    async def test_find_liveness_expirations_passes_parameters(
        self,
        service: ServiceTimeoutScanner,
        mock_reader: AsyncMock,
    ) -> None:
        """Test find_liveness_expirations passes all parameters correctly."""
        now = datetime.now(UTC)
        domain = "test_domain"
        correlation_id = uuid4()

        await service.find_liveness_expirations(
            now=now,
            domain=domain,
            correlation_id=correlation_id,
        )

        call_args = mock_reader.get_overdue_liveness_registrations.call_args
        assert call_args.kwargs["now"] == now
        assert call_args.kwargs["domain"] == domain
        assert call_args.kwargs["limit"] == service.batch_size
        assert call_args.kwargs["correlation_id"] == correlation_id

    async def test_find_liveness_expirations_empty_result(
        self,
        service: ServiceTimeoutScanner,
        mock_reader: AsyncMock,
    ) -> None:
        """Test find_liveness_expirations returns empty list when no expirations."""
        mock_reader.get_overdue_liveness_registrations.return_value = []

        result = await service.find_liveness_expirations(now=datetime.now(UTC))

        assert result == []

    async def test_find_liveness_expirations_generates_correlation_id(
        self,
        service: ServiceTimeoutScanner,
        mock_reader: AsyncMock,
    ) -> None:
        """Test find_liveness_expirations generates correlation ID if not provided."""
        await service.find_liveness_expirations(now=datetime.now(UTC))

        call_args = mock_reader.get_overdue_liveness_registrations.call_args
        assert call_args.kwargs["correlation_id"] is not None


@pytest.mark.unit
class TestModelTimeoutQueryResult:
    """Test ModelTimeoutQueryResult model."""

    def test_result_model_creation(self) -> None:
        """Test result model can be created with required fields."""
        now = datetime.now(UTC)
        result = ModelTimeoutQueryResult(
            ack_timeouts=[],
            liveness_expirations=[],
            query_time=now,
            query_duration_ms=10.5,
        )

        assert result.query_time == now
        assert result.query_duration_ms == 10.5

    def test_result_model_with_projections(self) -> None:
        """Test result model with projection lists."""
        now = datetime.now(UTC)
        ack_proj = create_mock_projection(state=EnumRegistrationState.AWAITING_ACK)
        liveness_proj = create_mock_projection(state=EnumRegistrationState.ACTIVE)

        result = ModelTimeoutQueryResult(
            ack_timeouts=[ack_proj],
            liveness_expirations=[liveness_proj],
            query_time=now,
            query_duration_ms=5.0,
        )

        assert len(result.ack_timeouts) == 1
        assert len(result.liveness_expirations) == 1

    def test_total_overdue_count_property(self) -> None:
        """Test total_overdue_count property calculation."""
        now = datetime.now(UTC)
        result = ModelTimeoutQueryResult(
            ack_timeouts=[
                create_mock_projection(state=EnumRegistrationState.AWAITING_ACK),
                create_mock_projection(state=EnumRegistrationState.ACCEPTED),
            ],
            liveness_expirations=[
                create_mock_projection(state=EnumRegistrationState.ACTIVE),
            ],
            query_time=now,
            query_duration_ms=1.0,
        )

        assert result.total_overdue_count == 3

    def test_has_overdue_entities_true(self) -> None:
        """Test has_overdue_entities returns True when entities exist."""
        now = datetime.now(UTC)
        result = ModelTimeoutQueryResult(
            ack_timeouts=[create_mock_projection()],
            liveness_expirations=[],
            query_time=now,
            query_duration_ms=1.0,
        )

        assert result.has_overdue_entities is True

    def test_has_overdue_entities_false(self) -> None:
        """Test has_overdue_entities returns False when no entities."""
        now = datetime.now(UTC)
        result = ModelTimeoutQueryResult(
            ack_timeouts=[],
            liveness_expirations=[],
            query_time=now,
            query_duration_ms=1.0,
        )

        assert result.has_overdue_entities is False

    def test_result_model_is_frozen(self) -> None:
        """Test result model is immutable."""
        now = datetime.now(UTC)
        result = ModelTimeoutQueryResult(
            ack_timeouts=[],
            liveness_expirations=[],
            query_time=now,
            query_duration_ms=1.0,
        )

        with pytest.raises(ValidationError):
            result.query_duration_ms = 999.0  # type: ignore[misc]

    def test_result_model_rejects_negative_duration(self) -> None:
        """Test result model rejects negative duration."""
        now = datetime.now(UTC)

        with pytest.raises(ValidationError):
            ModelTimeoutQueryResult(
                ack_timeouts=[],
                liveness_expirations=[],
                query_time=now,
                query_duration_ms=-1.0,
            )

    def test_result_model_default_lists(self) -> None:
        """Test result model defaults to empty lists."""
        now = datetime.now(UTC)
        result = ModelTimeoutQueryResult(
            query_time=now,
            query_duration_ms=1.0,
        )

        assert result.ack_timeouts == []
        assert result.liveness_expirations == []


@pytest.mark.unit
@pytest.mark.asyncio
class TestServiceTimeoutScannerErrorHandling:
    """Test error handling for service operations."""

    async def test_find_overdue_entities_propagates_connection_error(
        self,
        service: ServiceTimeoutScanner,
        mock_reader: AsyncMock,
    ) -> None:
        """Test find_overdue_entities propagates connection errors."""
        mock_reader.get_overdue_ack_registrations.side_effect = InfraConnectionError(
            "Connection refused"
        )

        with pytest.raises(InfraConnectionError):
            await service.find_overdue_entities(now=datetime.now(UTC))

    async def test_find_overdue_entities_propagates_timeout_error(
        self,
        service: ServiceTimeoutScanner,
        mock_reader: AsyncMock,
    ) -> None:
        """Test find_overdue_entities propagates timeout errors."""
        mock_reader.get_overdue_ack_registrations.side_effect = InfraTimeoutError(
            "Query timed out",
            context=ModelTimeoutErrorContext(
                transport_type=EnumInfraTransportType.DATABASE,
                operation="get_overdue_ack_registrations",
            ),
        )

        with pytest.raises(InfraTimeoutError):
            await service.find_overdue_entities(now=datetime.now(UTC))

    async def test_find_overdue_entities_propagates_circuit_breaker_error(
        self,
        service: ServiceTimeoutScanner,
        mock_reader: AsyncMock,
    ) -> None:
        """Test find_overdue_entities propagates circuit breaker errors."""
        mock_reader.get_overdue_ack_registrations.side_effect = InfraUnavailableError(
            "Circuit breaker is open"
        )

        with pytest.raises(InfraUnavailableError):
            await service.find_overdue_entities(now=datetime.now(UTC))

    async def test_find_ack_timeouts_propagates_errors(
        self,
        service: ServiceTimeoutScanner,
        mock_reader: AsyncMock,
    ) -> None:
        """Test find_ack_timeouts propagates reader errors."""
        mock_reader.get_overdue_ack_registrations.side_effect = InfraConnectionError(
            "Connection refused"
        )

        with pytest.raises(InfraConnectionError):
            await service.find_ack_timeouts(now=datetime.now(UTC))

    async def test_find_liveness_expirations_propagates_errors(
        self,
        service: ServiceTimeoutScanner,
        mock_reader: AsyncMock,
    ) -> None:
        """Test find_liveness_expirations propagates reader errors."""
        mock_reader.get_overdue_liveness_registrations.side_effect = (
            InfraConnectionError("Connection refused")
        )

        with pytest.raises(InfraConnectionError):
            await service.find_liveness_expirations(now=datetime.now(UTC))

    async def test_liveness_error_after_successful_ack_query(
        self,
        service: ServiceTimeoutScanner,
        mock_reader: AsyncMock,
    ) -> None:
        """Test error handling when liveness query fails after ack succeeds."""
        # Ack query succeeds
        mock_reader.get_overdue_ack_registrations.return_value = []
        # Liveness query fails
        mock_reader.get_overdue_liveness_registrations.side_effect = InfraTimeoutError(
            "Liveness query timed out",
            context=ModelTimeoutErrorContext(
                transport_type=EnumInfraTransportType.DATABASE,
                operation="get_overdue_liveness_registrations",
            ),
        )

        with pytest.raises(InfraTimeoutError):
            await service.find_overdue_entities(now=datetime.now(UTC))


@pytest.mark.unit
@pytest.mark.asyncio
class TestServiceTimeoutScannerBatchSize:
    """Test batch size configuration and usage."""

    async def test_custom_batch_size_used_in_queries(
        self,
        mock_container: MagicMock,
        mock_reader: AsyncMock,
    ) -> None:
        """Test custom batch size is passed to reader queries."""
        custom_batch = 25
        service = ServiceTimeoutScanner(
            container=mock_container,
            projection_reader=mock_reader,
            batch_size=custom_batch,
        )

        await service.find_overdue_entities(now=datetime.now(UTC))

        # Verify batch size in both queries
        ack_call = mock_reader.get_overdue_ack_registrations.call_args
        liveness_call = mock_reader.get_overdue_liveness_registrations.call_args

        assert ack_call.kwargs["limit"] == custom_batch
        assert liveness_call.kwargs["limit"] == custom_batch

    async def test_batch_size_none_uses_default(
        self,
        mock_container: MagicMock,
        mock_reader: AsyncMock,
    ) -> None:
        """Test None batch size uses default."""
        service = ServiceTimeoutScanner(
            container=mock_container,
            projection_reader=mock_reader,
            batch_size=None,
        )

        assert service.batch_size == ServiceTimeoutScanner.DEFAULT_BATCH_SIZE
