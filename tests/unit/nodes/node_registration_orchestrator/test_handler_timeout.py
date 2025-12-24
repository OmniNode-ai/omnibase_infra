# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Comprehensive unit tests for HandlerTimeout.

This test suite validates:
- Handler instantiation with required dependencies
- RuntimeTick processing with injected time
- Correlation ID propagation from tick
- Query and emission coordination
- Error handling and result model
- Processing time tracking

Test Organization:
    - TestHandlerTimeoutBasics: Instantiation and configuration
    - TestHandlerTimeoutHandle: Main handle() method tests
    - TestHandlerTimeoutErrorHandling: Error scenarios
    - TestModelTimeoutHandlerResult: Result model tests

Coverage Goals:
    - >90% code coverage for handler
    - All code paths tested
    - Error handling validated
    - Timing metadata verified

Related Tickets:
    - OMN-932 (C2): Durable Timeout Handling
    - OMN-888 (C1): Registration Orchestrator
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from omnibase_infra.enums import EnumRegistrationState
from omnibase_infra.errors import (
    InfraConnectionError,
    InfraTimeoutError,
    InfraUnavailableError,
)
from omnibase_infra.models.projection import ModelRegistrationProjection
from omnibase_infra.models.registration.model_node_capabilities import (
    ModelNodeCapabilities,
)
from omnibase_infra.nodes.node_registration_orchestrator.handler_timeout import (
    HandlerTimeout,
    ModelTimeoutHandlerResult,
)
from omnibase_infra.runtime.models.model_runtime_tick import ModelRuntimeTick
from omnibase_infra.services import (
    ModelTimeoutEmissionResult,
    ModelTimeoutQueryResult,
)


def create_mock_tick(
    now: datetime | None = None,
    tick_id: None = None,
    sequence_number: int = 1,
    scheduler_id: str = "test-scheduler",
) -> ModelRuntimeTick:
    """Create a mock RuntimeTick for testing."""
    test_now = now or datetime.now(UTC)
    return ModelRuntimeTick(
        now=test_now,
        tick_id=tick_id or uuid4(),
        sequence_number=sequence_number,
        scheduled_at=test_now,
        correlation_id=uuid4(),
        scheduler_id=scheduler_id,
        tick_interval_ms=1000,
    )


def create_mock_projection(
    state: EnumRegistrationState = EnumRegistrationState.ACTIVE,
    ack_deadline: datetime | None = None,
    liveness_deadline: datetime | None = None,
) -> ModelRegistrationProjection:
    """Create a mock projection with sensible defaults."""
    now = datetime.now(UTC)
    return ModelRegistrationProjection(
        entity_id=uuid4(),
        domain="registration",
        current_state=state,
        node_type="effect",
        node_version="1.0.0",
        capabilities=ModelNodeCapabilities(),
        ack_deadline=ack_deadline,
        liveness_deadline=liveness_deadline,
        ack_timeout_emitted_at=None,
        liveness_timeout_emitted_at=None,
        last_applied_event_id=uuid4(),
        last_applied_offset=100,
        registered_at=now,
        updated_at=now,
    )


def create_mock_query_result(
    ack_timeouts: list[ModelRegistrationProjection] | None = None,
    liveness_expirations: list[ModelRegistrationProjection] | None = None,
    query_time: datetime | None = None,
) -> ModelTimeoutQueryResult:
    """Create a mock query result."""
    return ModelTimeoutQueryResult(
        ack_timeouts=ack_timeouts or [],
        liveness_expirations=liveness_expirations or [],
        query_time=query_time or datetime.now(UTC),
        query_duration_ms=5.0,
    )


def create_mock_emission_result(
    ack_emitted: int = 0,
    liveness_emitted: int = 0,
    markers_updated: int = 0,
    errors: list[str] | None = None,
    tick_id: None = None,
    correlation_id: None = None,
) -> ModelTimeoutEmissionResult:
    """Create a mock emission result."""
    return ModelTimeoutEmissionResult(
        ack_timeouts_emitted=ack_emitted,
        liveness_expirations_emitted=liveness_emitted,
        markers_updated=markers_updated,
        errors=errors or [],
        processing_time_ms=10.0,
        tick_id=tick_id or uuid4(),
        correlation_id=correlation_id or uuid4(),
    )


@pytest.fixture
def mock_timeout_query() -> AsyncMock:
    """Create a mock ServiceTimeoutQuery."""
    query = AsyncMock()
    query.find_overdue_entities = AsyncMock(
        return_value=create_mock_query_result(),
    )
    return query


@pytest.fixture
def mock_timeout_emission() -> AsyncMock:
    """Create a mock ServiceTimeoutEmission."""
    emission = AsyncMock()
    emission.process_timeouts = AsyncMock(
        return_value=create_mock_emission_result(),
    )
    return emission


@pytest.fixture
def handler(
    mock_timeout_query: AsyncMock,
    mock_timeout_emission: AsyncMock,
) -> HandlerTimeout:
    """Create a HandlerTimeout instance with mocked dependencies."""
    return HandlerTimeout(
        timeout_query=mock_timeout_query,
        timeout_emission=mock_timeout_emission,
    )


@pytest.mark.unit
class TestHandlerTimeoutBasics:
    """Test basic handler instantiation and configuration."""

    def test_handler_instantiation(
        self,
        mock_timeout_query: AsyncMock,
        mock_timeout_emission: AsyncMock,
    ) -> None:
        """Test that handler initializes correctly with dependencies."""
        handler = HandlerTimeout(
            timeout_query=mock_timeout_query,
            timeout_emission=mock_timeout_emission,
        )

        assert handler._timeout_query is mock_timeout_query
        assert handler._timeout_emission is mock_timeout_emission

    def test_handler_stores_dependencies(
        self,
        handler: HandlerTimeout,
        mock_timeout_query: AsyncMock,
        mock_timeout_emission: AsyncMock,
    ) -> None:
        """Test that handler stores dependencies correctly."""
        assert handler._timeout_query is mock_timeout_query
        assert handler._timeout_emission is mock_timeout_emission


@pytest.mark.unit
@pytest.mark.asyncio
class TestHandlerTimeoutHandle:
    """Test the main handle() method."""

    async def test_handle_uses_tick_now(
        self,
        handler: HandlerTimeout,
        mock_timeout_query: AsyncMock,
        mock_timeout_emission: AsyncMock,
    ) -> None:
        """Test that handle() uses tick.now, not system clock."""
        # Create a tick with a specific time in the past
        past_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
        tick = create_mock_tick(now=past_time)

        await handler.handle(tick)

        # Verify query was called with tick.now
        query_call = mock_timeout_query.find_overdue_entities.call_args
        assert query_call.kwargs["now"] == past_time

        # Verify emission was called with tick.now
        emission_call = mock_timeout_emission.process_timeouts.call_args
        assert emission_call.kwargs["now"] == past_time

    async def test_handle_propagates_correlation_id(
        self,
        handler: HandlerTimeout,
        mock_timeout_query: AsyncMock,
        mock_timeout_emission: AsyncMock,
    ) -> None:
        """Test that handle() propagates correlation_id from tick."""
        tick = create_mock_tick()

        await handler.handle(tick)

        # Verify correlation_id in query call
        query_call = mock_timeout_query.find_overdue_entities.call_args
        assert query_call.kwargs["correlation_id"] == tick.correlation_id

        # Verify correlation_id in emission call
        emission_call = mock_timeout_emission.process_timeouts.call_args
        assert emission_call.kwargs["correlation_id"] == tick.correlation_id

    async def test_handle_propagates_tick_id(
        self,
        handler: HandlerTimeout,
        mock_timeout_query: AsyncMock,
        mock_timeout_emission: AsyncMock,
    ) -> None:
        """Test that handle() propagates tick_id to emission."""
        tick = create_mock_tick()

        await handler.handle(tick)

        # Verify tick_id in emission call
        emission_call = mock_timeout_emission.process_timeouts.call_args
        assert emission_call.kwargs["tick_id"] == tick.tick_id

    async def test_handle_passes_domain_parameter(
        self,
        handler: HandlerTimeout,
        mock_timeout_query: AsyncMock,
        mock_timeout_emission: AsyncMock,
    ) -> None:
        """Test that handle() passes domain to both services."""
        tick = create_mock_tick()
        custom_domain = "custom_domain"

        await handler.handle(tick, domain=custom_domain)

        # Verify domain in query call
        query_call = mock_timeout_query.find_overdue_entities.call_args
        assert query_call.kwargs["domain"] == custom_domain

        # Verify domain in emission call
        emission_call = mock_timeout_emission.process_timeouts.call_args
        assert emission_call.kwargs["domain"] == custom_domain

    async def test_handle_returns_success_result(
        self,
        handler: HandlerTimeout,
        mock_timeout_query: AsyncMock,
        mock_timeout_emission: AsyncMock,
    ) -> None:
        """Test that handle() returns success result on normal execution."""
        tick = create_mock_tick()

        result = await handler.handle(tick)

        assert isinstance(result, ModelTimeoutHandlerResult)
        assert result.success is True
        assert result.error is None
        assert result.tick_id == tick.tick_id
        assert result.tick_now == tick.now

    async def test_handle_returns_correct_counts(
        self,
        handler: HandlerTimeout,
        mock_timeout_query: AsyncMock,
        mock_timeout_emission: AsyncMock,
    ) -> None:
        """Test that handle() returns correct counts from query and emission."""
        now = datetime.now(UTC)
        past_deadline = now - timedelta(minutes=5)

        # Set up query result with overdue entities
        ack_projection = create_mock_projection(
            state=EnumRegistrationState.AWAITING_ACK,
            ack_deadline=past_deadline,
        )
        liveness_projection = create_mock_projection(
            state=EnumRegistrationState.ACTIVE,
            liveness_deadline=past_deadline,
        )
        mock_timeout_query.find_overdue_entities.return_value = create_mock_query_result(
            ack_timeouts=[ack_projection],
            liveness_expirations=[liveness_projection, liveness_projection],
            query_time=now,
        )

        # Set up emission result
        mock_timeout_emission.process_timeouts.return_value = create_mock_emission_result(
            ack_emitted=1,
            liveness_emitted=2,
            markers_updated=3,
        )

        tick = create_mock_tick(now=now)
        result = await handler.handle(tick)

        # Verify counts from query
        assert result.ack_timeouts_found == 1
        assert result.liveness_expirations_found == 2
        assert result.total_found == 3

        # Verify counts from emission
        assert result.ack_timeouts_emitted == 1
        assert result.liveness_expirations_emitted == 2
        assert result.total_emitted == 3
        assert result.markers_updated == 3

    async def test_handle_tracks_processing_time(
        self,
        handler: HandlerTimeout,
        mock_timeout_query: AsyncMock,
        mock_timeout_emission: AsyncMock,
    ) -> None:
        """Test that handle() tracks processing time."""
        tick = create_mock_tick()

        result = await handler.handle(tick)

        # Processing time should be a positive number
        assert result.processing_time_ms >= 0.0
        assert result.query_time_ms >= 0.0
        assert result.emission_time_ms >= 0.0

    async def test_handle_captures_emission_errors(
        self,
        handler: HandlerTimeout,
        mock_timeout_query: AsyncMock,
        mock_timeout_emission: AsyncMock,
    ) -> None:
        """Test that handle() captures non-fatal errors from emission."""
        # Set up emission result with errors
        mock_timeout_emission.process_timeouts.return_value = create_mock_emission_result(
            ack_emitted=0,
            errors=["Error 1", "Error 2"],
        )

        tick = create_mock_tick()
        result = await handler.handle(tick)

        # Result should still be success but with errors captured
        assert result.success is True
        assert result.errors == ["Error 1", "Error 2"]
        assert result.has_errors is True

    async def test_handle_empty_results(
        self,
        handler: HandlerTimeout,
        mock_timeout_query: AsyncMock,
        mock_timeout_emission: AsyncMock,
    ) -> None:
        """Test handle() with no overdue entities."""
        tick = create_mock_tick()

        result = await handler.handle(tick)

        assert result.ack_timeouts_found == 0
        assert result.liveness_expirations_found == 0
        assert result.total_found == 0
        assert result.total_emitted == 0
        assert result.success is True

    async def test_handle_default_domain(
        self,
        handler: HandlerTimeout,
        mock_timeout_query: AsyncMock,
        mock_timeout_emission: AsyncMock,
    ) -> None:
        """Test that handle() uses default domain 'registration'."""
        tick = create_mock_tick()

        await handler.handle(tick)

        query_call = mock_timeout_query.find_overdue_entities.call_args
        assert query_call.kwargs["domain"] == "registration"


@pytest.mark.unit
@pytest.mark.asyncio
class TestHandlerTimeoutErrorHandling:
    """Test error handling for handler operations."""

    async def test_handle_catches_query_connection_error(
        self,
        handler: HandlerTimeout,
        mock_timeout_query: AsyncMock,
        mock_timeout_emission: AsyncMock,
    ) -> None:
        """Test that handle() catches and returns connection errors from query."""
        mock_timeout_query.find_overdue_entities.side_effect = InfraConnectionError(
            "Connection refused"
        )

        tick = create_mock_tick()
        result = await handler.handle(tick)

        assert result.success is False
        assert result.error is not None
        assert "InfraConnectionError" in result.error
        assert result.tick_id == tick.tick_id

    async def test_handle_catches_query_timeout_error(
        self,
        handler: HandlerTimeout,
        mock_timeout_query: AsyncMock,
        mock_timeout_emission: AsyncMock,
    ) -> None:
        """Test that handle() catches and returns timeout errors from query."""
        mock_timeout_query.find_overdue_entities.side_effect = InfraTimeoutError(
            "Query timed out"
        )

        tick = create_mock_tick()
        result = await handler.handle(tick)

        assert result.success is False
        assert result.error is not None
        assert "InfraTimeoutError" in result.error

    async def test_handle_catches_emission_circuit_breaker_error(
        self,
        handler: HandlerTimeout,
        mock_timeout_query: AsyncMock,
        mock_timeout_emission: AsyncMock,
    ) -> None:
        """Test that handle() catches circuit breaker errors from emission."""
        mock_timeout_emission.process_timeouts.side_effect = InfraUnavailableError(
            "Circuit breaker is open"
        )

        tick = create_mock_tick()
        result = await handler.handle(tick)

        assert result.success is False
        assert result.error is not None
        assert "InfraUnavailableError" in result.error

    async def test_handle_catches_generic_exception(
        self,
        handler: HandlerTimeout,
        mock_timeout_query: AsyncMock,
        mock_timeout_emission: AsyncMock,
    ) -> None:
        """Test that handle() catches generic exceptions."""
        mock_timeout_query.find_overdue_entities.side_effect = RuntimeError(
            "Unexpected error"
        )

        tick = create_mock_tick()
        result = await handler.handle(tick)

        assert result.success is False
        assert result.error is not None
        assert "RuntimeError" in result.error
        assert "Unexpected error" in result.error

    async def test_handle_tracks_processing_time_on_error(
        self,
        handler: HandlerTimeout,
        mock_timeout_query: AsyncMock,
        mock_timeout_emission: AsyncMock,
    ) -> None:
        """Test that handle() tracks processing time even on error."""
        mock_timeout_query.find_overdue_entities.side_effect = InfraConnectionError(
            "Connection refused"
        )

        tick = create_mock_tick()
        result = await handler.handle(tick)

        assert result.processing_time_ms >= 0.0


@pytest.mark.unit
class TestModelTimeoutHandlerResult:
    """Test ModelTimeoutHandlerResult model."""

    def test_result_model_creation(self) -> None:
        """Test result model can be created with required fields."""
        tick_id = uuid4()
        now = datetime.now(UTC)

        result = ModelTimeoutHandlerResult(
            tick_id=tick_id,
            tick_now=now,
            processing_time_ms=10.5,
        )

        assert result.tick_id == tick_id
        assert result.tick_now == now
        assert result.processing_time_ms == 10.5

    def test_result_model_defaults(self) -> None:
        """Test result model has sensible defaults."""
        result = ModelTimeoutHandlerResult(
            tick_id=uuid4(),
            tick_now=datetime.now(UTC),
            processing_time_ms=10.0,
        )

        assert result.ack_timeouts_found == 0
        assert result.liveness_expirations_found == 0
        assert result.ack_timeouts_emitted == 0
        assert result.liveness_expirations_emitted == 0
        assert result.markers_updated == 0
        assert result.query_time_ms == 0.0
        assert result.emission_time_ms == 0.0
        assert result.success is True
        assert result.error is None
        assert result.errors == []

    def test_total_found_property(self) -> None:
        """Test total_found property calculation."""
        result = ModelTimeoutHandlerResult(
            tick_id=uuid4(),
            tick_now=datetime.now(UTC),
            ack_timeouts_found=2,
            liveness_expirations_found=3,
            processing_time_ms=10.0,
        )

        assert result.total_found == 5

    def test_total_emitted_property(self) -> None:
        """Test total_emitted property calculation."""
        result = ModelTimeoutHandlerResult(
            tick_id=uuid4(),
            tick_now=datetime.now(UTC),
            ack_timeouts_emitted=1,
            liveness_expirations_emitted=2,
            processing_time_ms=10.0,
        )

        assert result.total_emitted == 3

    def test_has_errors_with_error(self) -> None:
        """Test has_errors property when error is set."""
        result = ModelTimeoutHandlerResult(
            tick_id=uuid4(),
            tick_now=datetime.now(UTC),
            processing_time_ms=10.0,
            success=False,
            error="Some error",
        )

        assert result.has_errors is True

    def test_has_errors_with_errors_list(self) -> None:
        """Test has_errors property when errors list is populated."""
        result = ModelTimeoutHandlerResult(
            tick_id=uuid4(),
            tick_now=datetime.now(UTC),
            processing_time_ms=10.0,
            errors=["Error 1"],
        )

        assert result.has_errors is True

    def test_has_errors_false(self) -> None:
        """Test has_errors property returns False when no errors."""
        result = ModelTimeoutHandlerResult(
            tick_id=uuid4(),
            tick_now=datetime.now(UTC),
            processing_time_ms=10.0,
        )

        assert result.has_errors is False

    def test_result_model_is_frozen(self) -> None:
        """Test result model is immutable."""
        result = ModelTimeoutHandlerResult(
            tick_id=uuid4(),
            tick_now=datetime.now(UTC),
            processing_time_ms=10.0,
        )

        with pytest.raises(Exception):  # Pydantic validation error
            result.processing_time_ms = 999.0  # type: ignore[misc]

    def test_result_model_rejects_negative_counts(self) -> None:
        """Test result model rejects negative count values."""
        with pytest.raises(Exception):  # Pydantic validation error
            ModelTimeoutHandlerResult(
                tick_id=uuid4(),
                tick_now=datetime.now(UTC),
                ack_timeouts_found=-1,
                processing_time_ms=10.0,
            )

    def test_result_model_rejects_negative_processing_time(self) -> None:
        """Test result model rejects negative processing time."""
        with pytest.raises(Exception):  # Pydantic validation error
            ModelTimeoutHandlerResult(
                tick_id=uuid4(),
                tick_now=datetime.now(UTC),
                processing_time_ms=-1.0,
            )


@pytest.mark.unit
@pytest.mark.asyncio
class TestHandlerTimeoutIntegration:
    """Integration tests for HandlerTimeout with services."""

    async def test_full_timeout_processing_flow(
        self,
        handler: HandlerTimeout,
        mock_timeout_query: AsyncMock,
        mock_timeout_emission: AsyncMock,
    ) -> None:
        """Test complete timeout processing flow."""
        now = datetime.now(UTC)
        past_deadline = now - timedelta(minutes=5)

        # Set up query result
        ack_projection = create_mock_projection(
            state=EnumRegistrationState.AWAITING_ACK,
            ack_deadline=past_deadline,
        )
        mock_timeout_query.find_overdue_entities.return_value = create_mock_query_result(
            ack_timeouts=[ack_projection],
            query_time=now,
        )

        # Set up emission result
        mock_timeout_emission.process_timeouts.return_value = create_mock_emission_result(
            ack_emitted=1,
            markers_updated=1,
        )

        tick = create_mock_tick(now=now)
        result = await handler.handle(tick)

        # Verify full flow
        assert result.success is True
        assert result.ack_timeouts_found == 1
        assert result.ack_timeouts_emitted == 1
        assert result.markers_updated == 1
        assert result.tick_id == tick.tick_id
        assert result.tick_now == now

        # Verify services were called correctly
        mock_timeout_query.find_overdue_entities.assert_called_once()
        mock_timeout_emission.process_timeouts.assert_called_once()
