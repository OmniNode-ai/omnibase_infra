# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Comprehensive unit tests for TimeoutCoordinator.

This test suite validates:
- Coordinator instantiation with required dependencies
- RuntimeTick coordination with injected time
- Correlation ID propagation from tick
- Query and emission coordination
- Error handling and result model
- Coordination time tracking

Test Organization:
    - TestTimeoutCoordinatorBasics: Instantiation and configuration
    - TestTimeoutCoordinatorCoordinate: Main coordinate() method tests
    - TestTimeoutCoordinatorErrorHandling: Error scenarios
    - TestModelTimeoutCoordinationResult: Result model tests

Coverage Goals:
    - >90% code coverage for coordinator
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
from uuid import UUID, uuid4

import pytest
from omnibase_core.enums.enum_node_kind import EnumNodeKind
from pydantic import ValidationError

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
from omnibase_infra.nodes.node_registration_orchestrator.timeout_coordinator import (
    ModelTimeoutCoordinationResult,
    TimeoutCoordinator,
)
from omnibase_infra.runtime.models.model_runtime_tick import ModelRuntimeTick
from omnibase_infra.services import (
    ModelTimeoutEmissionResult,
    ModelTimeoutQueryResult,
)


def create_mock_tick(
    now: datetime | None = None,
    tick_id: UUID | None = None,
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
        node_type=EnumNodeKind.EFFECT.value,
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
    tick_id: UUID | None = None,
    correlation_id: UUID | None = None,
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
    """Create a mock TimeoutScanner."""
    query = AsyncMock()
    query.find_overdue_entities = AsyncMock(
        return_value=create_mock_query_result(),
    )
    return query


@pytest.fixture
def mock_timeout_emission() -> AsyncMock:
    """Create a mock TimeoutEmitter."""
    emission = AsyncMock()
    emission.process_timeouts = AsyncMock(
        return_value=create_mock_emission_result(),
    )
    return emission


@pytest.fixture
def coordinator(
    mock_timeout_query: AsyncMock,
    mock_timeout_emission: AsyncMock,
) -> TimeoutCoordinator:
    """Create a TimeoutCoordinator instance with mocked dependencies."""
    return TimeoutCoordinator(
        timeout_query=mock_timeout_query,
        timeout_emission=mock_timeout_emission,
    )


@pytest.mark.unit
class TestTimeoutCoordinatorBasics:
    """Test basic coordinator instantiation and configuration."""

    def test_coordinator_instantiation(
        self,
        mock_timeout_query: AsyncMock,
        mock_timeout_emission: AsyncMock,
    ) -> None:
        """Test that coordinator initializes correctly with dependencies."""
        coordinator = TimeoutCoordinator(
            timeout_query=mock_timeout_query,
            timeout_emission=mock_timeout_emission,
        )

        assert coordinator._timeout_query is mock_timeout_query
        assert coordinator._timeout_emission is mock_timeout_emission

    def test_coordinator_stores_dependencies(
        self,
        coordinator: TimeoutCoordinator,
        mock_timeout_query: AsyncMock,
        mock_timeout_emission: AsyncMock,
    ) -> None:
        """Test that coordinator stores dependencies correctly."""
        assert coordinator._timeout_query is mock_timeout_query
        assert coordinator._timeout_emission is mock_timeout_emission


@pytest.mark.unit
@pytest.mark.asyncio
class TestTimeoutCoordinatorCoordinate:
    """Test the main coordinate() method."""

    async def test_coordinate_uses_tick_now(
        self,
        coordinator: TimeoutCoordinator,
        mock_timeout_query: AsyncMock,
        mock_timeout_emission: AsyncMock,
    ) -> None:
        """Test that coordinate() uses tick.now, not system clock."""
        # Create a tick with a specific time in the past
        past_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
        tick = create_mock_tick(now=past_time)

        await coordinator.coordinate(tick)

        # Verify query was called with tick.now
        query_call = mock_timeout_query.find_overdue_entities.call_args
        assert query_call.kwargs["now"] == past_time

        # Verify emission was called with tick.now
        emission_call = mock_timeout_emission.process_timeouts.call_args
        assert emission_call.kwargs["now"] == past_time

    async def test_coordinate_propagates_correlation_id(
        self,
        coordinator: TimeoutCoordinator,
        mock_timeout_query: AsyncMock,
        mock_timeout_emission: AsyncMock,
    ) -> None:
        """Test that coordinate() propagates correlation_id from tick."""
        tick = create_mock_tick()

        await coordinator.coordinate(tick)

        # Verify correlation_id in query call
        query_call = mock_timeout_query.find_overdue_entities.call_args
        assert query_call.kwargs["correlation_id"] == tick.correlation_id

        # Verify correlation_id in emission call
        emission_call = mock_timeout_emission.process_timeouts.call_args
        assert emission_call.kwargs["correlation_id"] == tick.correlation_id

    async def test_coordinate_propagates_tick_id(
        self,
        coordinator: TimeoutCoordinator,
        mock_timeout_query: AsyncMock,
        mock_timeout_emission: AsyncMock,
    ) -> None:
        """Test that coordinate() propagates tick_id to emission."""
        tick = create_mock_tick()

        await coordinator.coordinate(tick)

        # Verify tick_id in emission call
        emission_call = mock_timeout_emission.process_timeouts.call_args
        assert emission_call.kwargs["tick_id"] == tick.tick_id

    async def test_coordinate_passes_domain_parameter(
        self,
        coordinator: TimeoutCoordinator,
        mock_timeout_query: AsyncMock,
        mock_timeout_emission: AsyncMock,
    ) -> None:
        """Test that coordinate() passes domain to both services."""
        tick = create_mock_tick()
        custom_domain = "custom_domain"

        await coordinator.coordinate(tick, domain=custom_domain)

        # Verify domain in query call
        query_call = mock_timeout_query.find_overdue_entities.call_args
        assert query_call.kwargs["domain"] == custom_domain

        # Verify domain in emission call
        emission_call = mock_timeout_emission.process_timeouts.call_args
        assert emission_call.kwargs["domain"] == custom_domain

    async def test_coordinate_returns_success_result(
        self,
        coordinator: TimeoutCoordinator,
        mock_timeout_query: AsyncMock,
        mock_timeout_emission: AsyncMock,
    ) -> None:
        """Test that coordinate() returns success result on normal execution."""
        tick = create_mock_tick()

        result = await coordinator.coordinate(tick)

        assert isinstance(result, ModelTimeoutCoordinationResult)
        assert result.success is True
        assert result.error is None
        assert result.tick_id == tick.tick_id
        assert result.tick_now == tick.now

    async def test_coordinate_returns_correct_counts(
        self,
        coordinator: TimeoutCoordinator,
        mock_timeout_query: AsyncMock,
        mock_timeout_emission: AsyncMock,
    ) -> None:
        """Test that coordinate() returns correct counts from query and emission."""
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
        mock_timeout_query.find_overdue_entities.return_value = (
            create_mock_query_result(
                ack_timeouts=[ack_projection],
                liveness_expirations=[liveness_projection, liveness_projection],
                query_time=now,
            )
        )

        # Set up emission result
        mock_timeout_emission.process_timeouts.return_value = (
            create_mock_emission_result(
                ack_emitted=1,
                liveness_emitted=2,
                markers_updated=3,
            )
        )

        tick = create_mock_tick(now=now)
        result = await coordinator.coordinate(tick)

        # Verify counts from query
        assert result.ack_timeouts_found == 1
        assert result.liveness_expirations_found == 2
        assert result.total_found == 3

        # Verify counts from emission
        assert result.ack_timeouts_emitted == 1
        assert result.liveness_expirations_emitted == 2
        assert result.total_emitted == 3
        assert result.markers_updated == 3

    async def test_coordinate_tracks_coordination_time(
        self,
        coordinator: TimeoutCoordinator,
        mock_timeout_query: AsyncMock,
        mock_timeout_emission: AsyncMock,
    ) -> None:
        """Test that coordinate() tracks coordination time."""
        tick = create_mock_tick()

        result = await coordinator.coordinate(tick)

        # Coordination time should be a positive number
        assert result.coordination_time_ms >= 0.0
        assert result.query_time_ms >= 0.0
        assert result.emission_time_ms >= 0.0

    async def test_coordinate_captures_emission_errors(
        self,
        coordinator: TimeoutCoordinator,
        mock_timeout_query: AsyncMock,
        mock_timeout_emission: AsyncMock,
    ) -> None:
        """Test that coordinate() captures non-fatal errors from emission."""
        # Set up emission result with errors
        mock_timeout_emission.process_timeouts.return_value = (
            create_mock_emission_result(
                ack_emitted=0,
                errors=["Error 1", "Error 2"],
            )
        )

        tick = create_mock_tick()
        result = await coordinator.coordinate(tick)

        # Result should still be success but with errors captured
        assert result.success is True
        assert result.errors == ["Error 1", "Error 2"]
        assert result.has_errors is True

    async def test_coordinate_empty_results(
        self,
        coordinator: TimeoutCoordinator,
        mock_timeout_query: AsyncMock,
        mock_timeout_emission: AsyncMock,
    ) -> None:
        """Test coordinate() with no overdue entities."""
        tick = create_mock_tick()

        result = await coordinator.coordinate(tick)

        assert result.ack_timeouts_found == 0
        assert result.liveness_expirations_found == 0
        assert result.total_found == 0
        assert result.total_emitted == 0
        assert result.success is True

    async def test_coordinate_default_domain(
        self,
        coordinator: TimeoutCoordinator,
        mock_timeout_query: AsyncMock,
        mock_timeout_emission: AsyncMock,
    ) -> None:
        """Test that coordinate() uses default domain 'registration'."""
        tick = create_mock_tick()

        await coordinator.coordinate(tick)

        query_call = mock_timeout_query.find_overdue_entities.call_args
        assert query_call.kwargs["domain"] == "registration"


@pytest.mark.unit
@pytest.mark.asyncio
class TestTimeoutCoordinatorErrorHandling:
    """Test error handling for coordinator operations."""

    async def test_coordinate_catches_query_connection_error(
        self,
        coordinator: TimeoutCoordinator,
        mock_timeout_query: AsyncMock,
        mock_timeout_emission: AsyncMock,
    ) -> None:
        """Test that coordinate() catches and returns connection errors from query."""
        mock_timeout_query.find_overdue_entities.side_effect = InfraConnectionError(
            "Connection refused"
        )

        tick = create_mock_tick()
        result = await coordinator.coordinate(tick)

        assert result.success is False
        assert result.error is not None
        assert "InfraConnectionError" in result.error
        assert result.tick_id == tick.tick_id

    async def test_coordinate_catches_query_timeout_error(
        self,
        coordinator: TimeoutCoordinator,
        mock_timeout_query: AsyncMock,
        mock_timeout_emission: AsyncMock,
    ) -> None:
        """Test that coordinate() catches and returns timeout errors from query."""
        mock_timeout_query.find_overdue_entities.side_effect = InfraTimeoutError(
            "Query timed out"
        )

        tick = create_mock_tick()
        result = await coordinator.coordinate(tick)

        assert result.success is False
        assert result.error is not None
        assert "InfraTimeoutError" in result.error

    async def test_coordinate_catches_emission_circuit_breaker_error(
        self,
        coordinator: TimeoutCoordinator,
        mock_timeout_query: AsyncMock,
        mock_timeout_emission: AsyncMock,
    ) -> None:
        """Test that coordinate() catches circuit breaker errors from emission."""
        mock_timeout_emission.process_timeouts.side_effect = InfraUnavailableError(
            "Circuit breaker is open"
        )

        tick = create_mock_tick()
        result = await coordinator.coordinate(tick)

        assert result.success is False
        assert result.error is not None
        assert "InfraUnavailableError" in result.error

    async def test_coordinate_catches_generic_exception(
        self,
        coordinator: TimeoutCoordinator,
        mock_timeout_query: AsyncMock,
        mock_timeout_emission: AsyncMock,
    ) -> None:
        """Test that coordinate() catches generic exceptions."""
        mock_timeout_query.find_overdue_entities.side_effect = RuntimeError(
            "Unexpected error"
        )

        tick = create_mock_tick()
        result = await coordinator.coordinate(tick)

        assert result.success is False
        assert result.error is not None
        assert "RuntimeError" in result.error
        assert "Unexpected error" in result.error

    async def test_coordinate_tracks_coordination_time_on_error(
        self,
        coordinator: TimeoutCoordinator,
        mock_timeout_query: AsyncMock,
        mock_timeout_emission: AsyncMock,
    ) -> None:
        """Test that coordinate() tracks coordination time even on error."""
        mock_timeout_query.find_overdue_entities.side_effect = InfraConnectionError(
            "Connection refused"
        )

        tick = create_mock_tick()
        result = await coordinator.coordinate(tick)

        assert result.coordination_time_ms >= 0.0


@pytest.mark.unit
class TestModelTimeoutCoordinationResult:
    """Test ModelTimeoutCoordinationResult model."""

    def test_result_model_creation(self) -> None:
        """Test result model can be created with required fields."""
        tick_id = uuid4()
        now = datetime.now(UTC)

        result = ModelTimeoutCoordinationResult(
            tick_id=tick_id,
            tick_now=now,
            coordination_time_ms=10.5,
        )

        assert result.tick_id == tick_id
        assert result.tick_now == now
        assert result.coordination_time_ms == 10.5

    def test_result_model_defaults(self) -> None:
        """Test result model has sensible defaults."""
        result = ModelTimeoutCoordinationResult(
            tick_id=uuid4(),
            tick_now=datetime.now(UTC),
            coordination_time_ms=10.0,
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
        result = ModelTimeoutCoordinationResult(
            tick_id=uuid4(),
            tick_now=datetime.now(UTC),
            ack_timeouts_found=2,
            liveness_expirations_found=3,
            coordination_time_ms=10.0,
        )

        assert result.total_found == 5

    def test_total_emitted_property(self) -> None:
        """Test total_emitted property calculation."""
        result = ModelTimeoutCoordinationResult(
            tick_id=uuid4(),
            tick_now=datetime.now(UTC),
            ack_timeouts_emitted=1,
            liveness_expirations_emitted=2,
            coordination_time_ms=10.0,
        )

        assert result.total_emitted == 3

    def test_has_errors_with_error(self) -> None:
        """Test has_errors property when error is set."""
        result = ModelTimeoutCoordinationResult(
            tick_id=uuid4(),
            tick_now=datetime.now(UTC),
            coordination_time_ms=10.0,
            success=False,
            error="Some error",
        )

        assert result.has_errors is True

    def test_has_errors_with_errors_list(self) -> None:
        """Test has_errors property when errors list is populated."""
        result = ModelTimeoutCoordinationResult(
            tick_id=uuid4(),
            tick_now=datetime.now(UTC),
            coordination_time_ms=10.0,
            errors=["Error 1"],
        )

        assert result.has_errors is True

    def test_has_errors_false(self) -> None:
        """Test has_errors property returns False when no errors."""
        result = ModelTimeoutCoordinationResult(
            tick_id=uuid4(),
            tick_now=datetime.now(UTC),
            coordination_time_ms=10.0,
        )

        assert result.has_errors is False

    def test_result_model_is_frozen(self) -> None:
        """Test result model is immutable."""
        result = ModelTimeoutCoordinationResult(
            tick_id=uuid4(),
            tick_now=datetime.now(UTC),
            coordination_time_ms=10.0,
        )

        with pytest.raises(ValidationError):
            result.coordination_time_ms = 999.0  # type: ignore[misc]

    def test_result_model_rejects_negative_counts(self) -> None:
        """Test result model rejects negative count values."""
        with pytest.raises(ValidationError):
            ModelTimeoutCoordinationResult(
                tick_id=uuid4(),
                tick_now=datetime.now(UTC),
                ack_timeouts_found=-1,
                coordination_time_ms=10.0,
            )

    def test_result_model_rejects_negative_coordination_time(self) -> None:
        """Test result model rejects negative coordination time."""
        with pytest.raises(ValidationError):
            ModelTimeoutCoordinationResult(
                tick_id=uuid4(),
                tick_now=datetime.now(UTC),
                coordination_time_ms=-1.0,
            )


@pytest.mark.unit
@pytest.mark.asyncio
class TestTimeoutCoordinatorIntegration:
    """Integration tests for TimeoutCoordinator with services."""

    async def test_full_timeout_coordination_flow(
        self,
        coordinator: TimeoutCoordinator,
        mock_timeout_query: AsyncMock,
        mock_timeout_emission: AsyncMock,
    ) -> None:
        """Test complete timeout coordination flow."""
        now = datetime.now(UTC)
        past_deadline = now - timedelta(minutes=5)

        # Set up query result
        ack_projection = create_mock_projection(
            state=EnumRegistrationState.AWAITING_ACK,
            ack_deadline=past_deadline,
        )
        mock_timeout_query.find_overdue_entities.return_value = (
            create_mock_query_result(
                ack_timeouts=[ack_projection],
                query_time=now,
            )
        )

        # Set up emission result
        mock_timeout_emission.process_timeouts.return_value = (
            create_mock_emission_result(
                ack_emitted=1,
                markers_updated=1,
            )
        )

        tick = create_mock_tick(now=now)
        result = await coordinator.coordinate(tick)

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
