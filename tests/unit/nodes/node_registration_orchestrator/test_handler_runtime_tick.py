# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for HandlerRuntimeTick.

Tests validate:
- Handler detects overdue ack deadlines and emits timeout events
- Handler uses projection.needs_ack_timeout_event() for deduplication
- Handler detects overdue liveness deadlines
- Handler uses injected `now` for all deadline comparisons

G2 Acceptance Criteria:
    5. test_handler_runtime_tick_detects_ack_timeout
    6. test_handler_runtime_tick_deduplicates_timeout

Related Tickets:
    - OMN-888 (C1): Registration Orchestrator
    - OMN-932 (C2): Durable Timeout Handling
    - G2: Test orchestrator logic
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest

from omnibase_core.models.dispatch.model_handler_output import ModelHandlerOutput
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_core.models.primitives.model_semver import ModelSemVer
from omnibase_infra.enums import EnumRegistrationState
from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.models.projection import ModelRegistrationProjection
from omnibase_infra.models.registration import ModelNodeCapabilities
from omnibase_infra.models.registration.events import (
    ModelNodeLivenessExpired,
    ModelNodeRegistrationAckTimedOut,
)
from omnibase_infra.nodes.node_registration_orchestrator.handlers.handler_runtime_tick import (
    HandlerRuntimeTick,
)
from omnibase_infra.projectors.projection_reader_registration import (
    ProjectionReaderRegistration,
)
from omnibase_infra.runtime.models.model_runtime_tick import ModelRuntimeTick

# Fixed test time for deterministic testing
TEST_NOW = datetime(2025, 1, 15, 12, 0, 0, tzinfo=UTC)


def create_mock_projection_reader() -> AsyncMock:
    """Create a mock ProjectionReaderRegistration."""
    mock = AsyncMock(spec=ProjectionReaderRegistration)
    mock.get_overdue_ack_registrations = AsyncMock(return_value=[])
    mock.get_overdue_liveness_registrations = AsyncMock(return_value=[])
    return mock


def create_runtime_tick(now: datetime = TEST_NOW) -> ModelRuntimeTick:
    """Create a test runtime tick."""
    return ModelRuntimeTick(
        now=now,
        tick_id=uuid4(),
        sequence_number=1,
        scheduled_at=now,
        correlation_id=uuid4(),
        scheduler_id="test-scheduler",
        tick_interval_ms=1000,
    )


def create_envelope(
    tick: ModelRuntimeTick,
    now: datetime | None = None,
    correlation_id: UUID | None = None,
) -> ModelEventEnvelope[ModelRuntimeTick]:
    """Create a test envelope wrapping a runtime tick."""
    return ModelEventEnvelope(
        envelope_id=uuid4(),
        payload=tick,
        envelope_timestamp=now or datetime.now(UTC),
        correlation_id=correlation_id or uuid4(),
        source="test",
    )


def create_projection(
    entity_id: UUID,
    state: EnumRegistrationState,
    ack_deadline: datetime | None = None,
    liveness_deadline: datetime | None = None,
    ack_timeout_emitted_at: datetime | None = None,
    liveness_timeout_emitted_at: datetime | None = None,
) -> ModelRegistrationProjection:
    """Create a test projection."""
    return ModelRegistrationProjection(
        entity_id=entity_id,
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
        last_applied_offset=0,
        registered_at=TEST_NOW - timedelta(hours=1),
        updated_at=TEST_NOW - timedelta(minutes=5),
    )


class TestHandlerRuntimeTickDetectsAckTimeout:
    """G2 Requirement 5: Handler detects ack timeout."""

    @pytest.mark.asyncio
    async def test_handler_runtime_tick_detects_ack_timeout(self) -> None:
        """Given projection with overdue ack_deadline,
        And projection.needs_ack_timeout_event() returns True,
        When handler processes RuntimeTick,
        Then emits ModelNodeRegistrationAckTimedOut.
        """
        # Arrange
        mock_reader = create_mock_projection_reader()

        node_id = uuid4()
        overdue_projection = create_projection(
            entity_id=node_id,
            state=EnumRegistrationState.AWAITING_ACK,
            ack_deadline=TEST_NOW - timedelta(minutes=5),  # Overdue
            ack_timeout_emitted_at=None,  # Not yet emitted
        )
        mock_reader.get_overdue_ack_registrations.return_value = [overdue_projection]
        mock_reader.get_overdue_liveness_registrations.return_value = []

        handler = HandlerRuntimeTick(mock_reader)
        tick = create_runtime_tick(now=TEST_NOW)
        envelope = create_envelope(
            tick, now=TEST_NOW, correlation_id=tick.correlation_id
        )

        # Act
        output = await handler.handle(envelope)

        # Assert
        assert isinstance(output, ModelHandlerOutput)
        assert output.handler_id == "handler-runtime-tick"
        assert len(output.events) == 1
        timeout_event = output.events[0]
        assert isinstance(timeout_event, ModelNodeRegistrationAckTimedOut)
        assert timeout_event.node_id == node_id
        assert timeout_event.entity_id == node_id
        assert timeout_event.correlation_id == tick.correlation_id
        assert timeout_event.causation_id == tick.tick_id
        assert timeout_event.emitted_at == TEST_NOW
        assert timeout_event.deadline_at == overdue_projection.ack_deadline

    @pytest.mark.asyncio
    async def test_emits_ack_timeout_for_awaiting_ack_state(self) -> None:
        """Test ack timeout detection for AWAITING_ACK state."""
        mock_reader = create_mock_projection_reader()

        node_id = uuid4()
        overdue_projection = create_projection(
            entity_id=node_id,
            state=EnumRegistrationState.AWAITING_ACK,
            ack_deadline=TEST_NOW - timedelta(minutes=1),
            ack_timeout_emitted_at=None,
        )
        mock_reader.get_overdue_ack_registrations.return_value = [overdue_projection]

        handler = HandlerRuntimeTick(mock_reader)
        tick = create_runtime_tick(now=TEST_NOW)
        envelope = create_envelope(
            tick, now=TEST_NOW, correlation_id=tick.correlation_id
        )

        output = await handler.handle(envelope)

        assert isinstance(output, ModelHandlerOutput)
        assert output.handler_id == "handler-runtime-tick"
        assert len(output.events) == 1
        assert isinstance(output.events[0], ModelNodeRegistrationAckTimedOut)

    @pytest.mark.asyncio
    async def test_emits_ack_timeout_for_accepted_state(self) -> None:
        """Test ack timeout detection for ACCEPTED state."""
        mock_reader = create_mock_projection_reader()

        node_id = uuid4()
        overdue_projection = create_projection(
            entity_id=node_id,
            state=EnumRegistrationState.ACCEPTED,
            ack_deadline=TEST_NOW - timedelta(minutes=1),
            ack_timeout_emitted_at=None,
        )
        mock_reader.get_overdue_ack_registrations.return_value = [overdue_projection]

        handler = HandlerRuntimeTick(mock_reader)
        tick = create_runtime_tick(now=TEST_NOW)
        envelope = create_envelope(
            tick, now=TEST_NOW, correlation_id=tick.correlation_id
        )

        output = await handler.handle(envelope)

        assert isinstance(output, ModelHandlerOutput)
        assert output.handler_id == "handler-runtime-tick"
        assert len(output.events) == 1
        assert isinstance(output.events[0], ModelNodeRegistrationAckTimedOut)


class TestHandlerRuntimeTickDeduplicatesTimeout:
    """G2 Requirement 6: Handler deduplicates timeout events."""

    @pytest.mark.asyncio
    async def test_handler_runtime_tick_deduplicates_timeout(self) -> None:
        """Given projection with overdue ack_deadline,
        But projection.needs_ack_timeout_event() returns False,
        When handler processes RuntimeTick,
        Then returns empty list (deduplication works).
        """
        # Arrange
        mock_reader = create_mock_projection_reader()

        node_id = uuid4()
        # Projection where timeout was already emitted
        already_emitted_projection = create_projection(
            entity_id=node_id,
            state=EnumRegistrationState.AWAITING_ACK,
            ack_deadline=TEST_NOW - timedelta(minutes=5),  # Overdue
            ack_timeout_emitted_at=TEST_NOW - timedelta(minutes=1),  # Already emitted!
        )
        # The reader returns this, but needs_ack_timeout_event() will return False
        mock_reader.get_overdue_ack_registrations.return_value = [
            already_emitted_projection
        ]
        mock_reader.get_overdue_liveness_registrations.return_value = []

        handler = HandlerRuntimeTick(mock_reader)
        tick = create_runtime_tick(now=TEST_NOW)
        envelope = create_envelope(
            tick, now=TEST_NOW, correlation_id=tick.correlation_id
        )

        # Act
        output = await handler.handle(envelope)

        # Assert - no events because deduplication filters it out
        assert isinstance(output, ModelHandlerOutput)
        assert output.handler_id == "handler-runtime-tick"
        assert output.events == ()

    @pytest.mark.asyncio
    async def test_no_duplicate_ack_timeout_when_already_emitted(self) -> None:
        """Test that ack timeout is not emitted twice."""
        mock_reader = create_mock_projection_reader()

        node_id = uuid4()
        # Projection with ack_timeout_emitted_at set (already processed)
        already_emitted = create_projection(
            entity_id=node_id,
            state=EnumRegistrationState.AWAITING_ACK,
            ack_deadline=TEST_NOW - timedelta(minutes=5),
            ack_timeout_emitted_at=TEST_NOW - timedelta(seconds=30),  # Already emitted
        )
        mock_reader.get_overdue_ack_registrations.return_value = [already_emitted]

        handler = HandlerRuntimeTick(mock_reader)
        tick = create_runtime_tick(now=TEST_NOW)
        envelope = create_envelope(
            tick, now=TEST_NOW, correlation_id=tick.correlation_id
        )

        output = await handler.handle(envelope)

        # Deduplication should prevent event emission
        assert isinstance(output, ModelHandlerOutput)
        assert output.handler_id == "handler-runtime-tick"
        assert output.events == ()


class TestHandlerRuntimeTickLivenessExpiry:
    """Test liveness deadline detection."""

    @pytest.mark.asyncio
    async def test_detects_liveness_expiry(self) -> None:
        """Test that liveness expiry is detected for active nodes."""
        mock_reader = create_mock_projection_reader()

        node_id = uuid4()
        overdue_projection = create_projection(
            entity_id=node_id,
            state=EnumRegistrationState.ACTIVE,
            liveness_deadline=TEST_NOW - timedelta(minutes=2),  # Overdue
            liveness_timeout_emitted_at=None,  # Not yet emitted
        )
        mock_reader.get_overdue_ack_registrations.return_value = []
        mock_reader.get_overdue_liveness_registrations.return_value = [
            overdue_projection
        ]

        handler = HandlerRuntimeTick(mock_reader)
        tick = create_runtime_tick(now=TEST_NOW)
        envelope = create_envelope(
            tick, now=TEST_NOW, correlation_id=tick.correlation_id
        )

        output = await handler.handle(envelope)

        assert isinstance(output, ModelHandlerOutput)
        assert output.handler_id == "handler-runtime-tick"
        assert len(output.events) == 1
        liveness_event = output.events[0]
        assert isinstance(liveness_event, ModelNodeLivenessExpired)
        assert liveness_event.node_id == node_id
        assert liveness_event.entity_id == node_id
        assert liveness_event.correlation_id == tick.correlation_id
        assert liveness_event.causation_id == tick.tick_id
        assert liveness_event.emitted_at == TEST_NOW
        # last_heartbeat_at is None when no heartbeat has ever been received.
        # Per ModelNodeLivenessExpired contract: "None if never received".
        # This is semantically correct - using registered_at would falsely imply
        # a heartbeat was received at registration time.
        assert liveness_event.last_heartbeat_at is None

    @pytest.mark.asyncio
    async def test_no_duplicate_liveness_expiry_when_already_emitted(self) -> None:
        """Test that liveness expiry is not emitted twice."""
        mock_reader = create_mock_projection_reader()

        node_id = uuid4()
        already_emitted = create_projection(
            entity_id=node_id,
            state=EnumRegistrationState.ACTIVE,
            liveness_deadline=TEST_NOW - timedelta(minutes=2),
            liveness_timeout_emitted_at=TEST_NOW - timedelta(seconds=30),  # Already!
        )
        mock_reader.get_overdue_liveness_registrations.return_value = [already_emitted]

        handler = HandlerRuntimeTick(mock_reader)
        tick = create_runtime_tick(now=TEST_NOW)
        envelope = create_envelope(
            tick, now=TEST_NOW, correlation_id=tick.correlation_id
        )

        output = await handler.handle(envelope)

        # Deduplication should prevent event emission
        assert isinstance(output, ModelHandlerOutput)
        assert output.handler_id == "handler-runtime-tick"
        assert output.events == ()


class TestHandlerRuntimeTickMultipleTimeouts:
    """Test handling of multiple timeout events."""

    @pytest.mark.asyncio
    async def test_emits_multiple_ack_timeouts(self) -> None:
        """Test that multiple overdue ack deadlines emit multiple events."""
        mock_reader = create_mock_projection_reader()

        # Create multiple overdue projections
        projections = []
        for _ in range(3):
            proj = create_projection(
                entity_id=uuid4(),
                state=EnumRegistrationState.AWAITING_ACK,
                ack_deadline=TEST_NOW - timedelta(minutes=5),
                ack_timeout_emitted_at=None,
            )
            projections.append(proj)

        mock_reader.get_overdue_ack_registrations.return_value = projections
        mock_reader.get_overdue_liveness_registrations.return_value = []

        handler = HandlerRuntimeTick(mock_reader)
        tick = create_runtime_tick(now=TEST_NOW)
        envelope = create_envelope(
            tick, now=TEST_NOW, correlation_id=tick.correlation_id
        )

        output = await handler.handle(envelope)

        assert isinstance(output, ModelHandlerOutput)
        assert output.handler_id == "handler-runtime-tick"
        assert len(output.events) == 3
        for event in output.events:
            assert isinstance(event, ModelNodeRegistrationAckTimedOut)

    @pytest.mark.asyncio
    async def test_emits_both_ack_and_liveness_timeouts(self) -> None:
        """Test that both ack and liveness timeouts can be emitted."""
        mock_reader = create_mock_projection_reader()

        # One ack timeout
        ack_overdue = create_projection(
            entity_id=uuid4(),
            state=EnumRegistrationState.AWAITING_ACK,
            ack_deadline=TEST_NOW - timedelta(minutes=5),
            ack_timeout_emitted_at=None,
        )
        # One liveness timeout
        liveness_overdue = create_projection(
            entity_id=uuid4(),
            state=EnumRegistrationState.ACTIVE,
            liveness_deadline=TEST_NOW - timedelta(minutes=2),
            liveness_timeout_emitted_at=None,
        )

        mock_reader.get_overdue_ack_registrations.return_value = [ack_overdue]
        mock_reader.get_overdue_liveness_registrations.return_value = [liveness_overdue]

        handler = HandlerRuntimeTick(mock_reader)
        tick = create_runtime_tick(now=TEST_NOW)
        envelope = create_envelope(
            tick, now=TEST_NOW, correlation_id=tick.correlation_id
        )

        output = await handler.handle(envelope)

        assert isinstance(output, ModelHandlerOutput)
        assert output.handler_id == "handler-runtime-tick"
        assert len(output.events) == 2
        # First event is ack timeout
        assert isinstance(output.events[0], ModelNodeRegistrationAckTimedOut)
        # Second event is liveness expiry
        assert isinstance(output.events[1], ModelNodeLivenessExpired)
        # last_heartbeat_at is None when no heartbeat has ever been received.
        # Per ModelNodeLivenessExpired contract: "None if never received".
        assert output.events[1].last_heartbeat_at is None


class TestHandlerRuntimeTickNoTimeouts:
    """Test handling when no timeouts are detected."""

    @pytest.mark.asyncio
    async def test_no_events_when_no_overdue_deadlines(self) -> None:
        """Test that no events are emitted when no deadlines are overdue."""
        mock_reader = create_mock_projection_reader()
        mock_reader.get_overdue_ack_registrations.return_value = []
        mock_reader.get_overdue_liveness_registrations.return_value = []

        handler = HandlerRuntimeTick(mock_reader)
        tick = create_runtime_tick(now=TEST_NOW)
        envelope = create_envelope(
            tick, now=TEST_NOW, correlation_id=tick.correlation_id
        )

        output = await handler.handle(envelope)

        assert isinstance(output, ModelHandlerOutput)
        assert output.handler_id == "handler-runtime-tick"
        assert output.events == ()


class TestHandlerRuntimeTickInjectedNow:
    """Test that handler uses injected `now` parameter from envelope."""

    @pytest.mark.asyncio
    async def test_uses_envelope_timestamp_for_ack_deadline_query(self) -> None:
        """Test that envelope timestamp is used for ack deadline queries."""
        mock_reader = create_mock_projection_reader()
        mock_reader.get_overdue_ack_registrations.return_value = []
        mock_reader.get_overdue_liveness_registrations.return_value = []

        handler = HandlerRuntimeTick(mock_reader)

        custom_now = datetime(2025, 6, 15, 10, 30, 0, tzinfo=UTC)
        correlation_id = uuid4()
        tick = create_runtime_tick(now=custom_now)
        envelope = create_envelope(tick, now=custom_now, correlation_id=correlation_id)

        await handler.handle(envelope)

        # Verify the reader was called with timestamp from envelope
        mock_reader.get_overdue_ack_registrations.assert_called_once_with(
            now=custom_now,
            domain="registration",
            correlation_id=correlation_id,
        )

    @pytest.mark.asyncio
    async def test_uses_envelope_timestamp_for_liveness_deadline_query(self) -> None:
        """Test that envelope timestamp is used for liveness deadline queries."""
        mock_reader = create_mock_projection_reader()
        mock_reader.get_overdue_ack_registrations.return_value = []
        mock_reader.get_overdue_liveness_registrations.return_value = []

        handler = HandlerRuntimeTick(mock_reader)

        custom_now = datetime(2025, 6, 15, 10, 30, 0, tzinfo=UTC)
        correlation_id = uuid4()
        tick = create_runtime_tick(now=custom_now)
        envelope = create_envelope(tick, now=custom_now, correlation_id=correlation_id)

        await handler.handle(envelope)

        # Verify the reader was called with timestamp from envelope
        mock_reader.get_overdue_liveness_registrations.assert_called_once_with(
            now=custom_now,
            domain="registration",
            correlation_id=correlation_id,
        )

    @pytest.mark.asyncio
    async def test_timeout_event_uses_envelope_timestamp_for_emitted_at(self) -> None:
        """Test that timeout events use envelope timestamp for emitted_at field."""
        mock_reader = create_mock_projection_reader()

        custom_now = datetime(2025, 6, 15, 10, 30, 0, tzinfo=UTC)
        node_id = uuid4()
        overdue_projection = create_projection(
            entity_id=node_id,
            state=EnumRegistrationState.AWAITING_ACK,
            ack_deadline=custom_now - timedelta(minutes=5),
            ack_timeout_emitted_at=None,
        )
        mock_reader.get_overdue_ack_registrations.return_value = [overdue_projection]

        handler = HandlerRuntimeTick(mock_reader)
        tick = create_runtime_tick(now=custom_now)
        envelope = create_envelope(
            tick, now=custom_now, correlation_id=tick.correlation_id
        )

        output = await handler.handle(envelope)

        assert isinstance(output, ModelHandlerOutput)
        assert output.handler_id == "handler-runtime-tick"
        assert len(output.events) == 1
        # emitted_at should be the envelope timestamp, not system time
        assert output.events[0].emitted_at == custom_now


class TestHandlerRuntimeTickTimezoneValidation:
    """Test that handler validates timezone-awareness of envelope timestamp."""

    @pytest.mark.asyncio
    async def test_raises_protocol_configuration_error_for_naive_datetime(self) -> None:
        """Test that handler raises ProtocolConfigurationError if envelope timestamp is naive."""
        mock_reader = create_mock_projection_reader()
        handler = HandlerRuntimeTick(mock_reader)

        # Create a naive datetime (no timezone info)
        naive_now = datetime(2025, 1, 15, 12, 0, 0)  # No tzinfo!
        assert naive_now.tzinfo is None  # Confirm it's naive

        tick = create_runtime_tick(now=TEST_NOW)
        envelope = create_envelope(tick, now=naive_now, correlation_id=uuid4())

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            await handler.handle(envelope)

        assert "timezone-aware" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_accepts_timezone_aware_datetime(self) -> None:
        """Test that handler accepts timezone-aware datetime."""
        mock_reader = create_mock_projection_reader()
        mock_reader.get_overdue_ack_registrations.return_value = []
        mock_reader.get_overdue_liveness_registrations.return_value = []

        handler = HandlerRuntimeTick(mock_reader)

        # Use timezone-aware datetime
        aware_now = datetime(2025, 1, 15, 12, 0, 0, tzinfo=UTC)
        assert aware_now.tzinfo is not None  # Confirm it's aware

        tick = create_runtime_tick(now=aware_now)
        correlation_id = uuid4()
        envelope = create_envelope(tick, now=aware_now, correlation_id=correlation_id)

        # Should not raise - timezone-aware datetime is valid
        output = await handler.handle(envelope)

        assert isinstance(output, ModelHandlerOutput)
        assert output.handler_id == "handler-runtime-tick"
        assert output.events == ()  # No events expected (empty projections)
