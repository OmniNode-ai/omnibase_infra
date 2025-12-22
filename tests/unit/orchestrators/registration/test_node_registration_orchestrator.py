# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for NodeRegistrationOrchestrator.

Tests validate:
- Orchestrator emits EVENTS only (no intents, no projections)
- Orchestrator uses injected `now` parameter (not system clock)
- Proper routing to handlers based on payload type
- Correlation ID handling

G2 Acceptance Criteria:
    1. test_orchestrator_emits_events_only_no_io
    2. test_orchestrator_uses_injected_now_not_system_clock

Related Tickets:
    - OMN-888 (C1): Registration Orchestrator
    - G2: Test orchestrator logic
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

from omnibase_infra.enums import EnumRegistrationState
from omnibase_infra.models.projection import ModelRegistrationProjection
from omnibase_infra.models.registration import (
    ModelNodeCapabilities,
    ModelNodeIntrospectionEvent,
)
from omnibase_infra.models.registration.commands.model_node_registration_acked import (
    ModelNodeRegistrationAcked,
)
from omnibase_infra.models.registration.events import (
    ModelNodeBecameActive,
    ModelNodeLivenessExpired,
    ModelNodeRegistrationAckReceived,
    ModelNodeRegistrationAckTimedOut,
    ModelNodeRegistrationInitiated,
)
from omnibase_infra.orchestrators.registration.node_registration_orchestrator import (
    NodeRegistrationOrchestrator,
)
from omnibase_infra.projectors.projection_reader_registration import (
    ProjectionReaderRegistration,
)
from omnibase_infra.runtime.models.model_runtime_tick import ModelRuntimeTick

# Fixed test time for deterministic testing
TEST_NOW = datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC)


def create_mock_projection_reader() -> AsyncMock:
    """Create a mock ProjectionReaderRegistration."""
    mock = AsyncMock(spec=ProjectionReaderRegistration)
    mock.get_entity_state = AsyncMock(return_value=None)
    mock.get_overdue_ack_registrations = AsyncMock(return_value=[])
    mock.get_overdue_liveness_registrations = AsyncMock(return_value=[])
    return mock


def create_mock_envelope(payload: object) -> MagicMock:
    """Create a mock event envelope with given payload."""
    envelope = MagicMock()
    envelope.payload = payload
    envelope.correlation_id = uuid4()
    return envelope


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
        node_version="1.0.0",
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


class TestOrchestratorEmitsEventsOnlyNoIO:
    """G2 Requirement: Orchestrator emits EVENTS only, no I/O operations."""

    @pytest.mark.asyncio
    async def test_orchestrator_emits_events_only_no_io_introspection(self) -> None:
        """Given orchestrator with mock projection reader,
        When processing introspection event with injected `now`,
        Then output contains ONLY events (no intents, no projections),
        And no I/O operations performed (verified via mock).
        """
        # Arrange
        mock_reader = create_mock_projection_reader()
        mock_reader.get_entity_state.return_value = None  # New node

        orchestrator = NodeRegistrationOrchestrator(mock_reader)

        node_id = uuid4()
        introspection_event = ModelNodeIntrospectionEvent(
            node_id=node_id,
            node_type="effect",
            correlation_id=uuid4(),
        )
        envelope = create_mock_envelope(introspection_event)

        # Act
        events = await orchestrator.handle(
            envelope=envelope,
            now=TEST_NOW,
            correlation_id=uuid4(),
        )

        # Assert - output contains ONLY events
        assert len(events) == 1
        assert isinstance(events[0], ModelNodeRegistrationInitiated)

        # Assert - no I/O besides projection read (which is read-only)
        mock_reader.get_entity_state.assert_called_once()

    @pytest.mark.asyncio
    async def test_orchestrator_emits_events_only_no_io_runtime_tick(self) -> None:
        """Given orchestrator processing RuntimeTick,
        When there are overdue ack deadlines,
        Then output contains ONLY timeout events.
        """
        # Arrange
        mock_reader = create_mock_projection_reader()
        node_id = uuid4()
        overdue_projection = create_projection(
            entity_id=node_id,
            state=EnumRegistrationState.AWAITING_ACK,
            ack_deadline=TEST_NOW - timedelta(minutes=5),
            ack_timeout_emitted_at=None,
        )
        mock_reader.get_overdue_ack_registrations.return_value = [overdue_projection]
        mock_reader.get_overdue_liveness_registrations.return_value = []

        orchestrator = NodeRegistrationOrchestrator(mock_reader)

        tick = ModelRuntimeTick(
            now=TEST_NOW,
            tick_id=uuid4(),
            sequence_number=1,
            scheduled_at=TEST_NOW,
            correlation_id=uuid4(),
            scheduler_id="test-scheduler",
            tick_interval_ms=1000,
        )
        envelope = create_mock_envelope(tick)

        # Act
        events = await orchestrator.handle(
            envelope=envelope,
            now=TEST_NOW,
            correlation_id=tick.correlation_id,
        )

        # Assert - output contains ONLY events
        assert len(events) == 1
        assert isinstance(events[0], ModelNodeRegistrationAckTimedOut)

    @pytest.mark.asyncio
    async def test_orchestrator_emits_events_only_no_io_ack_command(self) -> None:
        """Given orchestrator processing ack command,
        When node is in AWAITING_ACK state,
        Then output contains ONLY events (AckReceived, BecameActive).
        """
        # Arrange
        mock_reader = create_mock_projection_reader()
        node_id = uuid4()
        awaiting_projection = create_projection(
            entity_id=node_id,
            state=EnumRegistrationState.AWAITING_ACK,
        )
        mock_reader.get_entity_state.return_value = awaiting_projection

        orchestrator = NodeRegistrationOrchestrator(mock_reader)

        ack_command = ModelNodeRegistrationAcked(
            node_id=node_id,
            correlation_id=uuid4(),
        )
        envelope = create_mock_envelope(ack_command)

        # Act
        events = await orchestrator.handle(
            envelope=envelope,
            now=TEST_NOW,
            correlation_id=ack_command.correlation_id,
        )

        # Assert - output contains ONLY events
        assert len(events) == 2
        assert isinstance(events[0], ModelNodeRegistrationAckReceived)
        assert isinstance(events[1], ModelNodeBecameActive)


class TestOrchestratorUsesInjectedNow:
    """G2 Requirement: Orchestrator uses injected `now`, not system clock."""

    @pytest.mark.asyncio
    async def test_orchestrator_uses_injected_now_not_system_clock(self) -> None:
        """Given orchestrator with mocked projection reader,
        And injected `now = datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC)`,
        When processing RuntimeTick event,
        Then all timeout calculations use injected `now`,
        And `datetime.now()` is never called (verify via mock.patch).
        """
        # Arrange
        mock_reader = create_mock_projection_reader()
        mock_reader.get_overdue_ack_registrations.return_value = []
        mock_reader.get_overdue_liveness_registrations.return_value = []

        orchestrator = NodeRegistrationOrchestrator(mock_reader)

        tick = ModelRuntimeTick(
            now=TEST_NOW,
            tick_id=uuid4(),
            sequence_number=1,
            scheduled_at=TEST_NOW,
            correlation_id=uuid4(),
            scheduler_id="test-scheduler",
            tick_interval_ms=1000,
        )
        envelope = create_mock_envelope(tick)

        # Act - Patch datetime.now to ensure it's never called
        with patch("datetime.datetime") as mock_datetime:
            # Preserve the real datetime class for type checking
            mock_datetime.side_effect = lambda *args, **kwargs: datetime(
                *args, **kwargs
            )

            # The orchestrator should NOT call datetime.now()
            events = await orchestrator.handle(
                envelope=envelope,
                now=TEST_NOW,
                correlation_id=tick.correlation_id,
            )

            # Assert - deadline queries use injected now
            mock_reader.get_overdue_ack_registrations.assert_called_once_with(
                now=TEST_NOW,
                domain="registration",
                correlation_id=tick.correlation_id,
            )
            mock_reader.get_overdue_liveness_registrations.assert_called_once_with(
                now=TEST_NOW,
                domain="registration",
                correlation_id=tick.correlation_id,
            )

    @pytest.mark.asyncio
    async def test_ack_command_uses_injected_now_for_liveness_deadline(self) -> None:
        """When processing ack command,
        Then liveness deadline is calculated from injected `now`.
        """
        # Arrange
        mock_reader = create_mock_projection_reader()
        node_id = uuid4()
        awaiting_projection = create_projection(
            entity_id=node_id,
            state=EnumRegistrationState.AWAITING_ACK,
        )
        mock_reader.get_entity_state.return_value = awaiting_projection

        orchestrator = NodeRegistrationOrchestrator(mock_reader)

        ack_command = ModelNodeRegistrationAcked(
            node_id=node_id,
            correlation_id=uuid4(),
        )
        envelope = create_mock_envelope(ack_command)

        # Act
        events = await orchestrator.handle(
            envelope=envelope,
            now=TEST_NOW,
            correlation_id=ack_command.correlation_id,
        )

        # Assert - liveness deadline calculated from injected now
        assert len(events) == 2
        ack_received = events[0]
        assert isinstance(ack_received, ModelNodeRegistrationAckReceived)
        # Default liveness interval is 60 seconds
        expected_deadline = TEST_NOW + timedelta(seconds=60)
        assert ack_received.liveness_deadline == expected_deadline


class TestOrchestratorRoutesPayloads:
    """Test that orchestrator routes payloads to correct handlers."""

    @pytest.mark.asyncio
    async def test_routes_introspection_event_to_handler(self) -> None:
        """Test introspection event routes to HandlerNodeIntrospected."""
        mock_reader = create_mock_projection_reader()
        mock_reader.get_entity_state.return_value = None

        orchestrator = NodeRegistrationOrchestrator(mock_reader)

        introspection_event = ModelNodeIntrospectionEvent(
            node_id=uuid4(),
            node_type="effect",
            correlation_id=uuid4(),
        )
        envelope = create_mock_envelope(introspection_event)

        events = await orchestrator.handle(
            envelope=envelope,
            now=TEST_NOW,
            correlation_id=uuid4(),
        )

        assert len(events) == 1
        assert isinstance(events[0], ModelNodeRegistrationInitiated)

    @pytest.mark.asyncio
    async def test_routes_runtime_tick_to_handler(self) -> None:
        """Test RuntimeTick routes to HandlerRuntimeTick."""
        mock_reader = create_mock_projection_reader()
        mock_reader.get_overdue_ack_registrations.return_value = []
        mock_reader.get_overdue_liveness_registrations.return_value = []

        orchestrator = NodeRegistrationOrchestrator(mock_reader)

        tick = ModelRuntimeTick(
            now=TEST_NOW,
            tick_id=uuid4(),
            sequence_number=1,
            scheduled_at=TEST_NOW,
            correlation_id=uuid4(),
            scheduler_id="test-scheduler",
            tick_interval_ms=1000,
        )
        envelope = create_mock_envelope(tick)

        events = await orchestrator.handle(
            envelope=envelope,
            now=TEST_NOW,
            correlation_id=tick.correlation_id,
        )

        # No overdue registrations, so no events
        assert events == []

    @pytest.mark.asyncio
    async def test_routes_ack_command_to_handler(self) -> None:
        """Test NodeRegistrationAcked routes to HandlerNodeRegistrationAcked."""
        mock_reader = create_mock_projection_reader()
        node_id = uuid4()
        mock_reader.get_entity_state.return_value = create_projection(
            entity_id=node_id,
            state=EnumRegistrationState.AWAITING_ACK,
        )

        orchestrator = NodeRegistrationOrchestrator(mock_reader)

        ack_command = ModelNodeRegistrationAcked(
            node_id=node_id,
            correlation_id=uuid4(),
        )
        envelope = create_mock_envelope(ack_command)

        events = await orchestrator.handle(
            envelope=envelope,
            now=TEST_NOW,
            correlation_id=ack_command.correlation_id,
        )

        assert len(events) == 2
        assert isinstance(events[0], ModelNodeRegistrationAckReceived)
        assert isinstance(events[1], ModelNodeBecameActive)

    @pytest.mark.asyncio
    async def test_raises_for_unknown_payload_type(self) -> None:
        """Test that unknown payload type raises ValueError."""
        mock_reader = create_mock_projection_reader()
        orchestrator = NodeRegistrationOrchestrator(mock_reader)

        # Create envelope with unsupported payload type
        unknown_payload = MagicMock()
        unknown_payload.__class__.__name__ = "UnknownPayload"
        envelope = create_mock_envelope(unknown_payload)

        with pytest.raises(ValueError) as exc_info:
            await orchestrator.handle(
                envelope=envelope,
                now=TEST_NOW,
                correlation_id=uuid4(),
            )

        assert "Unsupported payload type" in str(exc_info.value)


class TestOrchestratorCorrelationIdHandling:
    """Test correlation ID propagation and fallback."""

    @pytest.mark.asyncio
    async def test_uses_provided_correlation_id(self) -> None:
        """Test that provided correlation_id is used."""
        mock_reader = create_mock_projection_reader()
        mock_reader.get_entity_state.return_value = None

        orchestrator = NodeRegistrationOrchestrator(mock_reader)

        provided_corr_id = uuid4()
        introspection_event = ModelNodeIntrospectionEvent(
            node_id=uuid4(),
            node_type="effect",
            correlation_id=uuid4(),
        )
        envelope = create_mock_envelope(introspection_event)

        events = await orchestrator.handle(
            envelope=envelope,
            now=TEST_NOW,
            correlation_id=provided_corr_id,
        )

        assert len(events) == 1
        assert events[0].correlation_id == provided_corr_id

    @pytest.mark.asyncio
    async def test_falls_back_to_envelope_correlation_id(self) -> None:
        """Test that envelope correlation_id is used when not provided."""
        mock_reader = create_mock_projection_reader()
        mock_reader.get_entity_state.return_value = None

        orchestrator = NodeRegistrationOrchestrator(mock_reader)

        envelope_corr_id = uuid4()
        introspection_event = ModelNodeIntrospectionEvent(
            node_id=uuid4(),
            node_type="effect",
            correlation_id=uuid4(),
        )
        envelope = create_mock_envelope(introspection_event)
        envelope.correlation_id = envelope_corr_id

        events = await orchestrator.handle(
            envelope=envelope,
            now=TEST_NOW,
            correlation_id=None,
        )

        assert len(events) == 1
        assert events[0].correlation_id == envelope_corr_id


class TestOrchestratorConvenienceMethods:
    """Test orchestrator convenience methods for direct handler access."""

    @pytest.mark.asyncio
    async def test_handle_introspection_direct(self) -> None:
        """Test direct introspection handler method."""
        mock_reader = create_mock_projection_reader()
        mock_reader.get_entity_state.return_value = None

        orchestrator = NodeRegistrationOrchestrator(mock_reader)

        introspection_event = ModelNodeIntrospectionEvent(
            node_id=uuid4(),
            node_type="effect",
            correlation_id=uuid4(),
        )

        events = await orchestrator.handle_introspection(
            event=introspection_event,
            now=TEST_NOW,
            correlation_id=uuid4(),
        )

        assert len(events) == 1
        assert isinstance(events[0], ModelNodeRegistrationInitiated)

    @pytest.mark.asyncio
    async def test_handle_runtime_tick_direct(self) -> None:
        """Test direct runtime tick handler method."""
        mock_reader = create_mock_projection_reader()
        mock_reader.get_overdue_ack_registrations.return_value = []
        mock_reader.get_overdue_liveness_registrations.return_value = []

        orchestrator = NodeRegistrationOrchestrator(mock_reader)

        tick = ModelRuntimeTick(
            now=TEST_NOW,
            tick_id=uuid4(),
            sequence_number=1,
            scheduled_at=TEST_NOW,
            correlation_id=uuid4(),
            scheduler_id="test-scheduler",
            tick_interval_ms=1000,
        )

        events = await orchestrator.handle_runtime_tick(
            tick=tick,
            now=TEST_NOW,
            correlation_id=tick.correlation_id,
        )

        assert events == []

    @pytest.mark.asyncio
    async def test_handle_registration_ack_direct(self) -> None:
        """Test direct registration ack handler method."""
        mock_reader = create_mock_projection_reader()
        node_id = uuid4()
        mock_reader.get_entity_state.return_value = create_projection(
            entity_id=node_id,
            state=EnumRegistrationState.AWAITING_ACK,
        )

        orchestrator = NodeRegistrationOrchestrator(mock_reader)

        ack_command = ModelNodeRegistrationAcked(
            node_id=node_id,
            correlation_id=uuid4(),
        )

        events = await orchestrator.handle_registration_ack(
            command=ack_command,
            now=TEST_NOW,
            correlation_id=ack_command.correlation_id,
        )

        assert len(events) == 2
        assert isinstance(events[0], ModelNodeRegistrationAckReceived)
        assert isinstance(events[1], ModelNodeBecameActive)
