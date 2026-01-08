# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for HandlerNodeIntrospected.

Tests validate:
- Handler emits NodeRegistrationInitiated for new nodes
- Handler skips registration for nodes in blocking states
- Handler re-initiates registration for nodes in retriable states
- State decision matrix per C1 requirements

G2 Acceptance Criteria:
    3. test_handler_node_introspected_emits_initiated
    4. test_handler_node_introspected_skips_active_node

Related Tickets:
    - OMN-888 (C1): Registration Orchestrator
    - G2: Test orchestrator logic
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest

from omnibase_infra.enums import EnumRegistrationState
from omnibase_infra.models.projection import ModelRegistrationProjection
from omnibase_infra.models.registration import (
    ModelNodeCapabilities,
    ModelNodeIntrospectionEvent,
)
from omnibase_infra.models.registration.events import ModelNodeRegistrationInitiated
from omnibase_infra.nodes.node_registration_orchestrator.handlers.handler_node_introspected import (
    HandlerNodeIntrospected,
)
from omnibase_infra.projectors.projection_reader_registration import (
    ProjectionReaderRegistration,
)

# Fixed test time for deterministic testing
TEST_NOW = datetime(2025, 1, 15, 12, 0, 0, tzinfo=UTC)


def create_mock_projection_reader() -> AsyncMock:
    """Create a mock ProjectionReaderRegistration."""
    mock = AsyncMock(spec=ProjectionReaderRegistration)
    mock.get_entity_state = AsyncMock(return_value=None)
    return mock


def create_projection(
    entity_id: UUID,
    state: EnumRegistrationState,
) -> ModelRegistrationProjection:
    """Create a test projection."""
    return ModelRegistrationProjection(
        entity_id=entity_id,
        domain="registration",
        current_state=state,
        node_type="effect",
        node_version="1.0.0",
        capabilities=ModelNodeCapabilities(),
        last_applied_event_id=uuid4(),
        last_applied_offset=0,
        registered_at=TEST_NOW - timedelta(hours=1),
        updated_at=TEST_NOW - timedelta(minutes=5),
    )


def create_introspection_event(
    node_id: UUID | None = None,
    timestamp: datetime | None = None,
) -> ModelNodeIntrospectionEvent:
    """Create a test introspection event."""
    return ModelNodeIntrospectionEvent(
        node_id=node_id or uuid4(),
        node_type="effect",
        correlation_id=uuid4(),
        timestamp=timestamp or TEST_NOW,
    )


class TestHandlerNodeIntrospectedEmitsInitiated:
    """G2 Requirement 3: Handler emits NodeRegistrationInitiated for new nodes."""

    @pytest.mark.asyncio
    async def test_handler_node_introspected_emits_initiated(self) -> None:
        """Given projection returns None (new node),
        When handler processes NodeIntrospectionEvent,
        Then emits ModelNodeRegistrationInitiated,
        And event.emitted_at equals injected `now`.
        """
        # Arrange
        mock_reader = create_mock_projection_reader()
        mock_reader.get_entity_state.return_value = None  # New node

        handler = HandlerNodeIntrospected(mock_reader)

        node_id = uuid4()
        correlation_id = uuid4()
        introspection_event = create_introspection_event(node_id=node_id)

        # Act
        events = await handler.handle(
            event=introspection_event,
            now=TEST_NOW,
            correlation_id=correlation_id,
        )

        # Assert
        assert len(events) == 1
        initiated = events[0]
        assert isinstance(initiated, ModelNodeRegistrationInitiated)
        assert initiated.node_id == node_id
        assert initiated.entity_id == node_id
        assert initiated.correlation_id == correlation_id
        # Causation ID should link to triggering event
        assert initiated.causation_id == introspection_event.correlation_id
        # Registration attempt ID should be generated
        assert initiated.registration_attempt_id is not None
        # Verify time injection: emitted_at must equal injected `now`
        assert initiated.emitted_at == TEST_NOW

    @pytest.mark.asyncio
    async def test_emits_initiated_for_new_node(self) -> None:
        """Test that new nodes (no projection) trigger registration."""
        mock_reader = create_mock_projection_reader()
        mock_reader.get_entity_state.return_value = None

        handler = HandlerNodeIntrospected(mock_reader)
        introspection_event = create_introspection_event()

        events = await handler.handle(
            event=introspection_event,
            now=TEST_NOW,
            correlation_id=uuid4(),
        )

        assert len(events) == 1
        assert isinstance(events[0], ModelNodeRegistrationInitiated)


class TestHandlerNodeIntrospectedSkipsBlockingStates:
    """G2 Requirement 4: Handler skips registration for nodes in blocking states."""

    @pytest.mark.asyncio
    async def test_handler_node_introspected_skips_active_node(self) -> None:
        """Given projection returns state=ACTIVE,
        When handler processes NodeIntrospectionEvent,
        Then returns empty list (no events).
        """
        # Arrange
        mock_reader = create_mock_projection_reader()
        node_id = uuid4()
        active_projection = create_projection(
            entity_id=node_id,
            state=EnumRegistrationState.ACTIVE,
        )
        mock_reader.get_entity_state.return_value = active_projection

        handler = HandlerNodeIntrospected(mock_reader)
        introspection_event = create_introspection_event(node_id=node_id)

        # Act
        events = await handler.handle(
            event=introspection_event,
            now=TEST_NOW,
            correlation_id=uuid4(),
        )

        # Assert
        assert events == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "blocking_state",
        [
            EnumRegistrationState.PENDING_REGISTRATION,
            EnumRegistrationState.ACCEPTED,
            EnumRegistrationState.AWAITING_ACK,
            EnumRegistrationState.ACK_RECEIVED,
            EnumRegistrationState.ACTIVE,
        ],
    )
    async def test_skips_nodes_in_blocking_states(
        self, blocking_state: EnumRegistrationState
    ) -> None:
        """Test that nodes in blocking states don't trigger new registration."""
        mock_reader = create_mock_projection_reader()
        node_id = uuid4()
        blocking_projection = create_projection(
            entity_id=node_id,
            state=blocking_state,
        )
        mock_reader.get_entity_state.return_value = blocking_projection

        handler = HandlerNodeIntrospected(mock_reader)
        introspection_event = create_introspection_event(node_id=node_id)

        events = await handler.handle(
            event=introspection_event,
            now=TEST_NOW,
            correlation_id=uuid4(),
        )

        assert events == [], f"Expected no events for state {blocking_state}"


class TestHandlerNodeIntrospectedRetriableStates:
    """Test that nodes in retriable states can re-register."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "retriable_state",
        [
            EnumRegistrationState.LIVENESS_EXPIRED,
            EnumRegistrationState.REJECTED,
            EnumRegistrationState.ACK_TIMED_OUT,
        ],
    )
    async def test_emits_initiated_for_retriable_states(
        self, retriable_state: EnumRegistrationState
    ) -> None:
        """Test that nodes in retriable states trigger new registration."""
        mock_reader = create_mock_projection_reader()
        node_id = uuid4()
        retriable_projection = create_projection(
            entity_id=node_id,
            state=retriable_state,
        )
        mock_reader.get_entity_state.return_value = retriable_projection

        handler = HandlerNodeIntrospected(mock_reader)
        introspection_event = create_introspection_event(node_id=node_id)

        events = await handler.handle(
            event=introspection_event,
            now=TEST_NOW,
            correlation_id=uuid4(),
        )

        assert len(events) == 1
        assert isinstance(events[0], ModelNodeRegistrationInitiated)
        assert events[0].node_id == node_id

    @pytest.mark.asyncio
    async def test_emits_initiated_for_liveness_expired_state(self) -> None:
        """Test that LIVENESS_EXPIRED state allows re-registration."""
        mock_reader = create_mock_projection_reader()
        node_id = uuid4()
        expired_projection = create_projection(
            entity_id=node_id,
            state=EnumRegistrationState.LIVENESS_EXPIRED,
        )
        mock_reader.get_entity_state.return_value = expired_projection

        handler = HandlerNodeIntrospected(mock_reader)
        introspection_event = create_introspection_event(node_id=node_id)

        events = await handler.handle(
            event=introspection_event,
            now=TEST_NOW,
            correlation_id=uuid4(),
        )

        assert len(events) == 1
        assert isinstance(events[0], ModelNodeRegistrationInitiated)

    @pytest.mark.asyncio
    async def test_emits_initiated_for_rejected_state(self) -> None:
        """Test that REJECTED state allows retry registration."""
        mock_reader = create_mock_projection_reader()
        node_id = uuid4()
        rejected_projection = create_projection(
            entity_id=node_id,
            state=EnumRegistrationState.REJECTED,
        )
        mock_reader.get_entity_state.return_value = rejected_projection

        handler = HandlerNodeIntrospected(mock_reader)
        introspection_event = create_introspection_event(node_id=node_id)

        events = await handler.handle(
            event=introspection_event,
            now=TEST_NOW,
            correlation_id=uuid4(),
        )

        assert len(events) == 1
        assert isinstance(events[0], ModelNodeRegistrationInitiated)

    @pytest.mark.asyncio
    async def test_emits_initiated_for_ack_timed_out_state(self) -> None:
        """Test that ACK_TIMED_OUT state allows retry registration."""
        mock_reader = create_mock_projection_reader()
        node_id = uuid4()
        timed_out_projection = create_projection(
            entity_id=node_id,
            state=EnumRegistrationState.ACK_TIMED_OUT,
        )
        mock_reader.get_entity_state.return_value = timed_out_projection

        handler = HandlerNodeIntrospected(mock_reader)
        introspection_event = create_introspection_event(node_id=node_id)

        events = await handler.handle(
            event=introspection_event,
            now=TEST_NOW,
            correlation_id=uuid4(),
        )

        assert len(events) == 1
        assert isinstance(events[0], ModelNodeRegistrationInitiated)


class TestHandlerNodeIntrospectedEventFields:
    """Test that emitted events have correct field values."""

    @pytest.mark.asyncio
    async def test_registration_attempt_id_is_unique(self) -> None:
        """Test that each registration attempt gets a unique ID."""
        mock_reader = create_mock_projection_reader()
        mock_reader.get_entity_state.return_value = None

        handler = HandlerNodeIntrospected(mock_reader)

        # Process same event twice
        introspection_event = create_introspection_event()

        events1 = await handler.handle(
            event=introspection_event,
            now=TEST_NOW,
            correlation_id=uuid4(),
        )
        events2 = await handler.handle(
            event=introspection_event,
            now=TEST_NOW,
            correlation_id=uuid4(),
        )

        # Both should succeed
        assert len(events1) == 1
        assert len(events2) == 1

        # But registration attempt IDs should differ
        assert events1[0].registration_attempt_id != events2[0].registration_attempt_id

    @pytest.mark.asyncio
    async def test_causation_id_links_to_introspection_event(self) -> None:
        """Test that causation_id links to the triggering introspection event."""
        mock_reader = create_mock_projection_reader()
        mock_reader.get_entity_state.return_value = None

        handler = HandlerNodeIntrospected(mock_reader)

        introspection_correlation_id = uuid4()
        introspection_event = ModelNodeIntrospectionEvent(
            node_id=uuid4(),
            node_type="effect",
            correlation_id=introspection_correlation_id,
            timestamp=TEST_NOW,
        )

        events = await handler.handle(
            event=introspection_event,
            now=TEST_NOW,
            correlation_id=uuid4(),
        )

        assert len(events) == 1
        # Causation should link to the introspection event's correlation ID
        assert events[0].causation_id == introspection_correlation_id

    @pytest.mark.asyncio
    async def test_entity_id_equals_node_id(self) -> None:
        """Test that entity_id equals node_id for registration domain."""
        mock_reader = create_mock_projection_reader()
        mock_reader.get_entity_state.return_value = None

        handler = HandlerNodeIntrospected(mock_reader)

        node_id = uuid4()
        introspection_event = create_introspection_event(node_id=node_id)

        events = await handler.handle(
            event=introspection_event,
            now=TEST_NOW,
            correlation_id=uuid4(),
        )

        assert len(events) == 1
        assert events[0].entity_id == node_id
        assert events[0].node_id == node_id
        assert events[0].entity_id == events[0].node_id


class TestHandlerNodeIntrospectedProjectionQueries:
    """Test projection reader interactions."""

    @pytest.mark.asyncio
    async def test_queries_projection_with_correct_params(self) -> None:
        """Test that projection is queried with correct parameters."""
        mock_reader = create_mock_projection_reader()
        mock_reader.get_entity_state.return_value = None

        handler = HandlerNodeIntrospected(mock_reader)

        node_id = uuid4()
        correlation_id = uuid4()
        introspection_event = create_introspection_event(node_id=node_id)

        await handler.handle(
            event=introspection_event,
            now=TEST_NOW,
            correlation_id=correlation_id,
        )

        mock_reader.get_entity_state.assert_called_once_with(
            entity_id=node_id,
            domain="registration",
            correlation_id=correlation_id,
        )


class TestHandlerNodeIntrospectedTimezoneValidation:
    """Test that handler validates timezone-awareness of now parameter."""

    @pytest.mark.asyncio
    async def test_raises_value_error_for_naive_datetime(self) -> None:
        """Test that handler raises ValueError if now is naive (no tzinfo)."""
        mock_reader = create_mock_projection_reader()
        handler = HandlerNodeIntrospected(mock_reader)

        # Create a naive datetime (no timezone info)
        naive_now = datetime(2025, 1, 15, 12, 0, 0)  # No tzinfo!
        assert naive_now.tzinfo is None  # Confirm it's naive

        introspection_event = create_introspection_event(node_id=uuid4())

        with pytest.raises(ValueError) as exc_info:
            await handler.handle(
                event=introspection_event,
                now=naive_now,
                correlation_id=uuid4(),
            )

        assert "timezone-aware" in str(exc_info.value)
        assert "naive" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_accepts_timezone_aware_datetime(self) -> None:
        """Test that handler accepts timezone-aware datetime."""
        mock_reader = create_mock_projection_reader()
        mock_reader.get_entity_state.return_value = None

        handler = HandlerNodeIntrospected(mock_reader)

        # Use timezone-aware datetime
        aware_now = datetime(2025, 1, 15, 12, 0, 0, tzinfo=UTC)
        assert aware_now.tzinfo is not None  # Confirm it's aware

        introspection_event = create_introspection_event(node_id=uuid4())

        # Should not raise - timezone-aware datetime is valid
        events = await handler.handle(
            event=introspection_event,
            now=aware_now,
            correlation_id=uuid4(),
        )

        assert len(events) == 1  # New node triggers registration


def create_mock_projector() -> AsyncMock:
    """Create a mock ProjectorRegistration."""
    from omnibase_infra.projectors import ProjectorRegistration

    mock = AsyncMock(spec=ProjectorRegistration)
    mock.persist_state_transition = AsyncMock(return_value=True)
    return mock


class TestHandlerNodeIntrospectedProjectionPersistence:
    """Test projection persistence when projector is configured."""

    @pytest.mark.asyncio
    async def test_persists_projection_when_projector_configured(self) -> None:
        """Test that projection is persisted when projector is provided."""
        mock_reader = create_mock_projection_reader()
        mock_reader.get_entity_state.return_value = None  # New node

        mock_projector = create_mock_projector()

        handler = HandlerNodeIntrospected(
            projection_reader=mock_reader,
            projector=mock_projector,
        )

        node_id = uuid4()
        correlation_id = uuid4()
        introspection_event = create_introspection_event(node_id=node_id)

        events = await handler.handle(
            event=introspection_event,
            now=TEST_NOW,
            correlation_id=correlation_id,
        )

        # Should emit event
        assert len(events) == 1
        assert isinstance(events[0], ModelNodeRegistrationInitiated)

        # Should have called projector to persist state transition
        mock_projector.persist_state_transition.assert_called_once()

        # Verify correct parameters were passed
        call_kwargs = mock_projector.persist_state_transition.call_args[1]
        assert call_kwargs["entity_id"] == node_id
        assert call_kwargs["domain"] == "registration"
        assert call_kwargs["new_state"] == EnumRegistrationState.PENDING_REGISTRATION
        assert call_kwargs["node_type"] == "effect"
        assert call_kwargs["now"] == TEST_NOW
        assert call_kwargs["correlation_id"] == correlation_id

    @pytest.mark.asyncio
    async def test_does_not_persist_when_no_projector(self) -> None:
        """Test that no persistence happens when projector is None."""
        mock_reader = create_mock_projection_reader()
        mock_reader.get_entity_state.return_value = None  # New node

        # No projector provided
        handler = HandlerNodeIntrospected(projection_reader=mock_reader)

        node_id = uuid4()
        introspection_event = create_introspection_event(node_id=node_id)

        events = await handler.handle(
            event=introspection_event,
            now=TEST_NOW,
            correlation_id=uuid4(),
        )

        # Should still emit event
        assert len(events) == 1
        assert isinstance(events[0], ModelNodeRegistrationInitiated)

    @pytest.mark.asyncio
    async def test_has_projector_property(self) -> None:
        """Test the has_projector property reflects configuration."""
        mock_reader = create_mock_projection_reader()
        mock_projector = create_mock_projector()

        handler_with_projector = HandlerNodeIntrospected(
            projection_reader=mock_reader,
            projector=mock_projector,
        )
        assert handler_with_projector.has_projector is True

        handler_without_projector = HandlerNodeIntrospected(
            projection_reader=mock_reader,
        )
        assert handler_without_projector.has_projector is False

    @pytest.mark.asyncio
    async def test_projection_uses_capabilities_from_event(self) -> None:
        """Test that projection uses capabilities from introspection event."""
        mock_reader = create_mock_projection_reader()
        mock_reader.get_entity_state.return_value = None

        mock_projector = create_mock_projector()

        handler = HandlerNodeIntrospected(
            projection_reader=mock_reader,
            projector=mock_projector,
        )

        # Create event with specific capabilities
        capabilities = ModelNodeCapabilities(
            postgres=True,
            read=True,
            write=True,
        )
        introspection_event = ModelNodeIntrospectionEvent(
            node_id=uuid4(),
            node_type="effect",
            node_version="2.0.0",
            capabilities=capabilities,
            correlation_id=uuid4(),
            timestamp=TEST_NOW,
        )

        await handler.handle(
            event=introspection_event,
            now=TEST_NOW,
            correlation_id=uuid4(),
        )

        # Verify capabilities were passed to projector
        call_kwargs = mock_projector.persist_state_transition.call_args[1]
        assert call_kwargs["capabilities"] == capabilities
        assert call_kwargs["node_version"] == "2.0.0"

    @pytest.mark.asyncio
    async def test_projection_calculates_ack_deadline(self) -> None:
        """Test that ack_deadline is calculated correctly."""
        mock_reader = create_mock_projection_reader()
        mock_reader.get_entity_state.return_value = None

        mock_projector = create_mock_projector()

        # Use custom ack timeout
        ack_timeout_seconds = 60.0
        handler = HandlerNodeIntrospected(
            projection_reader=mock_reader,
            projector=mock_projector,
            ack_timeout_seconds=ack_timeout_seconds,
        )

        introspection_event = create_introspection_event()

        await handler.handle(
            event=introspection_event,
            now=TEST_NOW,
            correlation_id=uuid4(),
        )

        # Verify ack_deadline was calculated
        call_kwargs = mock_projector.persist_state_transition.call_args[1]
        expected_deadline = TEST_NOW + timedelta(seconds=ack_timeout_seconds)
        assert call_kwargs["ack_deadline"] == expected_deadline

    @pytest.mark.asyncio
    async def test_default_ack_timeout(self) -> None:
        """Test that default ack timeout is 30 seconds."""
        mock_reader = create_mock_projection_reader()
        mock_reader.get_entity_state.return_value = None

        mock_projector = create_mock_projector()

        # Use default timeout
        handler = HandlerNodeIntrospected(
            projection_reader=mock_reader,
            projector=mock_projector,
        )

        introspection_event = create_introspection_event()

        await handler.handle(
            event=introspection_event,
            now=TEST_NOW,
            correlation_id=uuid4(),
        )

        # Verify default ack_deadline (30 seconds)
        call_kwargs = mock_projector.persist_state_transition.call_args[1]
        expected_deadline = TEST_NOW + timedelta(seconds=30.0)
        assert call_kwargs["ack_deadline"] == expected_deadline

    @pytest.mark.asyncio
    async def test_does_not_persist_for_blocking_states(self) -> None:
        """Test that no persistence happens for nodes in blocking states."""
        mock_reader = create_mock_projection_reader()
        node_id = uuid4()

        # Node is already active
        mock_reader.get_entity_state.return_value = create_projection(
            entity_id=node_id,
            state=EnumRegistrationState.ACTIVE,
        )

        mock_projector = create_mock_projector()

        handler = HandlerNodeIntrospected(
            projection_reader=mock_reader,
            projector=mock_projector,
        )

        introspection_event = create_introspection_event(node_id=node_id)

        events = await handler.handle(
            event=introspection_event,
            now=TEST_NOW,
            correlation_id=uuid4(),
        )

        # Should not emit events (node already active)
        assert events == []

        # Should NOT call projector
        mock_projector.persist_state_transition.assert_not_called()

    @pytest.mark.asyncio
    async def test_persists_for_retriable_states(self) -> None:
        """Test that projection is persisted for re-registration from retriable states."""
        mock_reader = create_mock_projection_reader()
        node_id = uuid4()

        # Node has expired liveness
        mock_reader.get_entity_state.return_value = create_projection(
            entity_id=node_id,
            state=EnumRegistrationState.LIVENESS_EXPIRED,
        )

        mock_projector = create_mock_projector()

        handler = HandlerNodeIntrospected(
            projection_reader=mock_reader,
            projector=mock_projector,
        )

        introspection_event = create_introspection_event(node_id=node_id)
        correlation_id = uuid4()

        events = await handler.handle(
            event=introspection_event,
            now=TEST_NOW,
            correlation_id=correlation_id,
        )

        # Should emit re-registration event
        assert len(events) == 1
        assert isinstance(events[0], ModelNodeRegistrationInitiated)

        # Should persist projection for re-registration
        mock_projector.persist_state_transition.assert_called_once()
        call_kwargs = mock_projector.persist_state_transition.call_args[1]
        assert call_kwargs["new_state"] == EnumRegistrationState.PENDING_REGISTRATION


def create_mock_consul_handler() -> AsyncMock:
    """Create a mock HandlerConsul."""
    from omnibase_infra.handlers import HandlerConsul

    mock = AsyncMock(spec=HandlerConsul)
    mock.execute = AsyncMock(return_value=None)
    return mock


class TestHandlerNodeIntrospectedConsulRegistration:
    """Test Consul registration (dual registration) functionality."""

    @pytest.mark.asyncio
    async def test_has_consul_handler_property(self) -> None:
        """Test the has_consul_handler property reflects configuration."""
        mock_reader = create_mock_projection_reader()
        mock_consul = create_mock_consul_handler()

        handler_with_consul = HandlerNodeIntrospected(
            projection_reader=mock_reader,
            consul_handler=mock_consul,
        )
        assert handler_with_consul.has_consul_handler is True

        handler_without_consul = HandlerNodeIntrospected(
            projection_reader=mock_reader,
        )
        assert handler_without_consul.has_consul_handler is False

    @pytest.mark.asyncio
    async def test_registers_with_consul_when_handler_provided(self) -> None:
        """Test that Consul registration is called when handler is provided."""
        mock_reader = create_mock_projection_reader()
        mock_reader.get_entity_state.return_value = None  # New node

        mock_consul = create_mock_consul_handler()

        handler = HandlerNodeIntrospected(
            projection_reader=mock_reader,
            consul_handler=mock_consul,
        )

        node_id = uuid4()
        introspection_event = create_introspection_event(node_id=node_id)

        events = await handler.handle(
            event=introspection_event,
            now=TEST_NOW,
            correlation_id=uuid4(),
        )

        # Should emit event
        assert len(events) == 1
        assert isinstance(events[0], ModelNodeRegistrationInitiated)

        # Should have called HandlerConsul.execute
        mock_consul.execute.assert_called_once()

        # Verify the call had correct structure
        call_args = mock_consul.execute.call_args[0][0]
        assert call_args["operation"] == "consul.register"
        assert "payload" in call_args

        payload = call_args["payload"]
        # Service name follows ONEX convention: onex-{node_type}
        assert payload["name"] == "onex-effect"
        assert payload["service_id"] == f"onex-effect-{node_id}"
        assert "onex" in payload["tags"]

    @pytest.mark.asyncio
    async def test_skips_consul_when_no_handler(self) -> None:
        """Test that Consul registration is skipped when no handler provided."""
        mock_reader = create_mock_projection_reader()
        mock_reader.get_entity_state.return_value = None  # New node

        # No consul handler
        handler = HandlerNodeIntrospected(projection_reader=mock_reader)

        node_id = uuid4()
        introspection_event = create_introspection_event(node_id=node_id)

        # Should not raise - just skip Consul registration
        events = await handler.handle(
            event=introspection_event,
            now=TEST_NOW,
            correlation_id=uuid4(),
        )

        assert len(events) == 1
        assert isinstance(events[0], ModelNodeRegistrationInitiated)

    @pytest.mark.asyncio
    async def test_continues_on_consul_error(self) -> None:
        """Test that handler continues even if Consul registration fails."""
        mock_reader = create_mock_projection_reader()
        mock_reader.get_entity_state.return_value = None  # New node

        mock_consul = create_mock_consul_handler()
        mock_consul.execute.side_effect = Exception("Consul connection failed")

        handler = HandlerNodeIntrospected(
            projection_reader=mock_reader,
            consul_handler=mock_consul,
        )

        node_id = uuid4()
        introspection_event = create_introspection_event(node_id=node_id)

        # Should not raise - Consul failure is non-fatal
        events = await handler.handle(
            event=introspection_event,
            now=TEST_NOW,
            correlation_id=uuid4(),
        )

        # Should still emit event despite Consul failure
        assert len(events) == 1
        assert isinstance(events[0], ModelNodeRegistrationInitiated)

        # Consul execute should have been attempted
        mock_consul.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_consul_registration_uses_correct_service_name_format(self) -> None:
        """Test service name follows ONEX convention: onex-{node_type}."""
        mock_reader = create_mock_projection_reader()
        mock_reader.get_entity_state.return_value = None

        mock_consul = create_mock_consul_handler()

        handler = HandlerNodeIntrospected(
            projection_reader=mock_reader,
            consul_handler=mock_consul,
        )

        node_id = uuid4()
        # Service name follows ONEX convention: onex-{node_type}
        # create_introspection_event uses node_type="effect" by default
        expected_service_name = "onex-effect"

        introspection_event = create_introspection_event(node_id=node_id)

        await handler.handle(
            event=introspection_event,
            now=TEST_NOW,
            correlation_id=uuid4(),
        )

        call_args = mock_consul.execute.call_args[0][0]
        payload = call_args["payload"]

        # Service name should match ONEX convention
        assert payload["name"] == expected_service_name

    @pytest.mark.asyncio
    async def test_consul_registration_extracts_address_from_endpoints(self) -> None:
        """Test that address/port are extracted from endpoint URLs."""
        mock_reader = create_mock_projection_reader()
        mock_reader.get_entity_state.return_value = None

        mock_consul = create_mock_consul_handler()

        handler = HandlerNodeIntrospected(
            projection_reader=mock_reader,
            consul_handler=mock_consul,
        )

        node_id = uuid4()
        introspection_event = ModelNodeIntrospectionEvent(
            node_id=node_id,
            node_type="effect",
            endpoints={
                "health": "http://test-node:8080/health",
                "api": "http://test-node:8080/api",
            },
            correlation_id=uuid4(),
            timestamp=TEST_NOW,
        )

        await handler.handle(
            event=introspection_event,
            now=TEST_NOW,
            correlation_id=uuid4(),
        )

        call_args = mock_consul.execute.call_args[0][0]
        payload = call_args["payload"]

        # Should extract address and port from endpoint URL
        assert payload.get("address") == "test-node"
        assert payload.get("port") == 8080

    @pytest.mark.asyncio
    async def test_consul_registration_with_full_dual_registration(self) -> None:
        """Test full dual registration with both PostgreSQL and Consul."""
        mock_reader = create_mock_projection_reader()
        mock_reader.get_entity_state.return_value = None  # New node

        mock_projector = create_mock_projector()
        mock_consul = create_mock_consul_handler()

        handler = HandlerNodeIntrospected(
            projection_reader=mock_reader,
            projector=mock_projector,
            consul_handler=mock_consul,
        )

        node_id = uuid4()
        introspection_event = create_introspection_event(node_id=node_id)

        events = await handler.handle(
            event=introspection_event,
            now=TEST_NOW,
            correlation_id=uuid4(),
        )

        # Should emit event
        assert len(events) == 1
        assert isinstance(events[0], ModelNodeRegistrationInitiated)

        # Both PostgreSQL and Consul should be called
        mock_projector.persist_state_transition.assert_called_once()
        mock_consul.execute.assert_called_once()
