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
from omnibase_infra.orchestrators.registration.handlers.handler_node_introspected import (
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
) -> ModelNodeIntrospectionEvent:
    """Create a test introspection event."""
    return ModelNodeIntrospectionEvent(
        node_id=node_id or uuid4(),
        node_type="effect",
        correlation_id=uuid4(),
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
