# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for HandlerNodeIntrospected.

Tests validate:
- Handler emits NodeRegistrationInitiated for new nodes
- Handler skips registration for nodes in blocking states
- Handler re-initiates registration for nodes in retriable states
- Handler returns intents for effect layer execution (OMN-2050)
- State decision matrix per C1 requirements

G2 Acceptance Criteria:
    3. test_handler_node_introspected_emits_initiated
    4. test_handler_node_introspected_skips_active_node

Related Tickets:
    - OMN-888 (C1): Registration Orchestrator
    - G2: Test orchestrator logic
    - OMN-2050: Wire MessageDispatchEngine as single consumer path
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest

from omnibase_core.enums import EnumNodeKind
from omnibase_core.models.dispatch.model_handler_output import ModelHandlerOutput
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_core.models.primitives.model_semver import ModelSemVer
from omnibase_core.models.reducer.model_intent import ModelIntent
from omnibase_infra.enums import EnumRegistrationState
from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.models.projection import ModelRegistrationProjection
from omnibase_infra.models.registration import (
    ModelNodeCapabilities,
    ModelNodeIntrospectionEvent,
)
from omnibase_infra.models.registration.events import ModelNodeRegistrationInitiated
from omnibase_infra.nodes.node_registration_orchestrator.handlers.handler_node_introspected import (
    HandlerNodeIntrospected,
)
from omnibase_infra.nodes.reducers.models.model_payload_consul_register import (
    ModelPayloadConsulRegister,
)
from omnibase_infra.nodes.reducers.models.model_payload_postgres_upsert_registration import (
    ModelPayloadPostgresUpsertRegistration,
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
        node_type=EnumNodeKind.EFFECT,
        node_version=ModelSemVer.parse("1.0.0"),
        capabilities=ModelNodeCapabilities(),  # ModelRegistrationProjection uses capabilities
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
        node_type=EnumNodeKind.EFFECT,
        correlation_id=uuid4(),
        timestamp=timestamp or TEST_NOW,
    )


def create_envelope(
    event: ModelNodeIntrospectionEvent,
    now: datetime | None = None,
    correlation_id: UUID | None = None,
) -> ModelEventEnvelope[ModelNodeIntrospectionEvent]:
    """Create a test envelope wrapping an introspection event."""
    return ModelEventEnvelope(
        envelope_id=uuid4(),
        payload=event,
        envelope_timestamp=now or TEST_NOW,
        correlation_id=correlation_id or uuid4(),
        source="test",
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
        envelope = create_envelope(
            introspection_event, now=TEST_NOW, correlation_id=correlation_id
        )

        # Act
        output = await handler.handle(envelope)

        # Assert
        assert isinstance(output, ModelHandlerOutput)
        assert output.handler_id == "handler-node-introspected"
        assert len(output.events) == 1
        initiated = output.events[0]
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
        envelope = create_envelope(introspection_event, now=TEST_NOW)

        output = await handler.handle(envelope)

        assert isinstance(output, ModelHandlerOutput)
        assert len(output.events) == 1
        assert isinstance(output.events[0], ModelNodeRegistrationInitiated)


class TestHandlerNodeIntrospectedSkipsBlockingStates:
    """G2 Requirement 4: Handler skips registration for nodes in blocking states."""

    @pytest.mark.asyncio
    async def test_handler_node_introspected_skips_active_node(self) -> None:
        """Given projection returns state=ACTIVE,
        When handler processes NodeIntrospectionEvent,
        Then returns empty events tuple.
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
        envelope = create_envelope(introspection_event, now=TEST_NOW)

        # Act
        output = await handler.handle(envelope)

        # Assert
        assert isinstance(output, ModelHandlerOutput)
        assert output.handler_id == "handler-node-introspected"
        assert len(output.events) == 0

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
        envelope = create_envelope(introspection_event, now=TEST_NOW)

        output = await handler.handle(envelope)

        assert isinstance(output, ModelHandlerOutput)
        assert len(output.events) == 0, f"Expected no events for state {blocking_state}"
        assert len(output.intents) == 0, (
            f"Expected no intents for state {blocking_state}"
        )


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
        envelope = create_envelope(introspection_event, now=TEST_NOW)

        output = await handler.handle(envelope)

        assert isinstance(output, ModelHandlerOutput)
        assert len(output.events) == 1
        assert isinstance(output.events[0], ModelNodeRegistrationInitiated)
        assert output.events[0].node_id == node_id

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
        envelope = create_envelope(introspection_event, now=TEST_NOW)

        output = await handler.handle(envelope)

        assert isinstance(output, ModelHandlerOutput)
        assert len(output.events) == 1
        assert isinstance(output.events[0], ModelNodeRegistrationInitiated)

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
        envelope = create_envelope(introspection_event, now=TEST_NOW)

        output = await handler.handle(envelope)

        assert isinstance(output, ModelHandlerOutput)
        assert len(output.events) == 1
        assert isinstance(output.events[0], ModelNodeRegistrationInitiated)

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
        envelope = create_envelope(introspection_event, now=TEST_NOW)

        output = await handler.handle(envelope)

        assert isinstance(output, ModelHandlerOutput)
        assert len(output.events) == 1
        assert isinstance(output.events[0], ModelNodeRegistrationInitiated)


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
        envelope1 = create_envelope(introspection_event, now=TEST_NOW)
        envelope2 = create_envelope(introspection_event, now=TEST_NOW)

        output1 = await handler.handle(envelope1)
        output2 = await handler.handle(envelope2)

        # Both should succeed
        assert len(output1.events) == 1
        assert len(output2.events) == 1

        # But registration attempt IDs should differ
        assert (
            output1.events[0].registration_attempt_id
            != output2.events[0].registration_attempt_id
        )

    @pytest.mark.asyncio
    async def test_causation_id_links_to_introspection_event(self) -> None:
        """Test that causation_id links to the triggering introspection event."""
        mock_reader = create_mock_projection_reader()
        mock_reader.get_entity_state.return_value = None

        handler = HandlerNodeIntrospected(mock_reader)

        introspection_correlation_id = uuid4()
        introspection_event = ModelNodeIntrospectionEvent(
            node_id=uuid4(),
            node_type=EnumNodeKind.EFFECT,
            correlation_id=introspection_correlation_id,
            timestamp=TEST_NOW,
        )
        envelope = create_envelope(introspection_event, now=TEST_NOW)

        output = await handler.handle(envelope)

        assert len(output.events) == 1
        # Causation should link to the introspection event's correlation ID
        assert output.events[0].causation_id == introspection_correlation_id

    @pytest.mark.asyncio
    async def test_entity_id_equals_node_id(self) -> None:
        """Test that entity_id equals node_id for registration domain."""
        mock_reader = create_mock_projection_reader()
        mock_reader.get_entity_state.return_value = None

        handler = HandlerNodeIntrospected(mock_reader)

        node_id = uuid4()
        introspection_event = create_introspection_event(node_id=node_id)
        envelope = create_envelope(introspection_event, now=TEST_NOW)

        output = await handler.handle(envelope)

        assert len(output.events) == 1
        assert output.events[0].entity_id == node_id
        assert output.events[0].node_id == node_id
        assert output.events[0].entity_id == output.events[0].node_id


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
        envelope = create_envelope(
            introspection_event, now=TEST_NOW, correlation_id=correlation_id
        )

        await handler.handle(envelope)

        mock_reader.get_entity_state.assert_called_once_with(
            entity_id=node_id,
            domain="registration",
            correlation_id=correlation_id,
        )


class TestHandlerNodeIntrospectedTimezoneValidation:
    """Test that handler validates timezone-awareness of envelope timestamp."""

    @pytest.mark.asyncio
    async def test_raises_protocol_configuration_error_for_naive_datetime(self) -> None:
        """Test that handler raises ProtocolConfigurationError if envelope_timestamp is naive (no tzinfo)."""
        mock_reader = create_mock_projection_reader()
        handler = HandlerNodeIntrospected(mock_reader)

        # Create a naive datetime (no timezone info)
        naive_now = datetime(2025, 1, 15, 12, 0, 0)  # No tzinfo!
        assert naive_now.tzinfo is None  # Confirm it's naive

        introspection_event = create_introspection_event(node_id=uuid4())
        envelope = create_envelope(introspection_event, now=naive_now)

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            await handler.handle(envelope)

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
        envelope = create_envelope(introspection_event, now=aware_now)

        # Should not raise - timezone-aware datetime is valid
        output = await handler.handle(envelope)

        assert isinstance(output, ModelHandlerOutput)
        assert len(output.events) == 1  # New node triggers registration


class TestHandlerNodeIntrospectedIntents:
    """Test intent-based output for effect layer execution (OMN-2050)."""

    @pytest.mark.asyncio
    async def test_returns_postgres_and_consul_intents(self) -> None:
        """Test that handler returns both postgres and consul intents."""
        mock_reader = create_mock_projection_reader()
        mock_reader.get_entity_state.return_value = None  # New node

        handler = HandlerNodeIntrospected(mock_reader)

        node_id = uuid4()
        correlation_id = uuid4()
        introspection_event = create_introspection_event(node_id=node_id)
        envelope = create_envelope(
            introspection_event, now=TEST_NOW, correlation_id=correlation_id
        )

        output = await handler.handle(envelope)

        # Should have 2 intents: postgres upsert + consul register
        assert len(output.intents) == 2

        # Verify intent types
        intent_payload_types = [type(i.payload).__name__ for i in output.intents]
        assert "ModelPayloadPostgresUpsertRegistration" in intent_payload_types
        assert "ModelPayloadConsulRegister" in intent_payload_types

    @pytest.mark.asyncio
    async def test_postgres_intent_has_correct_record(self) -> None:
        """Test that postgres intent has correct registration record data."""
        mock_reader = create_mock_projection_reader()
        mock_reader.get_entity_state.return_value = None

        handler = HandlerNodeIntrospected(mock_reader)

        node_id = uuid4()
        correlation_id = uuid4()
        introspection_event = create_introspection_event(node_id=node_id)
        envelope = create_envelope(
            introspection_event, now=TEST_NOW, correlation_id=correlation_id
        )

        output = await handler.handle(envelope)

        # Find the postgres intent
        postgres_intents = [
            i
            for i in output.intents
            if isinstance(i.payload, ModelPayloadPostgresUpsertRegistration)
        ]
        assert len(postgres_intents) == 1

        postgres_payload = postgres_intents[0].payload
        assert postgres_payload.correlation_id == correlation_id

        # Verify record contains expected fields
        # record is a BaseModel subclass (extra="allow"); convert to dict for access
        record = postgres_payload.record.model_dump()
        assert record["entity_id"] == node_id
        assert record["domain"] == "registration"
        assert (
            record["current_state"] == EnumRegistrationState.PENDING_REGISTRATION.value
        )
        assert record["node_type"] == "effect"

        # Verify intent envelope
        assert postgres_intents[0].intent_type
        assert f"{node_id}" in postgres_intents[0].target

    @pytest.mark.asyncio
    async def test_consul_intent_has_correct_service_name(self) -> None:
        """Test that consul intent follows ONEX service naming convention."""
        mock_reader = create_mock_projection_reader()
        mock_reader.get_entity_state.return_value = None

        handler = HandlerNodeIntrospected(mock_reader)

        node_id = uuid4()
        introspection_event = create_introspection_event(node_id=node_id)
        envelope = create_envelope(introspection_event, now=TEST_NOW)

        output = await handler.handle(envelope)

        # Find the consul intent
        consul_intents = [
            i
            for i in output.intents
            if isinstance(i.payload, ModelPayloadConsulRegister)
        ]
        assert len(consul_intents) == 1

        consul_payload = consul_intents[0].payload
        assert consul_payload.service_name == "onex-effect"
        assert consul_payload.service_id == f"onex-effect-{node_id}"
        assert "onex" in consul_payload.tags
        assert "node-type:effect" in consul_payload.tags

    @pytest.mark.asyncio
    async def test_no_intents_for_blocking_states(self) -> None:
        """Test that no intents are returned for nodes in blocking states."""
        mock_reader = create_mock_projection_reader()
        node_id = uuid4()

        # Node is already active
        mock_reader.get_entity_state.return_value = create_projection(
            entity_id=node_id,
            state=EnumRegistrationState.ACTIVE,
        )

        handler = HandlerNodeIntrospected(mock_reader)

        introspection_event = create_introspection_event(node_id=node_id)
        envelope = create_envelope(introspection_event, now=TEST_NOW)

        output = await handler.handle(envelope)

        assert len(output.events) == 0
        assert len(output.intents) == 0

    @pytest.mark.asyncio
    async def test_intents_for_retriable_states(self) -> None:
        """Test that intents are returned for re-registration from retriable states."""
        mock_reader = create_mock_projection_reader()
        node_id = uuid4()

        # Node has expired liveness
        mock_reader.get_entity_state.return_value = create_projection(
            entity_id=node_id,
            state=EnumRegistrationState.LIVENESS_EXPIRED,
        )

        handler = HandlerNodeIntrospected(mock_reader)

        introspection_event = create_introspection_event(node_id=node_id)
        envelope = create_envelope(introspection_event, now=TEST_NOW)

        output = await handler.handle(envelope)

        # Should have events and intents for re-registration
        assert len(output.events) == 1
        assert isinstance(output.events[0], ModelNodeRegistrationInitiated)
        assert len(output.intents) == 2

    @pytest.mark.asyncio
    async def test_ack_deadline_in_postgres_intent(self) -> None:
        """Test that ack_deadline is calculated correctly in postgres intent."""
        mock_reader = create_mock_projection_reader()
        mock_reader.get_entity_state.return_value = None

        ack_timeout_seconds = 60.0
        handler = HandlerNodeIntrospected(
            mock_reader,
            ack_timeout_seconds=ack_timeout_seconds,
        )

        introspection_event = create_introspection_event()
        envelope = create_envelope(introspection_event, now=TEST_NOW)

        output = await handler.handle(envelope)

        # Find postgres intent and verify ack_deadline
        postgres_intents = [
            i
            for i in output.intents
            if isinstance(i.payload, ModelPayloadPostgresUpsertRegistration)
        ]
        assert len(postgres_intents) == 1

        record = postgres_intents[0].payload.record.model_dump()
        expected_deadline = TEST_NOW + timedelta(seconds=ack_timeout_seconds)
        assert record["ack_deadline"] == expected_deadline

    @pytest.mark.asyncio
    async def test_default_ack_timeout_in_intent(self) -> None:
        """Test that default ack timeout (30s) is used in intent."""
        mock_reader = create_mock_projection_reader()
        mock_reader.get_entity_state.return_value = None

        handler = HandlerNodeIntrospected(mock_reader)

        introspection_event = create_introspection_event()
        envelope = create_envelope(introspection_event, now=TEST_NOW)

        output = await handler.handle(envelope)

        postgres_intents = [
            i
            for i in output.intents
            if isinstance(i.payload, ModelPayloadPostgresUpsertRegistration)
        ]
        record = postgres_intents[0].payload.record.model_dump()
        expected_deadline = TEST_NOW + timedelta(seconds=30.0)
        assert record["ack_deadline"] == expected_deadline

    @pytest.mark.asyncio
    async def test_capabilities_in_postgres_intent(self) -> None:
        """Test that capabilities from event are included in postgres intent."""
        mock_reader = create_mock_projection_reader()
        mock_reader.get_entity_state.return_value = None

        handler = HandlerNodeIntrospected(mock_reader)

        capabilities = ModelNodeCapabilities(
            postgres=True,
            read=True,
            write=True,
        )
        introspection_event = ModelNodeIntrospectionEvent(
            node_id=uuid4(),
            node_type=EnumNodeKind.EFFECT,
            node_version=ModelSemVer.parse("2.0.0"),
            declared_capabilities=capabilities,
            correlation_id=uuid4(),
            timestamp=TEST_NOW,
        )
        envelope = create_envelope(introspection_event, now=TEST_NOW)

        output = await handler.handle(envelope)

        postgres_intents = [
            i
            for i in output.intents
            if isinstance(i.payload, ModelPayloadPostgresUpsertRegistration)
        ]
        record = postgres_intents[0].payload.record.model_dump()
        assert record["capabilities"] == capabilities.model_dump(mode="json")
        assert record["node_version"] == "2.0.0"


class TestSanitizeToolName:
    """Test MCP tool name sanitization for Consul tags."""

    def test_sanitize_simple_name(self) -> None:
        """Test that simple names pass through with lowercase."""
        mock_reader = create_mock_projection_reader()
        handler = HandlerNodeIntrospected(projection_reader=mock_reader)

        result = handler._sanitize_tool_name("MyTool")
        assert result == "mytool"

    def test_sanitize_spaces_become_dashes(self) -> None:
        """Test that spaces are replaced with dashes."""
        mock_reader = create_mock_projection_reader()
        handler = HandlerNodeIntrospected(projection_reader=mock_reader)

        result = handler._sanitize_tool_name("My Cool Tool")
        assert result == "my-cool-tool"

    def test_sanitize_special_chars_become_dashes(self) -> None:
        """Test that special characters are replaced with dashes."""
        mock_reader = create_mock_projection_reader()
        handler = HandlerNodeIntrospected(projection_reader=mock_reader)

        result = handler._sanitize_tool_name("Tool (v2.0) - Beta!")
        assert result == "tool-v2-0-beta"

    def test_sanitize_multiple_special_chars_collapse(self) -> None:
        """Test that multiple consecutive special chars become single dash."""
        mock_reader = create_mock_projection_reader()
        handler = HandlerNodeIntrospected(projection_reader=mock_reader)

        result = handler._sanitize_tool_name("Tool   &&&   Name")
        assert result == "tool-name"

    def test_sanitize_leading_trailing_dashes_removed(self) -> None:
        """Test that leading/trailing dashes are removed."""
        mock_reader = create_mock_projection_reader()
        handler = HandlerNodeIntrospected(projection_reader=mock_reader)

        result = handler._sanitize_tool_name("  @@@Tool Name###  ")
        assert result == "tool-name"

    def test_sanitize_preserves_alphanumeric(self) -> None:
        """Test that alphanumeric characters are preserved."""
        mock_reader = create_mock_projection_reader()
        handler = HandlerNodeIntrospected(projection_reader=mock_reader)

        result = handler._sanitize_tool_name("Tool123Name456")
        assert result == "tool123name456"

    def test_sanitize_truncates_to_63_chars(self) -> None:
        """Test that result is truncated to 63 characters (DNS label limit)."""
        mock_reader = create_mock_projection_reader()
        handler = HandlerNodeIntrospected(projection_reader=mock_reader)

        long_name = "a" * 100  # 100 characters
        result = handler._sanitize_tool_name(long_name)
        assert len(result) == 63
        assert result == "a" * 63

    def test_sanitize_free_form_description(self) -> None:
        """Test sanitization of typical free-form description text."""
        mock_reader = create_mock_projection_reader()
        handler = HandlerNodeIntrospected(projection_reader=mock_reader)

        # Typical description that might come from metadata.description
        description = "Node Registration Orchestrator (handles service discovery)"
        result = handler._sanitize_tool_name(description)
        assert result == "node-registration-orchestrator-handles-service-discovery"

    def test_sanitize_unicode_chars(self) -> None:
        """Test that unicode characters are replaced with dashes."""
        mock_reader = create_mock_projection_reader()
        handler = HandlerNodeIntrospected(projection_reader=mock_reader)

        result = handler._sanitize_tool_name("Tool\u2019s Name")  # Smart apostrophe
        assert result == "tool-s-name"

    def test_sanitize_empty_after_sanitization(self) -> None:
        """Test handling of names that become empty after sanitization."""
        mock_reader = create_mock_projection_reader()
        handler = HandlerNodeIntrospected(projection_reader=mock_reader)

        # All special characters
        result = handler._sanitize_tool_name("@#$%^&*()")
        assert result == ""


class TestCapabilitiesJsonbCompatibility:
    """Test that capabilities serialization produces asyncpg-compatible JSONB values.

    The handler uses model_dump(mode="json") for capabilities. asyncpg's
    JSONB codec expects Python dicts, not JSON strings. This test ensures
    the serialization contract holds.
    """

    @pytest.mark.asyncio
    async def test_capabilities_model_dump_json_returns_dict(self) -> None:
        """model_dump(mode='json') must return a dict (not a JSON string) for JSONB."""
        mock_reader = create_mock_projection_reader()
        mock_reader.get_entity_state.return_value = None

        handler = HandlerNodeIntrospected(mock_reader)

        capabilities = ModelNodeCapabilities(
            postgres=True,
            read=True,
            write=True,
        )
        introspection_event = ModelNodeIntrospectionEvent(
            node_id=uuid4(),
            node_type=EnumNodeKind.EFFECT,
            declared_capabilities=capabilities,
            correlation_id=uuid4(),
            timestamp=TEST_NOW,
        )
        envelope = create_envelope(introspection_event, now=TEST_NOW)

        output = await handler.handle(envelope)

        # Find postgres intent and inspect capabilities
        postgres_intents = [
            i
            for i in output.intents
            if isinstance(i.payload, ModelPayloadPostgresUpsertRegistration)
        ]
        assert len(postgres_intents) == 1

        record = postgres_intents[0].payload.record.model_dump()
        caps = record["capabilities"]

        # asyncpg JSONB codec expects a Python dict, not a JSON string
        assert isinstance(caps, dict), (
            f"Expected dict for JSONB, got {type(caps).__name__}: {caps!r}"
        )

        # Verify the dict contains expected capability fields
        assert caps.get("postgres") is True
        assert caps.get("read") is True
        assert caps.get("write") is True

    @pytest.mark.asyncio
    async def test_empty_capabilities_returns_empty_dict(self) -> None:
        """Default capabilities should serialize to empty dict for JSONB."""
        mock_reader = create_mock_projection_reader()
        mock_reader.get_entity_state.return_value = None

        handler = HandlerNodeIntrospected(mock_reader)

        introspection_event = create_introspection_event()
        envelope = create_envelope(introspection_event, now=TEST_NOW)

        output = await handler.handle(envelope)

        postgres_intents = [
            i
            for i in output.intents
            if isinstance(i.payload, ModelPayloadPostgresUpsertRegistration)
        ]
        record = postgres_intents[0].payload.record.model_dump()
        caps = record["capabilities"]

        assert isinstance(caps, dict)
        # Default ModelNodeCapabilities with no params should produce a dict
