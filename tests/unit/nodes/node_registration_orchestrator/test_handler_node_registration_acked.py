# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for HandlerNodeRegistrationAcked.

Tests validate:
- Handler emits AckReceived and BecameActive for valid acks
- Handler ignores duplicate acks (idempotent)
- Handler handles acks in various FSM states
- Liveness deadline calculation uses injected `now`

G2 Acceptance Criteria:
    7. test_handler_acked_emits_active_events
    8. test_handler_acked_ignores_duplicate

Related Tickets:
    - OMN-888 (C1): Registration Orchestrator
    - OMN-889 (D1): Registration Reducer
    - G2: Test orchestrator logic
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_core.models.primitives.model_semver import ModelSemVer
from omnibase_infra.enums import EnumRegistrationState
from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.models.projection import ModelRegistrationProjection
from omnibase_infra.models.registration import ModelNodeCapabilities
from omnibase_infra.models.registration.commands.model_node_registration_acked import (
    ModelNodeRegistrationAcked,
)
from omnibase_infra.models.registration.events import (
    ModelNodeBecameActive,
    ModelNodeRegistrationAckReceived,
)
from omnibase_infra.nodes.node_registration_orchestrator.handlers.handler_node_registration_acked import (
    DEFAULT_LIVENESS_INTERVAL_SECONDS,
    ENV_LIVENESS_INTERVAL_SECONDS,
    HandlerNodeRegistrationAcked,
    get_liveness_interval_seconds,
)
from omnibase_infra.projectors.projection_reader_registration import (
    ProjectionReaderRegistration,
)

# Fixed test time for deterministic testing
TEST_NOW = datetime(2025, 1, 15, 12, 0, 0, tzinfo=UTC)

# Alias for test readability (uses the constant from the handler module)
TEST_DEFAULT_LIVENESS_INTERVAL = DEFAULT_LIVENESS_INTERVAL_SECONDS


def create_mock_projection_reader() -> AsyncMock:
    """Create a mock ProjectionReaderRegistration."""
    mock = AsyncMock(spec=ProjectionReaderRegistration)
    mock.get_entity_state = AsyncMock(return_value=None)
    return mock


def create_projection(
    entity_id: UUID,
    state: EnumRegistrationState,
    capabilities: ModelNodeCapabilities | None = None,
) -> ModelRegistrationProjection:
    """Create a test projection."""
    return ModelRegistrationProjection(
        entity_id=entity_id,
        domain="registration",
        current_state=state,
        node_type="effect",
        node_version=ModelSemVer.parse("1.0.0"),
        capabilities=capabilities or ModelNodeCapabilities(),
        last_applied_event_id=uuid4(),
        last_applied_offset=0,
        registered_at=TEST_NOW - timedelta(hours=1),
        updated_at=TEST_NOW - timedelta(minutes=5),
    )


def create_ack_command(
    node_id: UUID,
    timestamp: datetime | None = None,
) -> ModelNodeRegistrationAcked:
    """Create a test ack command."""
    return ModelNodeRegistrationAcked(
        node_id=node_id,
        correlation_id=uuid4(),
        timestamp=timestamp or TEST_NOW,
    )


def create_envelope(
    command: ModelNodeRegistrationAcked,
    now: datetime | None = None,
    correlation_id: UUID | None = None,
) -> ModelEventEnvelope[ModelNodeRegistrationAcked]:
    """Create a test event envelope wrapping an ack command."""
    return ModelEventEnvelope(
        envelope_id=uuid4(),
        payload=command,
        envelope_timestamp=now or TEST_NOW,
        correlation_id=correlation_id or uuid4(),
        source="test",
    )


class TestHandlerAckedEmitsActiveEvents:
    """G2 Requirement 7: Handler emits active events for valid acks."""

    @pytest.mark.asyncio
    async def test_handler_acked_emits_active_events(self) -> None:
        """Given projection with state=AWAITING_ACK,
        When handler processes NodeRegistrationAcked,
        Then emits NodeRegistrationAckReceived AND NodeBecameActive.
        """
        # Arrange
        mock_reader = create_mock_projection_reader()

        node_id = uuid4()
        capabilities = ModelNodeCapabilities(postgres=True, read=True, write=True)
        awaiting_projection = create_projection(
            entity_id=node_id,
            state=EnumRegistrationState.AWAITING_ACK,
            capabilities=capabilities,
        )
        mock_reader.get_entity_state.return_value = awaiting_projection

        handler = HandlerNodeRegistrationAcked(mock_reader)
        correlation_id = uuid4()
        ack_command = ModelNodeRegistrationAcked(
            node_id=node_id,
            correlation_id=correlation_id,
            timestamp=TEST_NOW,
        )

        # Act
        envelope = create_envelope(ack_command, TEST_NOW, correlation_id)
        output = await handler.handle(envelope)

        # Assert - handler_id is correct
        assert output.handler_id == "handler-node-registration-acked"

        # Assert - two events emitted
        assert len(output.events) == 2

        # First event: AckReceived
        ack_received = output.events[0]
        assert isinstance(ack_received, ModelNodeRegistrationAckReceived)
        assert ack_received.node_id == node_id
        assert ack_received.entity_id == node_id
        assert ack_received.correlation_id == correlation_id
        assert ack_received.causation_id == ack_command.command_id
        # Verify time injection: emitted_at must equal injected `now`
        assert ack_received.emitted_at == TEST_NOW
        # Liveness deadline = now + 60 seconds
        expected_deadline = TEST_NOW + timedelta(
            seconds=DEFAULT_LIVENESS_INTERVAL_SECONDS
        )
        assert ack_received.liveness_deadline == expected_deadline

        # Second event: BecameActive
        became_active = output.events[1]
        assert isinstance(became_active, ModelNodeBecameActive)
        assert became_active.node_id == node_id
        assert became_active.entity_id == node_id
        assert became_active.correlation_id == correlation_id
        assert became_active.causation_id == ack_command.command_id
        # Verify time injection: emitted_at must equal injected `now`
        assert became_active.emitted_at == TEST_NOW
        assert became_active.capabilities == capabilities

    @pytest.mark.asyncio
    async def test_emits_events_for_accepted_state(self) -> None:
        """Test that ACCEPTED state also allows ack processing."""
        mock_reader = create_mock_projection_reader()

        node_id = uuid4()
        accepted_projection = create_projection(
            entity_id=node_id,
            state=EnumRegistrationState.ACCEPTED,
        )
        mock_reader.get_entity_state.return_value = accepted_projection

        handler = HandlerNodeRegistrationAcked(mock_reader)
        ack_command = create_ack_command(node_id)

        envelope = create_envelope(ack_command, TEST_NOW, uuid4())
        output = await handler.handle(envelope)

        assert output.handler_id == "handler-node-registration-acked"
        assert len(output.events) == 2
        assert isinstance(output.events[0], ModelNodeRegistrationAckReceived)
        assert isinstance(output.events[1], ModelNodeBecameActive)


class TestHandlerAckedIgnoresDuplicate:
    """G2 Requirement 8: Handler ignores duplicate acks."""

    @pytest.mark.asyncio
    async def test_handler_acked_ignores_duplicate(self) -> None:
        """Given projection with state=ACTIVE,
        When handler processes NodeRegistrationAcked,
        Then returns empty events tuple (already active).
        """
        # Arrange
        mock_reader = create_mock_projection_reader()

        node_id = uuid4()
        active_projection = create_projection(
            entity_id=node_id,
            state=EnumRegistrationState.ACTIVE,
        )
        mock_reader.get_entity_state.return_value = active_projection

        handler = HandlerNodeRegistrationAcked(mock_reader)
        ack_command = create_ack_command(node_id)

        # Act
        envelope = create_envelope(ack_command, TEST_NOW, uuid4())
        output = await handler.handle(envelope)

        # Assert - no events (duplicate ack)
        assert output.handler_id == "handler-node-registration-acked"
        assert output.events == ()

    @pytest.mark.asyncio
    async def test_ignores_ack_for_ack_received_state(self) -> None:
        """Test that ACK_RECEIVED state ignores duplicate ack."""
        mock_reader = create_mock_projection_reader()

        node_id = uuid4()
        ack_received_projection = create_projection(
            entity_id=node_id,
            state=EnumRegistrationState.ACK_RECEIVED,
        )
        mock_reader.get_entity_state.return_value = ack_received_projection

        handler = HandlerNodeRegistrationAcked(mock_reader)
        ack_command = create_ack_command(node_id)

        envelope = create_envelope(ack_command, TEST_NOW, uuid4())
        output = await handler.handle(envelope)

        assert output.handler_id == "handler-node-registration-acked"
        assert output.events == ()


class TestHandlerAckedUnknownNode:
    """Test handling of acks for unknown nodes."""

    @pytest.mark.asyncio
    async def test_ignores_ack_for_unknown_node(self) -> None:
        """Test that ack for unknown node returns empty events tuple."""
        mock_reader = create_mock_projection_reader()
        mock_reader.get_entity_state.return_value = None  # Unknown node

        handler = HandlerNodeRegistrationAcked(mock_reader)
        ack_command = create_ack_command(uuid4())

        envelope = create_envelope(ack_command, TEST_NOW, uuid4())
        output = await handler.handle(envelope)

        assert output.handler_id == "handler-node-registration-acked"
        assert output.events == ()


class TestHandlerAckedPendingState:
    """Test handling of acks when in PENDING_REGISTRATION state."""

    @pytest.mark.asyncio
    async def test_ignores_ack_for_pending_state(self) -> None:
        """Test that ack is ignored if node is still pending (not yet accepted)."""
        mock_reader = create_mock_projection_reader()

        node_id = uuid4()
        pending_projection = create_projection(
            entity_id=node_id,
            state=EnumRegistrationState.PENDING_REGISTRATION,
        )
        mock_reader.get_entity_state.return_value = pending_projection

        handler = HandlerNodeRegistrationAcked(mock_reader)
        ack_command = create_ack_command(node_id)

        envelope = create_envelope(ack_command, TEST_NOW, uuid4())
        output = await handler.handle(envelope)

        # Ack too early - not yet accepted
        assert output.handler_id == "handler-node-registration-acked"
        assert output.events == ()


class TestHandlerAckedTerminalStates:
    """Test handling of acks when in terminal states."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "terminal_state",
        [
            EnumRegistrationState.ACK_TIMED_OUT,
            EnumRegistrationState.REJECTED,
            EnumRegistrationState.LIVENESS_EXPIRED,
        ],
    )
    async def test_ignores_ack_for_terminal_states(
        self, terminal_state: EnumRegistrationState
    ) -> None:
        """Test that ack is ignored for nodes in terminal states."""
        mock_reader = create_mock_projection_reader()

        node_id = uuid4()
        terminal_projection = create_projection(
            entity_id=node_id,
            state=terminal_state,
        )
        mock_reader.get_entity_state.return_value = terminal_projection

        handler = HandlerNodeRegistrationAcked(mock_reader)
        ack_command = create_ack_command(node_id)

        envelope = create_envelope(ack_command, TEST_NOW, uuid4())
        output = await handler.handle(envelope)

        # Terminal state - ack is meaningless
        assert output.handler_id == "handler-node-registration-acked"
        assert output.events == (), (
            f"Expected no events for terminal state {terminal_state}"
        )

    @pytest.mark.asyncio
    async def test_ignores_ack_when_already_timed_out(self) -> None:
        """Test that ack is ignored if registration already timed out."""
        mock_reader = create_mock_projection_reader()

        node_id = uuid4()
        timed_out_projection = create_projection(
            entity_id=node_id,
            state=EnumRegistrationState.ACK_TIMED_OUT,
        )
        mock_reader.get_entity_state.return_value = timed_out_projection

        handler = HandlerNodeRegistrationAcked(mock_reader)
        ack_command = create_ack_command(node_id)

        envelope = create_envelope(ack_command, TEST_NOW, uuid4())
        output = await handler.handle(envelope)

        # Too late - already timed out
        assert output.handler_id == "handler-node-registration-acked"
        assert output.events == ()


class TestHandlerAckedLivenessDeadline:
    """Test liveness deadline calculation."""

    @pytest.mark.asyncio
    async def test_liveness_deadline_uses_injected_now(self) -> None:
        """Test that liveness deadline is calculated from injected now."""
        mock_reader = create_mock_projection_reader()

        node_id = uuid4()
        awaiting_projection = create_projection(
            entity_id=node_id,
            state=EnumRegistrationState.AWAITING_ACK,
        )
        mock_reader.get_entity_state.return_value = awaiting_projection

        handler = HandlerNodeRegistrationAcked(mock_reader)
        ack_command = create_ack_command(node_id)

        custom_now = datetime(2025, 6, 15, 10, 30, 0, tzinfo=UTC)

        envelope = create_envelope(ack_command, custom_now, uuid4())
        output = await handler.handle(envelope)

        assert output.handler_id == "handler-node-registration-acked"
        assert len(output.events) == 2
        ack_received = output.events[0]
        assert isinstance(ack_received, ModelNodeRegistrationAckReceived)

        # Liveness deadline should be custom_now + 60 seconds
        expected_deadline = custom_now + timedelta(seconds=60)
        assert ack_received.liveness_deadline == expected_deadline

    @pytest.mark.asyncio
    async def test_custom_liveness_interval(self) -> None:
        """Test that custom liveness interval is respected."""
        mock_reader = create_mock_projection_reader()

        node_id = uuid4()
        awaiting_projection = create_projection(
            entity_id=node_id,
            state=EnumRegistrationState.AWAITING_ACK,
        )
        mock_reader.get_entity_state.return_value = awaiting_projection

        # Create handler with custom liveness interval
        custom_interval = 120  # 2 minutes
        handler = HandlerNodeRegistrationAcked(
            mock_reader, liveness_interval_seconds=custom_interval
        )
        ack_command = create_ack_command(node_id)

        envelope = create_envelope(ack_command, TEST_NOW, uuid4())
        output = await handler.handle(envelope)

        assert output.handler_id == "handler-node-registration-acked"
        assert len(output.events) == 2
        ack_received = output.events[0]

        # Liveness deadline should use custom interval
        expected_deadline = TEST_NOW + timedelta(seconds=custom_interval)
        assert ack_received.liveness_deadline == expected_deadline


class TestHandlerAckedCapabilitiesSnapshot:
    """Test that capabilities are captured in BecameActive event."""

    @pytest.mark.asyncio
    async def test_became_active_includes_capabilities(self) -> None:
        """Test that BecameActive event includes node capabilities."""
        mock_reader = create_mock_projection_reader()

        node_id = uuid4()
        capabilities = ModelNodeCapabilities(
            postgres=True,
            read=True,
            write=True,
            batch_size=100,
        )
        awaiting_projection = create_projection(
            entity_id=node_id,
            state=EnumRegistrationState.AWAITING_ACK,
            capabilities=capabilities,
        )
        mock_reader.get_entity_state.return_value = awaiting_projection

        handler = HandlerNodeRegistrationAcked(mock_reader)
        ack_command = create_ack_command(node_id)

        envelope = create_envelope(ack_command, TEST_NOW, uuid4())
        output = await handler.handle(envelope)

        assert output.handler_id == "handler-node-registration-acked"
        assert len(output.events) == 2
        became_active = output.events[1]
        assert isinstance(became_active, ModelNodeBecameActive)

        # Capabilities should match projection
        assert became_active.capabilities == capabilities
        assert became_active.capabilities.postgres is True
        assert became_active.capabilities.read is True
        assert became_active.capabilities.write is True
        assert became_active.capabilities.batch_size == 100


class TestHandlerAckedEventCausation:
    """Test causation ID linking in emitted events."""

    @pytest.mark.asyncio
    async def test_events_link_to_command_via_causation_id(self) -> None:
        """Test that emitted events link to the ack command via causation_id."""
        mock_reader = create_mock_projection_reader()

        node_id = uuid4()
        awaiting_projection = create_projection(
            entity_id=node_id,
            state=EnumRegistrationState.AWAITING_ACK,
        )
        mock_reader.get_entity_state.return_value = awaiting_projection

        handler = HandlerNodeRegistrationAcked(mock_reader)
        ack_command = create_ack_command(node_id)

        envelope = create_envelope(ack_command, TEST_NOW, uuid4())
        output = await handler.handle(envelope)

        assert output.handler_id == "handler-node-registration-acked"
        assert len(output.events) == 2

        # Both events should link to the command via causation_id
        for event in output.events:
            assert event.causation_id == ack_command.command_id


class TestHandlerAckedProjectionQueries:
    """Test projection reader interactions."""

    @pytest.mark.asyncio
    async def test_queries_projection_with_correct_params(self) -> None:
        """Test that projection is queried with correct parameters."""
        mock_reader = create_mock_projection_reader()
        mock_reader.get_entity_state.return_value = None

        handler = HandlerNodeRegistrationAcked(mock_reader)

        node_id = uuid4()
        correlation_id = uuid4()
        ack_command = ModelNodeRegistrationAcked(
            node_id=node_id,
            correlation_id=correlation_id,
            timestamp=TEST_NOW,
        )

        envelope = create_envelope(ack_command, TEST_NOW, correlation_id)
        output = await handler.handle(envelope)

        assert output.handler_id == "handler-node-registration-acked"
        mock_reader.get_entity_state.assert_called_once_with(
            entity_id=node_id,
            domain="registration",
            correlation_id=correlation_id,
        )


class TestGetLivenessIntervalSeconds:
    """Tests for get_liveness_interval_seconds configuration resolution."""

    def test_returns_default_when_no_config(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test returns default constant when no explicit value or env var."""
        # Ensure env var is not set
        monkeypatch.delenv(ENV_LIVENESS_INTERVAL_SECONDS, raising=False)

        result = get_liveness_interval_seconds()

        assert result == DEFAULT_LIVENESS_INTERVAL_SECONDS
        assert result == 60  # Verify actual default value

    def test_returns_explicit_value_when_provided(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test explicit value takes priority over env var and default."""
        # Set env var to a different value
        monkeypatch.setenv(ENV_LIVENESS_INTERVAL_SECONDS, "90")

        result = get_liveness_interval_seconds(explicit_value=120)

        # Explicit value should win
        assert result == 120

    def test_returns_env_var_when_no_explicit_value(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test env var is used when no explicit value provided."""
        monkeypatch.setenv(ENV_LIVENESS_INTERVAL_SECONDS, "180")

        result = get_liveness_interval_seconds()

        assert result == 180

    def test_explicit_none_uses_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that passing None explicitly uses env var."""
        monkeypatch.setenv(ENV_LIVENESS_INTERVAL_SECONDS, "45")

        result = get_liveness_interval_seconds(explicit_value=None)

        assert result == 45

    def test_raises_protocol_configuration_error_for_invalid_env_var(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test ProtocolConfigurationError raised when env var is not a valid integer."""
        monkeypatch.setenv(ENV_LIVENESS_INTERVAL_SECONDS, "not_a_number")

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            get_liveness_interval_seconds()

        assert ENV_LIVENESS_INTERVAL_SECONDS in str(exc_info.value)
        assert "not_a_number" in str(exc_info.value)

    def test_handler_uses_get_liveness_interval_internally(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that handler respects env var when no explicit value provided."""
        monkeypatch.setenv(ENV_LIVENESS_INTERVAL_SECONDS, "300")
        mock_reader = create_mock_projection_reader()

        # Create handler without explicit liveness_interval_seconds
        handler = HandlerNodeRegistrationAcked(mock_reader)

        # Handler should use the value from env var (resolved via get_liveness_interval_seconds)
        assert handler._liveness_interval_seconds == 300


class TestHandlerAckedTimezoneValidation:
    """Test that handler validates timezone-awareness of envelope timestamp."""

    @pytest.mark.asyncio
    async def test_raises_protocol_configuration_error_for_naive_datetime(self) -> None:
        """Test that handler raises ProtocolConfigurationError if envelope_timestamp is naive (no tzinfo)."""
        mock_reader = create_mock_projection_reader()
        handler = HandlerNodeRegistrationAcked(mock_reader)

        # Create a naive datetime (no timezone info)
        naive_now = datetime(2025, 1, 15, 12, 0, 0)  # No tzinfo!
        assert naive_now.tzinfo is None  # Confirm it's naive

        ack_command = create_ack_command(uuid4())

        # Create envelope with naive datetime
        envelope = create_envelope(ack_command, naive_now, uuid4())

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            await handler.handle(envelope)

        assert "timezone-aware" in str(exc_info.value)
        assert "naive" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_accepts_timezone_aware_datetime(self) -> None:
        """Test that handler accepts timezone-aware datetime."""
        mock_reader = create_mock_projection_reader()
        mock_reader.get_entity_state.return_value = None

        handler = HandlerNodeRegistrationAcked(mock_reader)

        # Use timezone-aware datetime
        aware_now = datetime(2025, 1, 15, 12, 0, 0, tzinfo=UTC)
        assert aware_now.tzinfo is not None  # Confirm it's aware

        ack_command = create_ack_command(uuid4())

        # Should not raise - timezone-aware datetime is valid
        envelope = create_envelope(ack_command, aware_now, uuid4())
        output = await handler.handle(envelope)

        # Unknown node - returns empty events (but should not raise)
        assert output.handler_id == "handler-node-registration-acked"
        assert output.events == ()
