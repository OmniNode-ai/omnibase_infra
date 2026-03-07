# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for DispatcherNodeIntrospected auto-ACK (OMN-3444).

Tests validate:
- When ONEX_REGISTRATION_AUTO_ACK=true and the reducer emits ModelNodeRegistrationAccepted,
  the dispatcher direct-publishes ModelNodeRegistrationAcked to the ack topic.
- When ONEX_REGISTRATION_AUTO_ACK is not set, no publish occurs.
- The auto-ACK envelope is NOT included in output_events.

Architecture note:
    Handlers cannot have event_bus access (architecture invariant).
    Auto-ACK is the dispatcher's responsibility (OMN-3444, Path B).
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from omnibase_core.enums import EnumNodeKind
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.models.registration import ModelNodeIntrospectionEvent
from omnibase_infra.nodes.node_registration_orchestrator.dispatchers.dispatcher_node_introspected import (
    DispatcherNodeIntrospected,
)
from omnibase_infra.nodes.node_registration_orchestrator.handlers.handler_node_introspected import (
    HandlerNodeIntrospected,
)
from omnibase_infra.nodes.node_registration_orchestrator.services import (
    RegistrationReducerService,
)
from omnibase_infra.projectors.projection_reader_registration import (
    ProjectionReaderRegistration,
)

TEST_NOW = datetime(2025, 1, 15, 12, 0, 0, tzinfo=UTC)

_ACK_TOPIC = "onex.cmd.platform.node-registration-acked.v1"


def _make_handler() -> HandlerNodeIntrospected:
    """Create a HandlerNodeIntrospected with a mock projection reader."""
    mock_reader = AsyncMock(spec=ProjectionReaderRegistration)
    mock_reader.get_entity_state = AsyncMock(
        return_value=None
    )  # new node -> AWAITING_ACK
    reducer = RegistrationReducerService(ack_timeout_seconds=30.0)
    return HandlerNodeIntrospected(mock_reader, reducer=reducer)


def _make_envelope(node_id: None = None) -> ModelEventEnvelope[object]:
    """Create an introspection event envelope."""

    nid = node_id or uuid4()
    event = ModelNodeIntrospectionEvent(
        node_id=nid,
        node_type=EnumNodeKind.EFFECT,
        correlation_id=uuid4(),
        timestamp=TEST_NOW,
    )
    return ModelEventEnvelope(
        envelope_id=uuid4(),
        payload=event,
        envelope_timestamp=TEST_NOW,
        correlation_id=uuid4(),
        source="test",
    )


class TestDispatcherNodeIntrospectedAutoAck:
    """Tests for ONEX_REGISTRATION_AUTO_ACK feature in the dispatcher (OMN-3444)."""

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_dispatcher_publishes_auto_ack_when_flag_enabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Gate: dispatcher direct-publishes ModelNodeRegistrationAcked when auto-ACK is on.

        When ONEX_REGISTRATION_AUTO_ACK=true and the reducer emits
        ModelNodeRegistrationAccepted (new-node path), the dispatcher must
        direct-publish ModelNodeRegistrationAcked to _ACK_TOPIC.
        The ack must NOT appear in the ModelDispatchResult.output_events.
        """
        from omnibase_core.protocols.event_bus.protocol_event_bus import (
            ProtocolEventBus,
        )
        from omnibase_infra.models.registration.commands.model_node_registration_acked import (
            ModelNodeRegistrationAcked,
        )

        monkeypatch.setenv("ONEX_REGISTRATION_AUTO_ACK", "true")

        mock_event_bus = AsyncMock(spec=ProtocolEventBus)
        dispatcher = DispatcherNodeIntrospected(
            _make_handler(), event_bus=mock_event_bus
        )

        envelope = _make_envelope()
        result = await dispatcher.handle(envelope)

        # publish_envelope must be called exactly once
        mock_event_bus.publish_envelope.assert_awaited_once()

        call_args = mock_event_bus.publish_envelope.call_args
        # Envelope is first positional arg; topic is keyword arg
        published_envelope = call_args.args[0]
        assert isinstance(published_envelope, ModelEventEnvelope)
        assert isinstance(published_envelope.payload, ModelNodeRegistrationAcked)

        # Topic must be the ack topic
        assert call_args.kwargs.get("topic") == _ACK_TOPIC

        # Auto-ACK must NOT appear in output_events (would be routed to wrong topic)
        ack_in_output = [
            e for e in result.output_events if isinstance(e, ModelNodeRegistrationAcked)
        ]
        assert len(ack_in_output) == 0

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_dispatcher_no_auto_ack_when_flag_disabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When ONEX_REGISTRATION_AUTO_ACK is not set, publish_envelope must not be called."""
        from omnibase_core.protocols.event_bus.protocol_event_bus import (
            ProtocolEventBus,
        )

        monkeypatch.delenv("ONEX_REGISTRATION_AUTO_ACK", raising=False)

        mock_event_bus = AsyncMock(spec=ProtocolEventBus)
        dispatcher = DispatcherNodeIntrospected(
            _make_handler(), event_bus=mock_event_bus
        )

        envelope = _make_envelope()
        await dispatcher.handle(envelope)

        mock_event_bus.publish_envelope.assert_not_awaited()
