# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for DispatcherNodeIntrospected (OMN-3444, OMN-5132).

OMN-5132: The reducer now transitions nodes directly to ACTIVE (no
AWAITING_ACK, no ModelNodeRegistrationAccepted). The dispatcher's
auto-ACK code path is never triggered because the reducer no longer
emits ModelNodeRegistrationAccepted. These tests verify that:
- The dispatcher processes introspection events successfully.
- No auto-ACK publish occurs (no ModelNodeRegistrationAccepted to trigger it).
- Output events include ModelNodeBecameActive (direct-to-active).
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from omnibase_core.enums import EnumNodeKind
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.models.registration import ModelNodeIntrospectionEvent
from omnibase_infra.models.registration.events.model_node_became_active import (
    ModelNodeBecameActive,
)
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


def _make_handler() -> HandlerNodeIntrospected:
    """Create a HandlerNodeIntrospected with a mock projection reader."""
    mock_reader = AsyncMock(spec=ProjectionReaderRegistration)
    mock_reader.get_entity_state = AsyncMock(return_value=None)  # new node -> ACTIVE
    reducer = RegistrationReducerService()
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


class TestDispatcherNodeIntrospectedDirectToActive:
    """Tests for direct-to-active registration in the dispatcher (OMN-5132)."""

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_dispatcher_emits_became_active(self) -> None:
        """Dispatcher output_events should include ModelNodeBecameActive."""
        from omnibase_core.protocols.event_bus.protocol_event_bus import (
            ProtocolEventBus,
        )

        mock_event_bus = AsyncMock(spec=ProtocolEventBus)
        dispatcher = DispatcherNodeIntrospected(
            _make_handler(), event_bus=mock_event_bus
        )

        envelope = _make_envelope()
        result = await dispatcher.handle(envelope)

        assert result.status.value == "success"
        became_active_events = [
            e for e in result.output_events if isinstance(e, ModelNodeBecameActive)
        ]
        assert len(became_active_events) == 1

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_dispatcher_no_auto_ack_publish(self) -> None:
        """No auto-ACK publish occurs because reducer no longer emits Accepted."""
        from omnibase_core.protocols.event_bus.protocol_event_bus import (
            ProtocolEventBus,
        )

        mock_event_bus = AsyncMock(spec=ProtocolEventBus)
        dispatcher = DispatcherNodeIntrospected(
            _make_handler(), event_bus=mock_event_bus
        )

        envelope = _make_envelope()
        await dispatcher.handle(envelope)

        # No publish_envelope call — reducer emits BecameActive, not Accepted
        mock_event_bus.publish_envelope.assert_not_awaited()
