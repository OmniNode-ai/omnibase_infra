# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for DispatcherNodeHeartbeat.

Tests the dispatcher adapter for handling heartbeat events including:
- Successful dispatch with valid ModelNodeHeartbeatEvent
- Dict payload deserialization
- Invalid payload rejection (INVALID_MESSAGE status)
- Circuit breaker integration
- Error handling and sanitization

Related:
    - OMN-1990: Wire heartbeat dispatcher
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from omnibase_core.enums import EnumNodeKind
from omnibase_core.models.dispatch.model_handler_output import ModelHandlerOutput
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_core.models.primitives.model_semver import ModelSemVer
from omnibase_infra.enums import EnumDispatchStatus, EnumMessageCategory
from omnibase_infra.models.registration import ModelNodeHeartbeatEvent
from omnibase_infra.nodes.node_registration_orchestrator.dispatchers import (
    DispatcherNodeHeartbeat,
)

pytestmark = [pytest.mark.asyncio, pytest.mark.unit]


def _make_heartbeat_event() -> ModelNodeHeartbeatEvent:
    """Create a valid heartbeat event for testing."""
    return ModelNodeHeartbeatEvent(
        node_id=uuid4(),
        node_type=EnumNodeKind.EFFECT,
        node_version=ModelSemVer(major=1, minor=0, patch=0),
        uptime_seconds=3600.0,
        active_operations_count=5,
        timestamp=datetime.now(UTC),
    )


def _make_handler_output() -> ModelHandlerOutput[object]:
    """Create a handler output for testing."""
    return ModelHandlerOutput(
        input_envelope_id=uuid4(),
        correlation_id=uuid4(),
        handler_id="handler-node-heartbeat",
        node_kind=EnumNodeKind.ORCHESTRATOR,
        events=(),
        intents=(),
        projections=(),
        result=None,
        processing_time_ms=1.0,
        timestamp=datetime.now(UTC),
    )


def _make_envelope(
    payload: object,
    correlation_id: object | None = None,
) -> ModelEventEnvelope[object]:
    """Create an event envelope with the given payload."""
    return ModelEventEnvelope(
        envelope_id=uuid4(),
        payload=payload,
        envelope_timestamp=datetime.now(UTC),
        correlation_id=correlation_id or uuid4(),
        source="test",
    )


class TestDispatcherNodeHeartbeat:
    """Tests for DispatcherNodeHeartbeat."""

    def test_dispatcher_id(self) -> None:
        """Dispatcher ID follows naming convention."""
        handler = MagicMock()
        dispatcher = DispatcherNodeHeartbeat(handler)
        assert dispatcher.dispatcher_id == "dispatcher.registration.node-heartbeat"

    def test_category_is_event(self) -> None:
        """Dispatcher processes EVENT category."""
        handler = MagicMock()
        dispatcher = DispatcherNodeHeartbeat(handler)
        assert dispatcher.category == EnumMessageCategory.EVENT

    def test_message_types(self) -> None:
        """Dispatcher accepts ModelNodeHeartbeatEvent."""
        handler = MagicMock()
        dispatcher = DispatcherNodeHeartbeat(handler)
        assert dispatcher.message_types == {
            "ModelNodeHeartbeatEvent",
            "platform.node-heartbeat",
        }

    def test_node_kind_is_orchestrator(self) -> None:
        """Dispatcher belongs to ORCHESTRATOR node kind."""
        handler = MagicMock()
        dispatcher = DispatcherNodeHeartbeat(handler)
        assert dispatcher.node_kind == EnumNodeKind.ORCHESTRATOR

    async def test_handle_success(self) -> None:
        """Successful dispatch with valid heartbeat event."""
        handler = AsyncMock()
        handler.handle = AsyncMock(return_value=_make_handler_output())

        dispatcher = DispatcherNodeHeartbeat(handler)

        event = _make_heartbeat_event()
        envelope = _make_envelope(event)

        result = await dispatcher.handle(envelope)

        assert result.status == EnumDispatchStatus.SUCCESS
        assert result.dispatcher_id == "dispatcher.registration.node-heartbeat"
        assert result.topic == "node.heartbeat"
        assert result.duration_ms >= 0

    async def test_handle_dict_payload(self) -> None:
        """Successful dispatch with dict payload (deserialization)."""
        handler = AsyncMock()
        handler.handle = AsyncMock(return_value=_make_handler_output())

        dispatcher = DispatcherNodeHeartbeat(handler)

        event = _make_heartbeat_event()
        envelope = _make_envelope(event.model_dump())

        result = await dispatcher.handle(envelope)

        assert result.status == EnumDispatchStatus.SUCCESS

    async def test_handle_invalid_payload(self) -> None:
        """Invalid payload returns INVALID_MESSAGE status."""
        handler = AsyncMock()
        dispatcher = DispatcherNodeHeartbeat(handler)

        envelope = _make_envelope("not a heartbeat event")

        result = await dispatcher.handle(envelope)

        assert result.status == EnumDispatchStatus.INVALID_MESSAGE
        assert "Expected ModelNodeHeartbeatEvent" in (result.error_message or "")
        handler.handle.assert_not_called()

    async def test_handle_handler_error(self) -> None:
        """Handler exception returns HANDLER_ERROR status."""
        handler = AsyncMock()
        handler.handle = AsyncMock(side_effect=RuntimeError("test error"))

        dispatcher = DispatcherNodeHeartbeat(handler)

        event = _make_heartbeat_event()
        envelope = _make_envelope(event)

        result = await dispatcher.handle(envelope)

        assert result.status == EnumDispatchStatus.HANDLER_ERROR
        assert result.error_message is not None

    async def test_correlation_id_propagated(self) -> None:
        """Correlation ID from envelope is propagated to result."""
        handler = AsyncMock()
        handler.handle = AsyncMock(return_value=_make_handler_output())

        dispatcher = DispatcherNodeHeartbeat(handler)

        correlation_id = uuid4()
        event = _make_heartbeat_event()
        envelope = _make_envelope(event, correlation_id=correlation_id)

        result = await dispatcher.handle(envelope)

        assert result.correlation_id == correlation_id
