# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for MixinNodeIntrospection._on_registration_accepted callback.

Validates the node-side ACK emission logic:
    - Receives a registration-accepted event from the event bus
    - Filters by entity_id/node_id (only responds to own node)
    - Constructs ModelNodeRegistrationAcked command
    - Publishes to the ACK command topic

These tests exercise _on_registration_accepted directly with mock event
bus and mock messages, without real Kafka infrastructure.

Related Tickets:
    - OMN-888 (C1): Registration Orchestrator
    - OMN-889 (D1): Registration Reducer
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

from omnibase_core.enums.enum_node_kind import EnumNodeKind
from omnibase_infra.mixins.mixin_node_introspection import MixinNodeIntrospection
from omnibase_infra.topics import SUFFIX_NODE_REGISTRATION_ACKED

# Deterministic test UUIDs
TEST_NODE_ID = UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
TEST_OTHER_NODE_ID = UUID("11111111-2222-3333-4444-555555555555")
TEST_CORRELATION_ID = UUID("cccccccc-dddd-eeee-ffff-000000000000")

# The topic that _on_registration_accepted publishes ACKs to
ACK_COMMAND_TOPIC = SUFFIX_NODE_REGISTRATION_ACKED


# ---------------------------------------------------------------------------
# Testable node subclass (minimal mixin host)
# ---------------------------------------------------------------------------


class StubAckNode(MixinNodeIntrospection):
    """Minimal node class for testing _on_registration_accepted.

    Manually initializes only the mixin attributes needed by the
    _on_registration_accepted callback, without calling the full
    initialize_introspection() method.
    """

    def __init__(
        self,
        node_id: UUID,
        event_bus: AsyncMock | None = None,
    ) -> None:
        # Do NOT call super().__init__() -- we manually wire the mixin attributes
        self._introspection_node_id: UUID | None = node_id
        self._introspection_event_bus = event_bus
        self._introspection_initialized = True
        self._introspection_env = "test"
        self._introspection_service = "test-service"
        self._introspection_node_name = "test-node"
        self._introspection_version = "1.0.0"
        self._time_provider = lambda: datetime.now(UTC)


# ---------------------------------------------------------------------------
# Mock message factory
# ---------------------------------------------------------------------------


def make_mock_message(
    entity_id: UUID | None = None,
    node_id: UUID | None = None,
    correlation_id: UUID | None = None,
    *,
    include_value: bool = True,
) -> MagicMock:
    """Build a mock ProtocolEventMessage with JSON-encoded value.

    The message value simulates what Kafka delivers: a JSON-encoded
    dict with entity_id/node_id and correlation_id fields.

    Args:
        entity_id: The entity_id to include in the message payload.
        node_id: Alternative to entity_id (fallback field).
        correlation_id: Correlation ID for tracing.
        include_value: If False, message.value returns None.

    Returns:
        MagicMock implementing ProtocolEventMessage interface.
    """
    msg = MagicMock()

    if not include_value:
        msg.value = None
        return msg

    payload: dict[str, str | None] = {}
    if entity_id is not None:
        payload["entity_id"] = str(entity_id)
    if node_id is not None:
        payload["node_id"] = str(node_id)
    if correlation_id is not None:
        payload["correlation_id"] = str(correlation_id)

    msg.value = json.dumps(payload).encode("utf-8")
    return msg


# ---------------------------------------------------------------------------
# Tests for _on_registration_accepted
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestOnRegistrationAcceptedEmitsAck:
    """Tests that _on_registration_accepted emits ACK for matching node."""

    @pytest.mark.asyncio
    async def test_emits_ack_for_matching_entity_id(self) -> None:
        """Matching entity_id -> publishes ModelNodeRegistrationAcked command."""
        event_bus = AsyncMock()
        event_bus.publish_envelope = AsyncMock()
        node = StubAckNode(node_id=TEST_NODE_ID, event_bus=event_bus)

        message = make_mock_message(
            entity_id=TEST_NODE_ID,
            correlation_id=TEST_CORRELATION_ID,
        )

        await node._on_registration_accepted(message)

        # Should have called publish_envelope exactly once
        event_bus.publish_envelope.assert_awaited_once()

        call_kwargs = event_bus.publish_envelope.call_args
        # publish_envelope is called with keyword args: envelope=..., topic=...
        envelope = call_kwargs.kwargs.get("envelope") or call_kwargs.args[0]
        topic = call_kwargs.kwargs.get("topic") or call_kwargs.args[1]

        assert topic == ACK_COMMAND_TOPIC

        # Verify the payload is the ACK command
        from omnibase_infra.models.registration.commands.model_node_registration_acked import (
            ModelNodeRegistrationAcked,
        )

        payload = envelope.payload
        assert isinstance(payload, ModelNodeRegistrationAcked)
        assert payload.node_id == TEST_NODE_ID
        assert payload.correlation_id == TEST_CORRELATION_ID

    @pytest.mark.asyncio
    async def test_emits_ack_for_matching_node_id_field(self) -> None:
        """Matching node_id (fallback field) -> publishes ACK command."""
        event_bus = AsyncMock()
        event_bus.publish_envelope = AsyncMock()
        node = StubAckNode(node_id=TEST_NODE_ID, event_bus=event_bus)

        # Use node_id field instead of entity_id
        message = make_mock_message(
            node_id=TEST_NODE_ID,
            correlation_id=TEST_CORRELATION_ID,
        )

        await node._on_registration_accepted(message)

        event_bus.publish_envelope.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_envelope_contains_correct_correlation_id(self) -> None:
        """Published envelope carries the same correlation_id from the accepted event."""
        event_bus = AsyncMock()
        event_bus.publish_envelope = AsyncMock()
        node = StubAckNode(node_id=TEST_NODE_ID, event_bus=event_bus)

        message = make_mock_message(
            entity_id=TEST_NODE_ID,
            correlation_id=TEST_CORRELATION_ID,
        )

        await node._on_registration_accepted(message)

        call_kwargs = event_bus.publish_envelope.call_args
        envelope = call_kwargs.kwargs.get("envelope") or call_kwargs.args[0]

        assert envelope.correlation_id == TEST_CORRELATION_ID

    @pytest.mark.asyncio
    async def test_generates_correlation_id_when_missing(self) -> None:
        """Missing correlation_id in event -> generates a new UUID for ACK."""
        event_bus = AsyncMock()
        event_bus.publish_envelope = AsyncMock()
        node = StubAckNode(node_id=TEST_NODE_ID, event_bus=event_bus)

        # No correlation_id in the message
        message = make_mock_message(entity_id=TEST_NODE_ID)

        await node._on_registration_accepted(message)

        event_bus.publish_envelope.assert_awaited_once()
        call_kwargs = event_bus.publish_envelope.call_args
        envelope = call_kwargs.kwargs.get("envelope") or call_kwargs.args[0]

        # A correlation_id should still be present (auto-generated)
        assert envelope.correlation_id is not None
        assert isinstance(envelope.correlation_id, UUID)


@pytest.mark.unit
class TestOnRegistrationAcceptedFiltering:
    """Tests that _on_registration_accepted filters out non-matching nodes."""

    @pytest.mark.asyncio
    async def test_ignores_different_entity_id(self) -> None:
        """Different entity_id -> does NOT publish ACK."""
        event_bus = AsyncMock()
        event_bus.publish_envelope = AsyncMock()
        node = StubAckNode(node_id=TEST_NODE_ID, event_bus=event_bus)

        message = make_mock_message(
            entity_id=TEST_OTHER_NODE_ID,
            correlation_id=TEST_CORRELATION_ID,
        )

        await node._on_registration_accepted(message)

        event_bus.publish_envelope.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_ignores_different_node_id(self) -> None:
        """Different node_id -> does NOT publish ACK."""
        event_bus = AsyncMock()
        event_bus.publish_envelope = AsyncMock()
        node = StubAckNode(node_id=TEST_NODE_ID, event_bus=event_bus)

        message = make_mock_message(
            node_id=TEST_OTHER_NODE_ID,
            correlation_id=TEST_CORRELATION_ID,
        )

        await node._on_registration_accepted(message)

        event_bus.publish_envelope.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_ignores_message_without_entity_or_node_id(self) -> None:
        """Message with no entity_id or node_id -> does NOT publish ACK."""
        event_bus = AsyncMock()
        event_bus.publish_envelope = AsyncMock()
        node = StubAckNode(node_id=TEST_NODE_ID, event_bus=event_bus)

        # Empty payload: no entity_id, no node_id
        msg = MagicMock()
        msg.value = json.dumps({}).encode("utf-8")

        await node._on_registration_accepted(msg)

        event_bus.publish_envelope.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_ignores_message_with_no_value(self) -> None:
        """Message with value=None -> does NOT publish ACK (skips gracefully)."""
        event_bus = AsyncMock()
        event_bus.publish_envelope = AsyncMock()
        node = StubAckNode(node_id=TEST_NODE_ID, event_bus=event_bus)

        message = make_mock_message(include_value=False)

        await node._on_registration_accepted(message)

        event_bus.publish_envelope.assert_not_awaited()


@pytest.mark.unit
class TestOnRegistrationAcceptedEdgeCases:
    """Edge cases and error handling for _on_registration_accepted."""

    @pytest.mark.asyncio
    async def test_no_event_bus_does_not_raise(self) -> None:
        """event_bus=None -> logs warning but does not raise."""
        node = StubAckNode(node_id=TEST_NODE_ID, event_bus=None)

        message = make_mock_message(
            entity_id=TEST_NODE_ID,
            correlation_id=TEST_CORRELATION_ID,
        )

        # Should not raise -- graceful degradation
        await node._on_registration_accepted(message)

    @pytest.mark.asyncio
    async def test_invalid_json_does_not_raise(self) -> None:
        """Malformed JSON value -> logs warning but does not raise."""
        event_bus = AsyncMock()
        event_bus.publish_envelope = AsyncMock()
        node = StubAckNode(node_id=TEST_NODE_ID, event_bus=event_bus)

        msg = MagicMock()
        msg.value = b"not-valid-json"

        # Should not raise -- wrapped in try/except
        await node._on_registration_accepted(msg)

        event_bus.publish_envelope.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_publish_envelope_failure_does_not_raise(self) -> None:
        """publish_envelope raises -> exception is caught, does not propagate."""
        event_bus = AsyncMock()
        event_bus.publish_envelope = AsyncMock(side_effect=RuntimeError("Kafka down"))
        node = StubAckNode(node_id=TEST_NODE_ID, event_bus=event_bus)

        message = make_mock_message(
            entity_id=TEST_NODE_ID,
            correlation_id=TEST_CORRELATION_ID,
        )

        # Should not raise -- wrapped in outer try/except
        await node._on_registration_accepted(message)

    @pytest.mark.asyncio
    async def test_invalid_correlation_id_generates_new_uuid(self) -> None:
        """Invalid correlation_id string -> generates a fresh UUID instead."""
        event_bus = AsyncMock()
        event_bus.publish_envelope = AsyncMock()
        node = StubAckNode(node_id=TEST_NODE_ID, event_bus=event_bus)

        msg = MagicMock()
        payload = {
            "entity_id": str(TEST_NODE_ID),
            "correlation_id": "not-a-valid-uuid",
        }
        msg.value = json.dumps(payload).encode("utf-8")

        await node._on_registration_accepted(msg)

        event_bus.publish_envelope.assert_awaited_once()
        call_kwargs = event_bus.publish_envelope.call_args
        envelope = call_kwargs.kwargs.get("envelope") or call_kwargs.args[0]

        # Should have a valid UUID (auto-generated)
        assert isinstance(envelope.correlation_id, UUID)
        # Should NOT be the invalid string
        assert str(envelope.correlation_id) != "not-a-valid-uuid"

    @pytest.mark.asyncio
    async def test_fallback_to_publish_when_no_publish_envelope(self) -> None:
        """Event bus without publish_envelope -> falls back to publish()."""
        event_bus = AsyncMock()
        # Remove publish_envelope so hasattr check fails
        del event_bus.publish_envelope
        event_bus.publish = AsyncMock()
        node = StubAckNode(node_id=TEST_NODE_ID, event_bus=event_bus)

        message = make_mock_message(
            entity_id=TEST_NODE_ID,
            correlation_id=TEST_CORRELATION_ID,
        )

        await node._on_registration_accepted(message)

        event_bus.publish.assert_awaited_once()
        call_kwargs = event_bus.publish.call_args
        topic = call_kwargs.kwargs.get("topic")
        assert topic == ACK_COMMAND_TOPIC

        # Verify the value is valid JSON containing the ACK command
        value = call_kwargs.kwargs.get("value")
        assert value is not None
        parsed = json.loads(value.decode("utf-8"))
        assert parsed["node_id"] == str(TEST_NODE_ID)
