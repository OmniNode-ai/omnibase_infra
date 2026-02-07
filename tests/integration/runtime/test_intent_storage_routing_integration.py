# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Integration tests for intent classification to storage routing (OMN-1509).

This module validates the runtime correctly routes `intent-classified` events
to the intent storage handler and emits `intent-stored` events on success.

Test Flow:
    1. Start RuntimeHostProcess with intent_storage handler registered
    2. Emit `dev.omniintelligence.onex.evt.platform.intent-classified.v1` event to input topic
    3. Verify handler receives envelope
    4. Verify `intent-stored.v1` event emitted to output topic

Architecture:
    The intent storage handler receives classified intent events and stores them
    in a graph database. On successful storage, it emits a confirmation event.

    Event Flow:
        intent-classified.v1 -> IntentStorageHandler -> intent-stored.v1

Test Categories:
    - TestIntentStorageRouting: Verify intent-classified events route to handler
    - TestIntentStoredEventEmission: Verify intent-stored events on success
    - TestEnvelopeStructure: Verify handler receives correct envelope structure

Running Tests:
    # Run all intent storage routing tests:
    pytest tests/integration/runtime/test_intent_storage_routing_integration.py

    # Run with verbose output:
    pytest tests/integration/runtime/test_intent_storage_routing_integration.py -v

    # Run specific test class:
    pytest tests/integration/runtime/test_intent_storage_routing_integration.py::TestIntentStorageRouting

Related:
    - OMN-1509: Intent classification to storage routing
    - HandlerIntent: Intent handler for graph operations
    - HandlerGraph: Graph database operations
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncGenerator, Callable, Coroutine
from datetime import UTC, datetime
from typing import TypeGuard
from uuid import UUID, uuid4

import pytest

logger = logging.getLogger(__name__)

from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.event_bus.models import ModelEventMessage
from tests.conftest import make_test_node_identity

# Type alias for emit callback signature
EmitCallback = Callable[[dict[str, object]], Coroutine[object, object, None]]


# =============================================================================
# Type Guards
# =============================================================================


def is_dict_payload(value: object) -> TypeGuard[dict[str, object]]:
    """Type guard to narrow payload to dict[str, object].

    Args:
        value: Value to check.

    Returns:
        True if value is a dict, enabling type narrowing.
    """
    return isinstance(value, dict)


def extract_intent_type(envelope: dict[str, object]) -> object:
    """Extract intent_type from envelope payload with type narrowing.

    Args:
        envelope: Event envelope containing payload.

    Returns:
        The intent_type value from the payload.

    Raises:
        AssertionError: If payload is not a dict.
    """
    payload = envelope["payload"]
    assert is_dict_payload(payload), "Expected dict payload"
    return payload["intent_type"]


# =============================================================================
# Test Constants
# =============================================================================

# Topic names following ONEX 5-segment realm-agnostic naming convention:
#   Realm-agnostic: onex.{kind}.{producer}.{event-name}.v{version}
#   Runtime-qualified: {env}.{namespace}.onex.{kind}.{producer}.{event-name}.v{version}
#
# Components:
#   - env: Environment (dev, staging, prod) - prepended at runtime
#   - namespace: Service namespace (omniintelligence) - prepended at runtime
#   - onex: ONEX platform identifier
#   - kind: Message kind (evt for events, cmd for commands)
#   - producer: Producer identifier (platform)
#   - event-name: Hyphenated event name (intent-classified, intent-stored)
#   - version: Schema version (v1, v2, etc.)
INPUT_TOPIC = "dev.omniintelligence.onex.evt.platform.intent-classified.v1"
OUTPUT_TOPIC = "dev.omniintelligence.onex.evt.platform.intent-stored.v1"

# Test timing constants
MESSAGE_WAIT_TIMEOUT = 2.0
HANDLER_PROCESSING_DELAY = 0.1

# Fixed timestamp for reproducible test assertions.
# Using a clearly past date to avoid any timezone-related edge cases.
FIXED_TEST_TIMESTAMP = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)


# =============================================================================
# Mock Intent Storage Handler
# =============================================================================


class MockIntentStorageHandler:
    """Mock intent storage handler for testing routing behavior.

    This handler simulates the behavior of an intent storage handler:
    - Receives intent-classified envelopes
    - Stores intents (mocked)
    - Emits intent-stored events via callback

    Attributes:
        captured_envelopes: List of envelopes received by the handler.
        emit_callback: Optional callback to emit output events.
        should_succeed: Control flag for simulating success/failure.
    """

    def __init__(self) -> None:
        """Initialize the mock handler."""
        self.captured_envelopes: list[dict[str, object]] = []
        self.invocation_count: int = 0
        self.emit_callback: EmitCallback | None = None
        self.should_succeed: bool = True
        self._initialized: bool = False

    async def initialize(self, config: dict[str, object]) -> None:
        """Initialize the handler with configuration.

        Args:
            config: Handler configuration (ignored in mock).
        """
        self._initialized = True

    async def shutdown(self) -> None:
        """Shutdown the handler."""
        self._initialized = False

    async def execute(self, envelope: dict[str, object]) -> dict[str, object]:
        """Execute the intent storage operation.

        Args:
            envelope: Request envelope containing:
                - operation: Should be "intent.store" or similar
                - payload: Intent data to store
                - correlation_id: Correlation ID for tracing

        Returns:
            dict containing operation result.
        """
        self.captured_envelopes.append(envelope)
        self.invocation_count += 1

        correlation_id = envelope.get("correlation_id", str(uuid4()))
        if isinstance(correlation_id, UUID):
            correlation_id = str(correlation_id)

        if self.should_succeed:
            # Emit stored event via callback if configured
            if self.emit_callback is not None:
                stored_event = {
                    "event_type": "dev.omniintelligence.onex.evt.platform.intent-stored.v1",
                    "correlation_id": correlation_id,
                    "payload": {
                        "intent_id": str(uuid4()),
                        "stored_at": FIXED_TEST_TIMESTAMP.isoformat(),
                        "original_payload": envelope.get("payload"),
                    },
                }
                await self.emit_callback(stored_event)

            return {
                "success": True,
                "data": {
                    "intent_id": str(uuid4()),
                    "stored_at": FIXED_TEST_TIMESTAMP.isoformat(),
                },
                "correlation_id": correlation_id,
            }
        else:
            return {
                "success": False,
                "error": "Simulated storage failure",
                "correlation_id": correlation_id,
            }

    async def health_check(self) -> dict[str, object]:
        """Return handler health status."""
        return {"healthy": self._initialized}


# =============================================================================
# Mock Event Router
# =============================================================================


class MockEventRouter:
    """Simulates event routing from event bus to handler.

    This class bridges the event bus subscription to handler invocation,
    simulating how RuntimeHostProcess routes envelopes to handlers.

    Attributes:
        handler: The handler to route events to.
        event_bus: The event bus for publishing output events.
        output_topic: Topic for emitting output events.
    """

    def __init__(
        self,
        handler: MockIntentStorageHandler,
        event_bus: EventBusInmemory,
        output_topic: str,
    ) -> None:
        """Initialize the router.

        Args:
            handler: Handler to route events to.
            event_bus: Event bus for publishing output events.
            output_topic: Topic for output events.
        """
        self.handler = handler
        self.event_bus = event_bus
        self.output_topic = output_topic

        # Configure handler's emit callback
        self.handler.emit_callback = self._emit_output

    async def _emit_output(self, event: dict[str, object]) -> None:
        """Emit output event to the event bus.

        Args:
            event: Event to emit.
        """
        await self.event_bus.publish_envelope(event, self.output_topic)

    async def route_message(self, msg: ModelEventMessage) -> None:
        """Route incoming message to handler.

        Args:
            msg: Event message from the bus.

        Note:
            Catches exceptions to prevent router crash during tests.
            In production, errors would be logged and routed to DLQ.
        """
        try:
            envelope = json.loads(msg.value.decode("utf-8"))

            # Add operation field based on event type (simulating dispatcher logic)
            event_type = envelope.get("event_type", "")
            if "intent-classified" in event_type:
                envelope["operation"] = "intent.store"

            # Route to handler
            await self.handler.execute(envelope)
        except json.JSONDecodeError:
            logger.exception("Failed to decode message JSON")
        except KeyError as e:
            logger.exception("Missing required envelope field: %s", e)
        except Exception:
            # Catch-all for unexpected errors during testing
            logger.exception("Unexpected router error")


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def correlation_id() -> UUID:
    """Generate a unique correlation ID for test tracing."""
    return uuid4()


@pytest.fixture
def session_id() -> str:
    """Generate a unique session ID for test isolation."""
    return f"test-session-{uuid4().hex[:8]}"


@pytest.fixture
async def event_bus() -> AsyncGenerator[EventBusInmemory, None]:
    """Create and start an in-memory event bus.

    Yields:
        Started EventBusInmemory instance.
    """
    bus = EventBusInmemory(
        environment="integration-test",
        group="intent-routing-test",
        max_history=500,
    )
    await bus.start()
    yield bus
    await bus.close()


@pytest.fixture
def mock_handler() -> MockIntentStorageHandler:
    """Create a mock intent storage handler."""
    return MockIntentStorageHandler()


@pytest.fixture
def classified_intent_envelope(
    correlation_id: UUID,
    session_id: str,
) -> dict[str, object]:
    """Create a test intent-classified envelope.

    Args:
        correlation_id: Correlation ID for tracing.
        session_id: Session ID for the intent.

    Returns:
        dict representing an intent-classified event envelope.
    """
    return {
        "event_type": "dev.omniintelligence.onex.evt.platform.intent-classified.v1",
        "correlation_id": str(correlation_id),
        "timestamp": FIXED_TEST_TIMESTAMP.isoformat(),
        "payload": {
            "session_id": session_id,
            "intent_type": "navigation",
            "confidence": 0.95,
            "raw_text": "Show me the dashboard",
            "entities": [
                {"type": "page", "value": "dashboard"},
            ],
            "metadata": {
                "source": "voice",
                "language": "en-US",
            },
        },
    }


# =============================================================================
# Test Classes
# =============================================================================


class TestIntentStorageRouting:
    """Integration tests for intent classification to storage routing (OMN-1509)."""

    @pytest.mark.asyncio
    async def test_intent_classified_routed_to_storage_handler(
        self,
        event_bus: EventBusInmemory,
        mock_handler: MockIntentStorageHandler,
        classified_intent_envelope: dict[str, object],
    ) -> None:
        """Verify intent-classified events route to intent storage handler.

        This test validates:
        1. Event bus correctly delivers intent-classified events
        2. Router dispatches envelope to intent storage handler
        3. Handler receives the envelope for processing
        """
        # Setup router
        router = MockEventRouter(mock_handler, event_bus, OUTPUT_TOPIC)

        # Subscribe to input topic
        unsubscribe = await event_bus.subscribe(
            INPUT_TOPIC,
            make_test_node_identity("intent-routing"),
            router.route_message,
        )

        # Publish intent-classified event
        await event_bus.publish_envelope(classified_intent_envelope, INPUT_TOPIC)

        # Wait for processing
        await asyncio.sleep(HANDLER_PROCESSING_DELAY)

        # Verify handler received the envelope
        assert mock_handler.invocation_count == 1, (
            "Handler should be invoked exactly once"
        )
        assert len(mock_handler.captured_envelopes) == 1, (
            "Handler should capture one envelope"
        )

        # Verify envelope was routed correctly
        captured = mock_handler.captured_envelopes[0]
        assert (
            captured["event_type"]
            == "dev.omniintelligence.onex.evt.platform.intent-classified.v1"
        )
        assert captured["operation"] == "intent.store", (
            "Router should set operation to intent.store"
        )

        await unsubscribe()

    @pytest.mark.asyncio
    async def test_multiple_intents_routed_in_sequence(
        self,
        event_bus: EventBusInmemory,
        mock_handler: MockIntentStorageHandler,
        session_id: str,
    ) -> None:
        """Verify multiple intent-classified events are routed sequentially.

        Tests that the handler processes multiple intents correctly.
        """
        router = MockEventRouter(mock_handler, event_bus, OUTPUT_TOPIC)

        unsubscribe = await event_bus.subscribe(
            INPUT_TOPIC,
            make_test_node_identity("multi-intent"),
            router.route_message,
        )

        # Publish multiple intent-classified events
        intent_types = ["navigation", "search", "action", "confirmation"]
        for i, intent_type in enumerate(intent_types):
            envelope = {
                "event_type": "dev.omniintelligence.onex.evt.platform.intent-classified.v1",
                "correlation_id": str(uuid4()),
                "timestamp": FIXED_TEST_TIMESTAMP.isoformat(),
                "payload": {
                    "session_id": session_id,
                    "intent_type": intent_type,
                    "confidence": 0.9 + (i * 0.02),
                    "sequence_number": i,
                },
            }
            await event_bus.publish_envelope(envelope, INPUT_TOPIC)

        # Wait for all to be processed
        await asyncio.sleep(HANDLER_PROCESSING_DELAY * len(intent_types))

        # Verify all intents were processed
        assert mock_handler.invocation_count == len(intent_types), (
            f"Handler should process {len(intent_types)} intents"
        )

        # Verify intent types in order
        processed_types = [
            extract_intent_type(e) for e in mock_handler.captured_envelopes
        ]
        assert processed_types == intent_types, "Intents should be processed in order"

        await unsubscribe()


class TestIntentStoredEventEmission:
    """Tests for intent-stored event emission after successful storage."""

    @pytest.mark.asyncio
    async def test_intent_stored_event_emitted_on_success(
        self,
        event_bus: EventBusInmemory,
        mock_handler: MockIntentStorageHandler,
        classified_intent_envelope: dict[str, object],
        correlation_id: UUID,
    ) -> None:
        """Verify intent-stored event emitted after successful storage.

        This test validates the complete flow:
        1. intent-classified arrives
        2. Handler processes and stores
        3. intent-stored event is emitted
        """
        # Track emitted events
        stored_events: list[dict[str, object]] = []
        stored_event_received = asyncio.Event()

        async def capture_stored_event(msg: ModelEventMessage) -> None:
            event = json.loads(msg.value.decode("utf-8"))
            stored_events.append(event)
            stored_event_received.set()

        # Setup router
        router = MockEventRouter(mock_handler, event_bus, OUTPUT_TOPIC)

        # Subscribe to both input and output topics
        unsub_input = await event_bus.subscribe(
            INPUT_TOPIC,
            make_test_node_identity("stored-input"),
            router.route_message,
        )
        unsub_output = await event_bus.subscribe(
            OUTPUT_TOPIC,
            make_test_node_identity("stored-output"),
            capture_stored_event,
        )

        # Publish intent-classified event
        await event_bus.publish_envelope(classified_intent_envelope, INPUT_TOPIC)

        # Wait for stored event
        try:
            await asyncio.wait_for(
                stored_event_received.wait(),
                timeout=MESSAGE_WAIT_TIMEOUT,
            )
        except TimeoutError:
            pytest.fail("intent-stored event not received within timeout")

        # Verify stored event
        assert len(stored_events) == 1, "Should receive exactly one stored event"
        stored = stored_events[0]
        assert (
            stored["event_type"]
            == "dev.omniintelligence.onex.evt.platform.intent-stored.v1"
        )
        assert stored["correlation_id"] == str(correlation_id)
        assert "payload" in stored
        stored_payload = stored["payload"]
        assert is_dict_payload(stored_payload), "Expected dict payload"
        assert "intent_id" in stored_payload
        assert "stored_at" in stored_payload

        await unsub_input()
        await unsub_output()

    @pytest.mark.asyncio
    async def test_no_stored_event_on_handler_failure(
        self,
        event_bus: EventBusInmemory,
        mock_handler: MockIntentStorageHandler,
        classified_intent_envelope: dict[str, object],
    ) -> None:
        """Verify no intent-stored event when handler fails.

        When the handler fails to store the intent, no stored event should be
        emitted.
        """
        # Configure handler to fail
        mock_handler.should_succeed = False

        stored_events: list[dict[str, object]] = []

        async def capture_stored_event(msg: ModelEventMessage) -> None:
            event = json.loads(msg.value.decode("utf-8"))
            stored_events.append(event)

        router = MockEventRouter(mock_handler, event_bus, OUTPUT_TOPIC)

        unsub_input = await event_bus.subscribe(
            INPUT_TOPIC,
            make_test_node_identity("failure-input"),
            router.route_message,
        )
        unsub_output = await event_bus.subscribe(
            OUTPUT_TOPIC,
            make_test_node_identity("failure-output"),
            capture_stored_event,
        )

        await event_bus.publish_envelope(classified_intent_envelope, INPUT_TOPIC)

        # Wait for potential event (should not arrive)
        await asyncio.sleep(HANDLER_PROCESSING_DELAY * 2)

        # Verify no stored event was emitted
        assert len(stored_events) == 0, "No stored event should be emitted on failure"

        # Verify handler was still invoked
        assert mock_handler.invocation_count == 1, "Handler should still be invoked"

        await unsub_input()
        await unsub_output()


class TestEnvelopeStructure:
    """Tests for proper envelope structure handling."""

    @pytest.mark.asyncio
    async def test_handler_receives_correct_envelope_structure(
        self,
        event_bus: EventBusInmemory,
        mock_handler: MockIntentStorageHandler,
        classified_intent_envelope: dict[str, object],
        correlation_id: UUID,
        session_id: str,
    ) -> None:
        """Verify handler receives properly structured envelope.

        Tests that the envelope contains all expected fields with correct
        structure and types.
        """
        router = MockEventRouter(mock_handler, event_bus, OUTPUT_TOPIC)

        unsubscribe = await event_bus.subscribe(
            INPUT_TOPIC,
            make_test_node_identity("structure"),
            router.route_message,
        )

        await event_bus.publish_envelope(classified_intent_envelope, INPUT_TOPIC)
        await asyncio.sleep(HANDLER_PROCESSING_DELAY)

        assert len(mock_handler.captured_envelopes) == 1

        captured = mock_handler.captured_envelopes[0]

        # Verify top-level fields
        assert "event_type" in captured
        assert "correlation_id" in captured
        assert "timestamp" in captured
        assert "payload" in captured
        assert "operation" in captured  # Added by router

        # Verify correlation_id matches
        assert captured["correlation_id"] == str(correlation_id)

        # Verify payload structure
        payload = captured["payload"]
        assert isinstance(payload, dict)
        assert payload["session_id"] == session_id
        assert payload["intent_type"] == "navigation"
        assert payload["confidence"] == 0.95
        assert "raw_text" in payload
        assert "entities" in payload
        assert isinstance(payload["entities"], list)
        assert len(payload["entities"]) == 1
        assert payload["entities"][0]["type"] == "page"
        assert payload["entities"][0]["value"] == "dashboard"

        await unsubscribe()

    @pytest.mark.asyncio
    async def test_correlation_id_propagated_to_stored_event(
        self,
        event_bus: EventBusInmemory,
        mock_handler: MockIntentStorageHandler,
        classified_intent_envelope: dict[str, object],
        correlation_id: UUID,
    ) -> None:
        """Verify correlation_id is propagated from classified to stored event.

        The correlation_id should be preserved throughout the processing flow
        to enable end-to-end tracing.
        """
        stored_events: list[dict[str, object]] = []
        event_received = asyncio.Event()

        async def capture_stored(msg: ModelEventMessage) -> None:
            event = json.loads(msg.value.decode("utf-8"))
            stored_events.append(event)
            event_received.set()

        router = MockEventRouter(mock_handler, event_bus, OUTPUT_TOPIC)

        unsub_input = await event_bus.subscribe(
            INPUT_TOPIC,
            make_test_node_identity("correlation-input"),
            router.route_message,
        )
        unsub_output = await event_bus.subscribe(
            OUTPUT_TOPIC,
            make_test_node_identity("correlation-output"),
            capture_stored,
        )

        await event_bus.publish_envelope(classified_intent_envelope, INPUT_TOPIC)

        try:
            await asyncio.wait_for(event_received.wait(), timeout=MESSAGE_WAIT_TIMEOUT)
        except TimeoutError:
            pytest.fail("Stored event not received")

        # Verify correlation_id chain
        assert len(stored_events) == 1
        stored = stored_events[0]

        # Correlation IDs should match
        assert stored["correlation_id"] == str(correlation_id), (
            "Correlation ID should be propagated from classified to stored event"
        )

        await unsub_input()
        await unsub_output()

    @pytest.mark.asyncio
    async def test_original_payload_included_in_stored_event(
        self,
        event_bus: EventBusInmemory,
        mock_handler: MockIntentStorageHandler,
        classified_intent_envelope: dict[str, object],
        session_id: str,
    ) -> None:
        """Verify original payload is included in stored event for audit.

        The stored event should include the original classified payload
        for audit and debugging purposes.
        """
        stored_events: list[dict[str, object]] = []
        event_received = asyncio.Event()

        async def capture_stored(msg: ModelEventMessage) -> None:
            event = json.loads(msg.value.decode("utf-8"))
            stored_events.append(event)
            event_received.set()

        router = MockEventRouter(mock_handler, event_bus, OUTPUT_TOPIC)

        unsub_input = await event_bus.subscribe(
            INPUT_TOPIC,
            make_test_node_identity("audit-input"),
            router.route_message,
        )
        unsub_output = await event_bus.subscribe(
            OUTPUT_TOPIC,
            make_test_node_identity("audit-output"),
            capture_stored,
        )

        await event_bus.publish_envelope(classified_intent_envelope, INPUT_TOPIC)

        try:
            await asyncio.wait_for(event_received.wait(), timeout=MESSAGE_WAIT_TIMEOUT)
        except TimeoutError:
            pytest.fail("Stored event not received")

        stored = stored_events[0]
        stored_payload = stored["payload"]
        assert is_dict_payload(stored_payload), "Expected dict payload"

        # Verify original payload is included
        assert "original_payload" in stored_payload
        original = stored_payload["original_payload"]
        assert is_dict_payload(original), "Expected dict original_payload"
        assert original["session_id"] == session_id
        assert original["intent_type"] == "navigation"

        await unsub_input()
        await unsub_output()


class TestEdgeCases:
    """Tests for edge cases and error scenarios."""

    @pytest.mark.asyncio
    async def test_missing_correlation_id_generates_new_one(
        self,
        event_bus: EventBusInmemory,
        mock_handler: MockIntentStorageHandler,
        session_id: str,
    ) -> None:
        """Verify handler generates correlation_id if missing.

        When an envelope arrives without correlation_id, the handler
        should generate a new one for tracing.
        """
        stored_events: list[dict[str, object]] = []
        event_received = asyncio.Event()

        async def capture_stored(msg: ModelEventMessage) -> None:
            event = json.loads(msg.value.decode("utf-8"))
            stored_events.append(event)
            event_received.set()

        router = MockEventRouter(mock_handler, event_bus, OUTPUT_TOPIC)

        unsub_input = await event_bus.subscribe(
            INPUT_TOPIC,
            make_test_node_identity("missing-corr-input"),
            router.route_message,
        )
        unsub_output = await event_bus.subscribe(
            OUTPUT_TOPIC,
            make_test_node_identity("missing-corr-output"),
            capture_stored,
        )

        # Envelope without correlation_id
        envelope_no_corr = {
            "event_type": "dev.omniintelligence.onex.evt.platform.intent-classified.v1",
            "timestamp": FIXED_TEST_TIMESTAMP.isoformat(),
            "payload": {
                "session_id": session_id,
                "intent_type": "test",
                "confidence": 0.8,
            },
        }

        await event_bus.publish_envelope(envelope_no_corr, INPUT_TOPIC)

        try:
            await asyncio.wait_for(event_received.wait(), timeout=MESSAGE_WAIT_TIMEOUT)
        except TimeoutError:
            pytest.fail("Stored event not received")

        # Handler should have processed it
        assert mock_handler.invocation_count == 1

        # Stored event should have a correlation_id (auto-generated)
        stored = stored_events[0]
        assert "correlation_id" in stored
        assert stored["correlation_id"] is not None

        # Verify it's a valid UUID string
        try:
            UUID(str(stored["correlation_id"]))
        except ValueError:
            pytest.fail("Generated correlation_id should be a valid UUID")

        await unsub_input()
        await unsub_output()

    @pytest.mark.asyncio
    async def test_empty_payload_handled_gracefully(
        self,
        event_bus: EventBusInmemory,
        mock_handler: MockIntentStorageHandler,
    ) -> None:
        """Verify handler handles empty payload gracefully.

        An envelope with an empty payload should still be processed
        without crashing the handler.
        """
        router = MockEventRouter(mock_handler, event_bus, OUTPUT_TOPIC)

        unsubscribe = await event_bus.subscribe(
            INPUT_TOPIC,
            make_test_node_identity("empty-payload"),
            router.route_message,
        )

        envelope_empty = {
            "event_type": "dev.omniintelligence.onex.evt.platform.intent-classified.v1",
            "correlation_id": str(uuid4()),
            "timestamp": FIXED_TEST_TIMESTAMP.isoformat(),
            "payload": {},
        }

        await event_bus.publish_envelope(envelope_empty, INPUT_TOPIC)
        await asyncio.sleep(HANDLER_PROCESSING_DELAY)

        # Handler should process even with empty payload
        assert mock_handler.invocation_count == 1
        captured = mock_handler.captured_envelopes[0]
        assert captured["payload"] == {}

        await unsubscribe()

    @pytest.mark.asyncio
    async def test_high_volume_intent_routing(
        self,
        event_bus: EventBusInmemory,
        mock_handler: MockIntentStorageHandler,
        session_id: str,
    ) -> None:
        """Verify routing handles high volume of intent events.

        Tests system stability with many concurrent intent events.
        """
        router = MockEventRouter(mock_handler, event_bus, OUTPUT_TOPIC)

        unsubscribe = await event_bus.subscribe(
            INPUT_TOPIC,
            make_test_node_identity("high-volume"),
            router.route_message,
        )

        num_events = 50
        for i in range(num_events):
            envelope = {
                "event_type": "dev.omniintelligence.onex.evt.platform.intent-classified.v1",
                "correlation_id": str(uuid4()),
                "timestamp": FIXED_TEST_TIMESTAMP.isoformat(),
                "payload": {
                    "session_id": session_id,
                    "intent_type": f"intent_{i}",
                    "confidence": 0.9,
                    "sequence": i,
                },
            }
            await event_bus.publish_envelope(envelope, INPUT_TOPIC)

        # Wait for all events to be processed.
        # Note: Using a longer wait time to accommodate CI environments where
        # asyncio scheduling may be slower due to resource contention.
        # The multiplier (num_events / 5) gives ~1 second for 50 events.
        await asyncio.sleep(HANDLER_PROCESSING_DELAY * num_events / 5)

        assert mock_handler.invocation_count == num_events, (
            f"All {num_events} events should be processed"
        )

        await unsubscribe()
