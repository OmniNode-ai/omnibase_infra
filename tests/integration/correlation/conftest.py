# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Pytest fixtures for correlation ID propagation integration tests.

This module provides fixtures and helper classes for testing correlation ID
propagation across service boundaries. The fixtures capture structured log
records and generate test correlation IDs, while mock handlers simulate
publisher/subscriber patterns with correlation tracking.

Fixtures:
    log_capture: Captures structured log records for correlation ID assertion
    correlation_id: Generates a unique correlation ID for testing

Helper Functions:
    assert_correlation_in_logs: Assert correlation ID appears in logs for given boundary

Mock Handlers:
    MockHandlerA: Publisher handler that emits events with correlation tracking
    MockHandlerB: Subscriber handler that can optionally fail for testing error paths
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

import pytest

if TYPE_CHECKING:
    from typing import Protocol

    class ProtocolTestEventBus(Protocol):
        """Test-specific protocol for SimpleAsyncEventBus - NOT interchangeable with production.

        WHY THIS CANNOT USE PRODUCTION PROTOCOL:
        -----------------------------------------
        The production ProtocolEventBusLike (omnibase_infra.protocols) has a different
        signature optimized for Kafka/binary message passing:

            Production: async def publish(topic: str, key: bytes | None, value: bytes) -> None
            Test:       async def publish(topic: str, message: dict[str, object]) -> None

        Key differences:
        1. Production uses bytes (key/value) for Kafka wire format compatibility
        2. Production requires a 'key' parameter for partitioning (even if None)
        3. Test uses dict[str, object] for simpler correlation ID verification
        4. Test omits 'key' as partitioning is irrelevant to correlation testing

        ADAPTER PATTERN CONSIDERATION:
        ------------------------------
        An adapter wrapping SimpleAsyncEventBus to implement ProtocolEventBusLike was
        considered but rejected because:
        1. It would require JSON serialization/deserialization overhead for no benefit
        2. The tests specifically verify dict-based message passing (correlation_id as key)
        3. Production code already has InMemoryEventBus that implements the full protocol
        4. These tests focus on correlation propagation logic, not message serialization

        RELATION TO PRODUCTION BEHAVIOR:
        --------------------------------
        While the interface differs, the semantics tested are identical to production:
        - Messages published to a topic reach all subscribers (same as Kafka consumer groups)
        - Messages contain correlation_id that must propagate unchanged
        - Handler chains (A -> B -> C) preserve correlation context

        The SimpleAsyncEventBus used in tests provides the minimal pub/sub needed to
        verify correlation propagation without external dependencies (Kafka, InMemoryEventBus
        lifecycle management).

        See Also:
            - omnibase_infra.protocols.protocol_event_bus_like.ProtocolEventBusLike
            - omnibase_infra.event_bus.inmemory_event_bus.InMemoryEventBus
            - test_correlation_propagation.SimpleAsyncEventBus (the implementation)
        """

        async def publish(self, topic: str, message: dict[str, object]) -> None:
            """Publish a message to a topic.

            Args:
                topic: Target topic name.
                message: Message dictionary containing correlation_id and payload.
            """
            ...

        def subscribe(self, topic: str, handler: object) -> None:
            """Subscribe a handler to a topic.

            Args:
                topic: Topic name to subscribe to.
                handler: Async callable that accepts message dict.
            """
            ...


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
async def log_capture() -> AsyncGenerator[list[logging.LogRecord], None]:
    """Capture structured log records for correlation ID assertion.

    This fixture sets up a custom logging handler that captures all log records
    from the omnibase_infra logger. Records can then be inspected to verify
    correlation IDs are properly propagated through the system.

    Yields:
        List of captured LogRecord objects that can be inspected for
        correlation_id attributes and message content.

    Example:
        async def test_correlation_logging(log_capture):
            # ... perform operations that log with correlation_id ...
            assert any(
                hasattr(r, 'correlation_id') for r in log_capture
            )
    """
    captured_records: list[logging.LogRecord] = []

    class CapturingHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            captured_records.append(record)

    handler = CapturingHandler()
    logger = logging.getLogger("omnibase_infra")
    original_level = logger.level
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    try:
        yield captured_records
    finally:
        await asyncio.sleep(0)  # Flush pending async logs
        try:
            logger.removeHandler(handler)
        except ValueError:  # Handler already removed
            pass
        logger.setLevel(original_level)


@pytest.fixture
def correlation_id() -> UUID:
    """Generate a unique correlation ID for testing.

    Returns:
        A UUID4 correlation ID that can be used to trace operations
        through the system under test.

    Example:
        def test_with_correlation(correlation_id):
            result = handler.execute(correlation_id=correlation_id)
            assert result.correlation_id == correlation_id
    """
    return uuid4()


# =============================================================================
# Helper Functions
# =============================================================================


def assert_correlation_in_logs(
    records: list[logging.LogRecord],
    correlation_id: UUID,
    boundary: str,
) -> None:
    """Assert correlation ID appears in logs for given boundary.

    Searches through captured log records for entries that contain the
    specified correlation_id and have the given boundary in either the
    message or the boundary attribute (from extra dict).

    Args:
        records: List of captured LogRecord objects from log_capture fixture.
        correlation_id: The correlation ID to search for in log records.
        boundary: A string that should appear in the boundary attribute or
            log message at the boundary being tested (e.g., "handler_a_entry",
            "handler_b_exit").

    Raises:
        AssertionError: If no log record with the given correlation_id
            matches the boundary.

    Example:
        def test_boundary_logging(log_capture, correlation_id):
            handler_a.execute(correlation_id)
            assert_correlation_in_logs(
                log_capture, correlation_id, "handler_a_entry"
            )
    """
    # Log records store correlation_id as string for consistent serialization;
    # compare as strings to handle both UUID and string attribute values.
    matching = [
        r
        for r in records
        if hasattr(r, "correlation_id")
        and str(getattr(r, "correlation_id", "")) == str(correlation_id)
    ]

    # Check both message content and boundary attribute
    found = any(
        boundary in str(r.msg) or getattr(r, "boundary", "") == boundary
        for r in matching
    )

    # Collect actual boundaries for better error message
    actual_boundaries = [getattr(r, "boundary", "<no boundary>") for r in matching]

    assert found, (
        f"No log with correlation_id {correlation_id} at boundary '{boundary}'. "
        f"Found {len(matching)} records with matching correlation_id. "
        f"Actual boundaries: {actual_boundaries}"
    )


# =============================================================================
# Mock Handlers
# =============================================================================


class MockHandlerA:
    """Mock handler that publishes events with correlation tracking.

    This handler simulates a service that receives a request and publishes
    an event to a message bus. It logs at entry and exit points with the
    correlation_id in the extra dict for structured logging verification.

    Attributes:
        _bus: The event bus implementation for publishing events.
        _logger: Logger instance for this handler.

    Example:
        async def test_handler_a_publishes(event_bus, log_capture, correlation_id):
            handler = MockHandlerA(event_bus)
            await handler.execute(correlation_id)
            assert_correlation_in_logs(log_capture, correlation_id, "handler_a_entry")
    """

    def __init__(self, event_bus: ProtocolTestEventBus) -> None:
        """Initialize handler with event bus dependency.

        Args:
            event_bus: The event bus to use for publishing events.
        """
        self._bus = event_bus
        self._logger = logging.getLogger("omnibase_infra.test.handler_a")

    async def execute(self, correlation_id: UUID) -> None:
        """Execute handler and publish event with correlation.

        Logs entry and exit points with correlation_id for tracing.
        Publishes a test message to the "correlation-test" topic.

        Args:
            correlation_id: The correlation ID to propagate through the operation.
        """
        self._logger.info(
            "Handler A executing",
            extra={
                "correlation_id": str(correlation_id),
                "boundary": "handler_a_entry",
            },
        )

        # Publish event - implementation depends on event bus interface
        await self._bus.publish(
            topic="correlation-test",
            message={"action": "test", "correlation_id": str(correlation_id)},
        )

        self._logger.info(
            "Handler A published event",
            extra={"correlation_id": str(correlation_id), "boundary": "handler_a_exit"},
        )


class MockHandlerB:
    """Mock handler that subscribes and optionally fails.

    This handler simulates a service that receives events from a message bus.
    It can be configured to fail intentionally for testing error handling
    and correlation ID propagation in failure scenarios.

    Attributes:
        _should_fail: Whether to raise an error after receiving the message.
        _logger: Logger instance for this handler.
        received_messages: List of messages received by this handler.

    Example:
        async def test_handler_b_receives(log_capture, correlation_id):
            handler = MockHandlerB(should_fail=False)
            await handler.handle({"correlation_id": str(correlation_id)})
            assert len(handler.received_messages) == 1
    """

    def __init__(self, should_fail: bool = False) -> None:
        """Initialize handler with optional failure mode.

        Args:
            should_fail: If True, handler will raise InfraUnavailableError
                after logging receipt of message. Useful for testing
                error propagation with correlation IDs.
        """
        self._should_fail = should_fail
        self._logger = logging.getLogger("omnibase_infra.test.handler_b")
        self.received_messages: list[dict[str, object]] = []

    async def handle(self, message: dict[str, object]) -> None:
        """Handle incoming message with correlation tracking.

        Extracts correlation_id from the message, logs entry and exit
        points, and optionally raises an error for failure testing.

        Args:
            message: The message to handle. Expected to have a
                "correlation_id" key with a string UUID value.

        Raises:
            InfraUnavailableError: If should_fail was set to True
                during initialization.
        """
        # Extract correlation_id from message
        correlation_id = message.get("correlation_id")

        self._logger.info(
            "Handler B received event",
            extra={
                "correlation_id": str(correlation_id),
                "boundary": "handler_b_entry",
            },
        )

        self.received_messages.append(message)

        if self._should_fail:
            from omnibase_infra.enums import EnumInfraTransportType
            from omnibase_infra.errors import (
                InfraUnavailableError,
                ModelInfraErrorContext,
            )

            cid = UUID(str(correlation_id)) if correlation_id else uuid4()
            context = ModelInfraErrorContext.with_correlation(
                correlation_id=cid,
                operation="handler_b_process",
                transport_type=EnumInfraTransportType.KAFKA,
            )
            raise InfraUnavailableError(
                "Intentional failure for testing",
                context=context,
            )

        self._logger.info(
            "Handler B completed",
            extra={
                "correlation_id": str(correlation_id),
                "boundary": "handler_b_exit",
            },
        )


class MockHandlerBForwarding:
    """Mock handler variant that forwards messages to another topic.

    This handler simulates a service in a chain that receives events and
    forwards them to downstream handlers. Used for testing correlation ID
    propagation across three or more handler boundaries (A -> B -> C chains).

    Unlike MockHandlerB which is a terminal handler, this variant takes an
    event bus and publishes to a downstream topic after processing.

    Attributes:
        _bus: The event bus implementation for forwarding events.
        _logger: Logger instance for this handler.
        received_messages: List of messages received by this handler.

    Example:
        async def test_handler_chain(event_bus, log_capture, correlation_id):
            handler_b = MockHandlerBForwarding(event_bus)
            handler_c = MockHandlerC()
            event_bus.subscribe("topic-bc", handler_c.handle)
            await handler_b.handle({"correlation_id": str(correlation_id)})
            assert len(handler_c.received_messages) == 1
    """

    def __init__(self, event_bus: ProtocolTestEventBus) -> None:
        """Initialize handler with event bus dependency.

        Args:
            event_bus: The event bus to use for forwarding events.
        """
        self._bus = event_bus
        self._logger = logging.getLogger("omnibase_infra.test.handler_b")
        self.received_messages: list[dict[str, object]] = []

    async def handle(self, message: dict[str, object]) -> None:
        """Handle message and forward to next handler.

        Extracts correlation_id from the message, logs entry/exit points,
        and forwards the message to the "topic-bc" downstream topic.

        Args:
            message: The message to handle. Expected to have a
                "correlation_id" key with a string UUID value.
        """
        correlation_id = message.get("correlation_id")

        self._logger.info(
            "Handler B received event",
            extra={
                "correlation_id": str(correlation_id),
                "boundary": "handler_b_entry",
            },
        )

        self.received_messages.append(message)

        # Forward to next topic with correlation ID
        await self._bus.publish(
            topic="topic-bc",
            message={
                "action": "forwarded",
                "correlation_id": str(correlation_id),
            },
        )

        self._logger.info(
            "Handler B forwarded event",
            extra={
                "correlation_id": str(correlation_id),
                "boundary": "handler_b_exit",
            },
        )


class MockHandlerC:
    """Mock handler for third-leg chain testing.

    This handler simulates a third service in a chain, used to verify
    correlation ID propagation across three or more handler boundaries.
    Similar to MockHandlerB but without failure mode, focused on simple
    receive-and-log behavior.

    Attributes:
        _logger: Logger instance for this handler.
        received_messages: List of messages received by this handler.

    Example:
        async def test_handler_c_receives(log_capture, correlation_id):
            handler = MockHandlerC()
            await handler.handle({"correlation_id": str(correlation_id)})
            assert len(handler.received_messages) == 1
    """

    def __init__(self) -> None:
        """Initialize handler with logger and message tracking."""
        self._logger = logging.getLogger("omnibase_infra.test.handler_c")
        self.received_messages: list[dict[str, object]] = []

    async def handle(self, message: dict[str, object]) -> None:
        """Handle incoming message with correlation tracking.

        Extracts correlation_id from the message and logs entry/exit
        points for verification.

        Args:
            message: The message to handle. Expected to have a
                "correlation_id" key with a string UUID value.
        """
        correlation_id = message.get("correlation_id")

        self._logger.info(
            "Handler C received event",
            extra={
                "correlation_id": str(correlation_id),
                "boundary": "handler_c_entry",
            },
        )

        self.received_messages.append(message)

        self._logger.info(
            "Handler C completed",
            extra={
                "correlation_id": str(correlation_id),
                "boundary": "handler_c_exit",
            },
        )


__all__ = [
    "MockHandlerA",
    "MockHandlerB",
    "MockHandlerBForwarding",
    "MockHandlerC",
    "assert_correlation_in_logs",
    "correlation_id",
    "log_capture",
]
