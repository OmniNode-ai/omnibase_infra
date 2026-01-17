# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""CI-friendly integration tests for correlation ID propagation.

This module provides integration tests that verify correlation IDs are properly
propagated across handler boundaries. The tests use mock handlers and a simple
async event bus to simulate real-world message passing scenarios.

Test Coverage:
    - Handler-to-handler correlation preservation
    - Correlation ID in error context
    - Correlation ID at log boundaries
    - Multi-handler chain propagation

These tests are designed to run in CI environments without external dependencies.
"""

from __future__ import annotations

import logging
from uuid import UUID

import pytest

from omnibase_infra.errors import InfraUnavailableError
from tests.integration.correlation.conftest import (
    MockHandlerA,
    MockHandlerB,
    MockHandlerBForwarding,
    MockHandlerC,
    SimpleAsyncEventBus,
    assert_correlation_in_logs,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.asyncio,
]


# =============================================================================
# Tests
# =============================================================================


class TestCorrelationPreservation:
    """Tests for correlation ID preservation across handler boundaries.

    TODO [OMN-1349]: Add edge case tests for robustness:
    - test_correlation_missing_from_message: Handler receives message without correlation_id
    - test_correlation_malformed_uuid_string: Handler receives invalid UUID string
    - test_correlation_none_value: Handler receives explicit None as correlation_id
    - test_correlation_concurrent_messages: Multiple messages with different correlation IDs
    - test_correlation_empty_message: Handler receives empty dict message
    """

    async def test_correlation_preserved_handler_to_handler(
        self,
        event_bus: SimpleAsyncEventBus,
        log_capture: list[logging.LogRecord],
        correlation_id: UUID,
    ) -> None:
        """Test correlation ID is preserved when Handler A publishes to Handler B.

        This test verifies the fundamental correlation propagation pattern:
        1. Handler A receives a request with a correlation ID
        2. Handler A publishes an event to the message bus
        3. Handler B receives the event with the same correlation ID
        4. The correlation ID is logged at entry/exit of both handlers
        """
        # Arrange
        handler_a = MockHandlerA(event_bus)
        handler_b = MockHandlerB(should_fail=False)

        # Subscribe handler B to the topic that handler A publishes to
        event_bus.subscribe("correlation-test", handler_b.handle)

        # Act
        await handler_a.execute(correlation_id)

        # Assert - Handler B received the message
        assert len(handler_b.received_messages) == 1, (
            f"Expected Handler B to receive exactly 1 message, "
            f"but received {len(handler_b.received_messages)}"
        )

        # Assert - Correlation ID was preserved
        # Messages serialize correlation_id as string for wire transport (JSON/Kafka)
        received_correlation_id = handler_b.received_messages[0].get("correlation_id")
        assert received_correlation_id == str(correlation_id), (
            f"Correlation ID not preserved in message. "
            f"Expected '{correlation_id}', got '{received_correlation_id}'"
        )

        # Assert - Correlation ID appears in logs at handler boundaries
        assert_correlation_in_logs(log_capture, correlation_id, "handler_a_entry")
        assert_correlation_in_logs(log_capture, correlation_id, "handler_a_exit")
        assert_correlation_in_logs(log_capture, correlation_id, "handler_b_entry")
        assert_correlation_in_logs(log_capture, correlation_id, "handler_b_exit")

    async def test_correlation_in_error_context(
        self,
        event_bus: SimpleAsyncEventBus,
        log_capture: list[logging.LogRecord],
        correlation_id: UUID,
    ) -> None:
        """Test correlation ID is included in error context when handler fails.

        When Handler B is configured to fail, the raised InfraUnavailableError
        should include the correlation ID in its context for proper error
        tracing and debugging.
        """
        # Arrange
        handler_a = MockHandlerA(event_bus)
        handler_b = MockHandlerB(should_fail=True)

        event_bus.subscribe("correlation-test", handler_b.handle)

        # Act & Assert
        with pytest.raises(InfraUnavailableError) as exc_info:
            await handler_a.execute(correlation_id)

        # Assert - Error contains the correlation ID in context
        error = exc_info.value
        assert error.model.correlation_id == correlation_id, (
            f"Expected correlation_id {correlation_id} in error context, "
            f"but got {error.model.correlation_id}"
        )

        # Assert - Handler entry was logged before failure
        assert_correlation_in_logs(log_capture, correlation_id, "handler_a_entry")
        assert_correlation_in_logs(log_capture, correlation_id, "handler_b_entry")

    async def test_correlation_in_logs_at_boundaries(
        self,
        event_bus: SimpleAsyncEventBus,
        log_capture: list[logging.LogRecord],
        correlation_id: UUID,
    ) -> None:
        """Test correlation ID appears at all four handler boundaries.

        This test explicitly verifies that the correlation ID is logged
        at each of the four boundary points:
        - handler_a_entry: When Handler A starts processing
        - handler_a_exit: When Handler A finishes and publishes
        - handler_b_entry: When Handler B receives the event
        - handler_b_exit: When Handler B completes processing
        """
        # Arrange
        handler_a = MockHandlerA(event_bus)
        handler_b = MockHandlerB(should_fail=False)

        event_bus.subscribe("correlation-test", handler_b.handle)

        # Act
        await handler_a.execute(correlation_id)

        # Assert - All 4 boundaries are logged with correlation ID
        boundaries = [
            "handler_a_entry",
            "handler_a_exit",
            "handler_b_entry",
            "handler_b_exit",
        ]

        for boundary in boundaries:
            assert_correlation_in_logs(log_capture, correlation_id, boundary)

        # Additional verification: check that we have log records with
        # the correlation_id attribute set.
        # Log records store correlation_id as string for consistent serialization.
        records_with_correlation = [
            r
            for r in log_capture
            if hasattr(r, "correlation_id")
            and str(getattr(r, "correlation_id", "")) == str(correlation_id)
        ]
        assert len(records_with_correlation) >= 4, (
            f"Expected at least 4 log records with correlation_id {correlation_id}, "
            f"found {len(records_with_correlation)}"
        )

    async def test_correlation_across_three_boundaries(
        self,
        log_capture: list[logging.LogRecord],
        correlation_id: UUID,
    ) -> None:
        """Test correlation ID preserved across A -> B -> C handler chain.

        This test verifies correlation propagation in a three-handler chain:
        1. Handler A publishes to "topic-ab" (received by Handler B)
        2. Handler B publishes to "topic-bc" (received by Handler C)
        3. Correlation ID is preserved at all 6 boundary points
        """
        # Arrange - Create event bus and handlers
        event_bus = SimpleAsyncEventBus()
        handler_a = MockHandlerA(event_bus)
        handler_c = MockHandlerC()

        # Use forwarding handler variant that passes messages to next topic
        handler_b = MockHandlerBForwarding(event_bus)

        # Subscribe handlers: A -> B -> C
        event_bus.subscribe("correlation-test", handler_b.handle)
        event_bus.subscribe("topic-bc", handler_c.handle)

        # Act
        await handler_a.execute(correlation_id)

        # Assert - All handlers received messages
        assert len(handler_b.received_messages) == 1, (
            f"Expected Handler B to receive exactly 1 message, "
            f"but received {len(handler_b.received_messages)}"
        )
        assert len(handler_c.received_messages) == 1, (
            f"Expected Handler C to receive exactly 1 message, "
            f"but received {len(handler_c.received_messages)}"
        )

        # Assert - Correlation ID preserved through all handlers
        # Messages serialize correlation_id as string for wire transport (JSON/Kafka)
        b_correlation = handler_b.received_messages[0].get("correlation_id")
        c_correlation = handler_c.received_messages[0].get("correlation_id")
        assert b_correlation == str(correlation_id), (
            f"Correlation ID not preserved in Handler B. "
            f"Expected '{correlation_id}', got '{b_correlation}'"
        )
        assert c_correlation == str(correlation_id), (
            f"Correlation ID not preserved in Handler C. "
            f"Expected '{correlation_id}', got '{c_correlation}'"
        )

        # Assert - All 6 boundaries are logged with correlation ID
        boundaries = [
            "handler_a_entry",
            "handler_a_exit",
            "handler_b_entry",
            "handler_b_exit",
            "handler_c_entry",
            "handler_c_exit",
        ]

        for boundary in boundaries:
            assert_correlation_in_logs(log_capture, correlation_id, boundary)
