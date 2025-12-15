# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for BaseRuntimeHostProcess.

Tests follow TDD approach:
1. Write tests first (red phase) - this file
2. Implement RuntimeHostProcess (green phase) - in separate task
3. Refactor if needed (refactor phase)

All tests validate:
- Initialization and configuration
- Event bus ownership and lifecycle
- Handler registration via wiring
- Envelope routing and processing
- Error handling and failure envelopes
- Health check functionality

Acceptance Criteria (OMN-249):
- RuntimeHostProcess owns and manages an InMemoryEventBus
- Registers handlers via wiring.py
- Subscribes to event bus and routes envelopes
- Handles errors by producing success=False response envelopes
- Processes envelopes sequentially (no parallelism in MVP)
- Has basic shutdown (no graceful drain)

Note: These tests are written for TDD "red phase". They will fail with
ImportError until RuntimeHostProcess is implemented. This is expected
behavior for test-driven development.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from omnibase_infra.event_bus.inmemory_event_bus import InMemoryEventBus
from omnibase_infra.event_bus.models import ModelEventHeaders, ModelEventMessage
from tests.helpers import DeterministicClock, DeterministicIdGenerator

# =============================================================================
# TDD Skip Helper - Check if RuntimeHostProcess is implemented
# =============================================================================

# Try to import RuntimeHostProcess to determine if implementation exists
_RUNTIME_HOST_IMPLEMENTED = False
try:
    from omnibase_infra.runtime.protocol_lifecycle_executor import (
        ProtocolLifecycleExecutor,
    )
    from omnibase_infra.runtime.runtime_host_process import RuntimeHostProcess

    _RUNTIME_HOST_IMPLEMENTED = True
except ImportError:
    # RuntimeHostProcess not implemented yet - define a placeholder for type checking
    RuntimeHostProcess = None  # type: ignore[misc, assignment]
    ProtocolLifecycleExecutor = None  # type: ignore[misc, assignment]

# Skip marker for all tests when implementation doesn't exist
pytestmark = pytest.mark.skipif(
    not _RUNTIME_HOST_IMPLEMENTED,
    reason="RuntimeHostProcess not yet implemented (TDD red phase)",
)

# =============================================================================
# Mock Classes for Testing
# =============================================================================


class MockHandler:
    """Mock handler that records calls for testing."""

    def __init__(self, handler_type: str = "mock") -> None:
        """Initialize mock handler.

        Args:
            handler_type: The type identifier for this handler.
        """
        self.handler_type = handler_type
        self.calls: list[dict[str, object]] = []
        self.initialized: bool = False
        self.shutdown_called: bool = False
        self.execute_delay: float = 0.0
        self.execute_error: Exception | None = None

    async def initialize(self, config: dict[str, object]) -> None:
        """Initialize the mock handler."""
        self.initialized = True
        self.config = config

    async def execute(self, envelope: dict[str, object]) -> dict[str, object]:
        """Execute the mock handler with the given envelope.

        Records the call and returns a success response.

        Args:
            envelope: The envelope to process.

        Returns:
            Response dict with status and correlation_id.
        """
        if self.execute_delay > 0:
            await asyncio.sleep(self.execute_delay)

        if self.execute_error is not None:
            raise self.execute_error

        self.calls.append(envelope)
        correlation_id = envelope.get("correlation_id", uuid4())
        return {
            "status": "success",
            "correlation_id": correlation_id,
            "payload": {"handler_type": self.handler_type, "processed": True},
        }

    async def shutdown(self) -> None:
        """Shutdown the mock handler."""
        self.shutdown_called = True
        self.initialized = False

    async def health_check(self) -> dict[str, object]:
        """Return health check status."""
        return {
            "healthy": self.initialized,
            "handler_type": self.handler_type,
        }


class MockFailingHandler(MockHandler):
    """Mock handler that always fails during execution."""

    def __init__(self, error_message: str = "Mock execution failed") -> None:
        """Initialize failing mock handler.

        Args:
            error_message: The error message to raise on execute.
        """
        super().__init__(handler_type="failing")
        self.error_message = error_message

    async def execute(self, envelope: dict[str, object]) -> dict[str, object]:
        """Always raise an exception."""
        self.calls.append(envelope)
        raise RuntimeError(self.error_message)


class MockEventBus:
    """Mock event bus that tracks operations for testing."""

    def __init__(self) -> None:
        """Initialize mock event bus."""
        self.started: bool = False
        self.closed: bool = False
        self.subscriptions: list[tuple[str, str, Callable[..., object]]] = []
        self.published: list[tuple[str, bytes | None, bytes]] = []
        self.unsubscribe_callbacks: list[AsyncMock] = []

    async def start(self) -> None:
        """Start the mock event bus."""
        self.started = True

    async def close(self) -> None:
        """Close the mock event bus."""
        self.closed = True
        self.started = False

    async def shutdown(self) -> None:
        """Shutdown alias for close."""
        await self.close()

    async def subscribe(
        self,
        topic: str,
        group_id: str,
        on_message: Callable[..., object],
    ) -> AsyncMock:
        """Subscribe to a topic.

        Args:
            topic: Topic to subscribe to.
            group_id: Consumer group identifier for this subscription.
            on_message: Async callback invoked for each message.

        Returns:
            Unsubscribe callback.
        """
        self.subscriptions.append((topic, group_id, on_message))
        unsubscribe = AsyncMock()
        self.unsubscribe_callbacks.append(unsubscribe)
        return unsubscribe

    async def publish(
        self,
        topic: str,
        key: bytes | None,
        value: bytes,
        headers: ModelEventHeaders | None = None,
    ) -> None:
        """Publish a message to a topic.

        Args:
            topic: Topic to publish to.
            key: Message key.
            value: Message value.
            headers: Optional message headers.
        """
        self.published.append((topic, key, value))

    async def publish_envelope(
        self,
        envelope: dict[str, object] | object,
        topic: str,
    ) -> None:
        """Publish an envelope to a topic.

        Args:
            envelope: The envelope to publish.
            topic: Topic to publish to.
        """
        if hasattr(envelope, "model_dump"):
            value = json.dumps(envelope.model_dump()).encode("utf-8")
        elif isinstance(envelope, dict):
            value = json.dumps(envelope).encode("utf-8")
        else:
            value = str(envelope).encode("utf-8")
        self.published.append((topic, None, value))

    async def health_check(self) -> dict[str, object]:
        """Return health check status."""
        return {
            "healthy": self.started and not self.closed,
            "started": self.started,
            "closed": self.closed,
        }


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_event_bus() -> MockEventBus:
    """Create mock event bus for testing."""
    return MockEventBus()


@pytest.fixture
def mock_handler() -> MockHandler:
    """Create mock handler for testing."""
    return MockHandler(handler_type="http")


@pytest.fixture
def mock_failing_handler() -> MockFailingHandler:
    """Create mock failing handler for testing."""
    return MockFailingHandler()


@pytest.fixture
def deterministic_id_gen() -> DeterministicIdGenerator:
    """Create deterministic ID generator for testing."""
    return DeterministicIdGenerator(seed=100)


@pytest.fixture
def deterministic_clock() -> DeterministicClock:
    """Create deterministic clock for testing."""
    return DeterministicClock()


@pytest.fixture
def sample_envelope() -> dict[str, object]:
    """Create sample envelope for testing."""
    return {
        "operation": "http.get",
        "payload": {"url": "https://example.com/api"},
        "correlation_id": uuid4(),
        "handler_type": "http",
    }


@pytest.fixture
def sample_event_message(sample_envelope: dict[str, object]) -> ModelEventMessage:
    """Create sample event message for testing."""
    return ModelEventMessage(
        topic="test.input",
        key=None,
        value=json.dumps(sample_envelope).encode("utf-8"),
        headers=ModelEventHeaders(
            source="test",
            event_type="test.request",
            correlation_id=uuid4(),
        ),
        offset="0",
        partition=0,
    )


# =============================================================================
# TestRuntimeHostProcessInitialization
# =============================================================================


class TestRuntimeHostProcessInitialization:
    """Test initialization and configuration."""

    @pytest.mark.asyncio
    async def test_creates_event_bus_on_init(self) -> None:
        """Test that RuntimeHostProcess creates an event bus on initialization.

        The process should own an InMemoryEventBus instance that it manages
        throughout its lifecycle.
        """
        # Import will fail until implementation exists - this is TDD red phase

        process = RuntimeHostProcess()

        assert process.event_bus is not None
        assert isinstance(process.event_bus, InMemoryEventBus)

    @pytest.mark.asyncio
    async def test_initializes_with_default_config(self) -> None:
        """Test that RuntimeHostProcess initializes with default configuration.

        Default config should include:
        - input_topic: The topic to subscribe to for incoming envelopes
        - output_topic: The topic to publish responses to
        - group_id: Consumer group identifier
        """

        process = RuntimeHostProcess()

        # Should have default configuration values
        assert process.input_topic is not None
        assert process.output_topic is not None
        assert process.group_id is not None

    @pytest.mark.asyncio
    async def test_initializes_with_custom_config(self) -> None:
        """Test that RuntimeHostProcess accepts custom configuration."""

        config: dict[str, object] = {
            "input_topic": "custom.input",
            "output_topic": "custom.output",
            "group_id": "custom-group",
        }

        process = RuntimeHostProcess(config=config)

        assert process.input_topic == "custom.input"
        assert process.output_topic == "custom.output"
        assert process.group_id == "custom-group"

    @pytest.mark.asyncio
    async def test_not_started_by_default(self) -> None:
        """Test that RuntimeHostProcess is not started by default.

        The process should be in an unstarted state after construction,
        requiring explicit start() call to begin processing.
        """

        process = RuntimeHostProcess()

        assert process.is_running is False
        # Event bus should also not be started
        health = await process.event_bus.health_check()
        assert health["started"] is False


# =============================================================================
# TestRuntimeHostProcessTimeoutValidation
# =============================================================================


class TestRuntimeHostProcessTimeoutValidation:
    """Test health_check_timeout_seconds configuration validation.

    Tests the bounds validation (1-60 seconds) per ModelLifecycleSubcontract.
    """

    @pytest.mark.asyncio
    async def test_timeout_below_minimum_is_clamped(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test that timeout values below minimum are clamped to 1.0."""
        import logging

        with caplog.at_level(logging.WARNING):
            process = RuntimeHostProcess(config={"health_check_timeout_seconds": 0.5})

        # Should be clamped to minimum
        assert process._health_check_timeout_seconds == 1.0

        # Warning should be logged
        warning_logs = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warning_logs) >= 1
        assert any(
            "out of valid range" in r.message or "clamping" in r.message
            for r in warning_logs
        )

    @pytest.mark.asyncio
    async def test_timeout_above_maximum_is_clamped(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test that timeout values above maximum are clamped to 60.0."""
        import logging

        with caplog.at_level(logging.WARNING):
            process = RuntimeHostProcess(config={"health_check_timeout_seconds": 120.0})

        # Should be clamped to maximum
        assert process._health_check_timeout_seconds == 60.0

        # Warning should be logged
        warning_logs = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warning_logs) >= 1
        assert any(
            "out of valid range" in r.message or "clamping" in r.message
            for r in warning_logs
        )

    @pytest.mark.asyncio
    async def test_timeout_within_range_is_accepted(self) -> None:
        """Test that timeout values within range are accepted as-is."""
        # Test minimum boundary
        process_min = RuntimeHostProcess(config={"health_check_timeout_seconds": 1.0})
        assert process_min._health_check_timeout_seconds == 1.0

        # Test maximum boundary
        process_max = RuntimeHostProcess(config={"health_check_timeout_seconds": 60.0})
        assert process_max._health_check_timeout_seconds == 60.0

        # Test middle value
        process_mid = RuntimeHostProcess(config={"health_check_timeout_seconds": 30.0})
        assert process_mid._health_check_timeout_seconds == 30.0

    @pytest.mark.asyncio
    async def test_timeout_integer_within_range_is_accepted(self) -> None:
        """Test that integer timeout values within range are accepted."""
        process = RuntimeHostProcess(config={"health_check_timeout_seconds": 10})
        assert process._health_check_timeout_seconds == 10.0

    @pytest.mark.asyncio
    async def test_timeout_string_within_range_is_accepted(self) -> None:
        """Test that valid string timeout values within range are accepted."""
        process = RuntimeHostProcess(config={"health_check_timeout_seconds": "15.5"})
        assert process._health_check_timeout_seconds == 15.5

    @pytest.mark.asyncio
    async def test_invalid_string_timeout_falls_back_to_default(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test that invalid string timeout values fall back to default."""
        import logging

        with caplog.at_level(logging.WARNING):
            process = RuntimeHostProcess(
                config={"health_check_timeout_seconds": "not-a-number"}
            )

        # Should fall back to default
        assert process._health_check_timeout_seconds == 5.0

        # Warning should be logged
        warning_logs = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warning_logs) >= 1
        assert any(
            "Invalid health_check_timeout_seconds" in r.message for r in warning_logs
        )

    @pytest.mark.asyncio
    async def test_timeout_zero_is_clamped_to_minimum(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test that timeout value of 0 is clamped to minimum."""
        import logging

        with caplog.at_level(logging.WARNING):
            process = RuntimeHostProcess(config={"health_check_timeout_seconds": 0})

        # Should be clamped to minimum
        assert process._health_check_timeout_seconds == 1.0

    @pytest.mark.asyncio
    async def test_timeout_negative_is_clamped_to_minimum(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test that negative timeout values are clamped to minimum."""
        import logging

        with caplog.at_level(logging.WARNING):
            process = RuntimeHostProcess(config={"health_check_timeout_seconds": -5.0})

        # Should be clamped to minimum
        assert process._health_check_timeout_seconds == 1.0

    @pytest.mark.asyncio
    async def test_timeout_default_when_not_specified(self) -> None:
        """Test that default timeout is used when not specified in config."""
        process = RuntimeHostProcess()
        assert process._health_check_timeout_seconds == 5.0

    @pytest.mark.asyncio
    async def test_timeout_default_when_none(self) -> None:
        """Test that default timeout is used when explicitly set to None."""
        process = RuntimeHostProcess(config={"health_check_timeout_seconds": None})
        assert process._health_check_timeout_seconds == 5.0


# =============================================================================
# TestRuntimeHostProcessLifecycle
# =============================================================================


class TestRuntimeHostProcessLifecycle:
    """Test start/stop lifecycle."""

    @pytest.mark.asyncio
    async def test_start_starts_event_bus(self) -> None:
        """Test that start() starts the event bus."""

        process = RuntimeHostProcess()
        await process.start()

        try:
            health = await process.event_bus.health_check()
            assert health["started"] is True
            assert process.is_running is True
        finally:
            await process.stop()

    @pytest.mark.asyncio
    async def test_start_wires_handlers(self) -> None:
        """Test that start() wires handlers via the wiring module.

        The RuntimeHostProcess should use the wiring module to register
        all configured handlers when started.
        """

        process = RuntimeHostProcess()

        with patch(
            "omnibase_infra.runtime.runtime_host_process.wire_handlers"
        ) as mock_wire:
            mock_wire.return_value = {}
            await process.start()

            try:
                mock_wire.assert_called_once()
            finally:
                await process.stop()

    @pytest.mark.asyncio
    async def test_start_subscribes_to_input_topic(self) -> None:
        """Test that start() subscribes to the input topic.

        The process should subscribe to the configured input topic
        to receive envelopes for processing.
        """

        process = RuntimeHostProcess(config={"input_topic": "test.input"})
        await process.start()

        try:
            # Verify subscription was created
            # The actual subscription mechanism depends on implementation
            assert process._subscription is not None
        finally:
            await process.stop()

    @pytest.mark.asyncio
    async def test_stop_closes_event_bus(self) -> None:
        """Test that stop() closes the event bus."""

        process = RuntimeHostProcess()
        await process.start()
        await process.stop()

        health = await process.event_bus.health_check()
        assert health["started"] is False
        assert process.is_running is False

    @pytest.mark.asyncio
    async def test_stop_clears_subscriptions(self) -> None:
        """Test that stop() clears all subscriptions.

        All subscriptions should be unsubscribed when the process stops
        to prevent message delivery to a stopped process.
        """

        process = RuntimeHostProcess()
        await process.start()
        await process.stop()

        # Subscription should be cleared
        assert process._subscription is None

    @pytest.mark.asyncio
    async def test_double_start_is_safe(self) -> None:
        """Test that calling start() twice is safe (idempotent)."""

        process = RuntimeHostProcess()
        await process.start()
        await process.start()  # Second start should be safe

        try:
            assert process.is_running is True
            health = await process.event_bus.health_check()
            assert health["started"] is True
        finally:
            await process.stop()

    @pytest.mark.asyncio
    async def test_double_stop_is_safe(self) -> None:
        """Test that calling stop() twice is safe (idempotent)."""

        process = RuntimeHostProcess()
        await process.start()
        await process.stop()
        await process.stop()  # Second stop should be safe

        assert process.is_running is False

    @pytest.mark.asyncio
    async def test_stop_calls_shutdown_on_all_handlers(self) -> None:
        """Test that stop() calls shutdown() on all registered handlers.

        When stopping, the process should call shutdown() on each handler
        to allow them to release resources (DB connections, Kafka connections, etc.).
        """
        process = RuntimeHostProcess()

        # Create multiple handlers
        http_handler = MockHandler(handler_type="http")
        db_handler = MockHandler(handler_type="db")

        # Patch _populate_handlers_from_registry to prevent auto-population
        async def noop_populate() -> None:
            pass

        with patch.object(process, "_populate_handlers_from_registry", noop_populate):
            with patch.object(
                process, "_handlers", {"http": http_handler, "db": db_handler}
            ):
                await process.start()

                # Verify handlers are not shutdown yet
                assert http_handler.shutdown_called is False
                assert db_handler.shutdown_called is False

                await process.stop()

                # Verify shutdown was called on all handlers
                assert http_handler.shutdown_called is True
                assert db_handler.shutdown_called is True

    @pytest.mark.asyncio
    async def test_stop_continues_on_handler_shutdown_error(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test that stop() continues shutting down handlers even if one fails.

        If a handler's shutdown() raises an exception, the process should
        log the error but continue shutting down other handlers.
        """
        import logging

        process = RuntimeHostProcess()

        # Create handlers where one will fail during shutdown
        http_handler = MockHandler(handler_type="http")
        failing_handler = MockHandler(handler_type="failing")
        db_handler = MockHandler(handler_type="db")

        # Make failing_handler's shutdown raise an exception
        async def failing_shutdown() -> None:
            raise RuntimeError("Simulated shutdown failure")

        failing_handler.shutdown = failing_shutdown

        # Patch _populate_handlers_from_registry to prevent auto-population
        async def noop_populate() -> None:
            pass

        with patch.object(process, "_populate_handlers_from_registry", noop_populate):
            with patch.object(
                process,
                "_handlers",
                {"http": http_handler, "failing": failing_handler, "db": db_handler},
            ):
                await process.start()

                with caplog.at_level(logging.ERROR):
                    await process.stop()

                # Verify other handlers were still shutdown
                assert http_handler.shutdown_called is True
                assert db_handler.shutdown_called is True

                # Verify error was logged
                error_logs = [r for r in caplog.records if r.levelno == logging.ERROR]
                assert len(error_logs) >= 1
                assert any(
                    "Error shutting down handler" in r.message for r in error_logs
                )

    @pytest.mark.asyncio
    async def test_stop_handles_handlers_without_shutdown_method(self) -> None:
        """Test that stop() gracefully handles handlers without shutdown().

        Some handlers may not implement a shutdown() method. The process
        should skip shutdown for those handlers without raising an error.
        """
        process = RuntimeHostProcess()

        # Create a handler-like object without shutdown method
        class HandlerWithoutShutdown:
            def __init__(self) -> None:
                self.handler_type = "no_shutdown"
                self.calls: list[dict[str, object]] = []

            async def execute(self, envelope: dict[str, object]) -> dict[str, object]:
                self.calls.append(envelope)
                return {"status": "success"}

        no_shutdown_handler = HandlerWithoutShutdown()
        regular_handler = MockHandler(handler_type="regular")

        # Patch _populate_handlers_from_registry to prevent auto-population
        async def noop_populate() -> None:
            pass

        with patch.object(process, "_populate_handlers_from_registry", noop_populate):
            with patch.object(
                process,
                "_handlers",
                {"no_shutdown": no_shutdown_handler, "regular": regular_handler},
            ):
                await process.start()
                # Should not raise any exception
                await process.stop()

                # Regular handler should still be shutdown
                assert regular_handler.shutdown_called is True

        # Process should be stopped
        assert process.is_running is False


# =============================================================================
# TestRuntimeHostProcessEnvelopeRouting
# =============================================================================


class TestRuntimeHostProcessEnvelopeRouting:
    """Test envelope consumption and routing."""

    @pytest.mark.asyncio
    async def test_routes_envelope_to_correct_handler(
        self,
        mock_handler: MockHandler,
        sample_envelope: dict[str, object],
    ) -> None:
        """Test that envelopes are routed to the correct handler based on type.

        The process should extract the handler_type from the envelope and
        route it to the appropriate registered handler.
        """

        process = RuntimeHostProcess()

        # Register mock handler
        with patch.object(process, "_handlers", {"http": mock_handler}):
            await process.start()

            try:
                await process._handle_envelope(sample_envelope)

                assert len(mock_handler.calls) == 1
                assert mock_handler.calls[0]["operation"] == "http.get"
            finally:
                await process.stop()

    @pytest.mark.asyncio
    async def test_publishes_response_to_output_topic(
        self,
        sample_envelope: dict[str, object],
    ) -> None:
        """Test that handler responses are published to the output topic."""

        process = RuntimeHostProcess(config={"output_topic": "test.output"})
        mock_handler = MockHandler(handler_type="http")

        with patch.object(process, "_handlers", {"http": mock_handler}):
            await process.start()

            try:
                # Spy on publish_envelope
                with patch.object(
                    process.event_bus, "publish_envelope", new_callable=AsyncMock
                ) as mock_publish:
                    await process._handle_envelope(sample_envelope)

                    mock_publish.assert_called_once()
                    call_args = mock_publish.call_args
                    assert call_args[0][1] == "test.output"  # topic argument
            finally:
                await process.stop()

    @pytest.mark.asyncio
    async def test_sequential_envelope_processing(
        self,
        mock_handler: MockHandler,
    ) -> None:
        """Test that envelopes are processed sequentially (no parallelism).

        MVP requirement: Envelopes should be processed one at a time,
        not in parallel.
        """

        process = RuntimeHostProcess()
        mock_handler.execute_delay = 0.1  # 100ms delay per execution

        envelopes = [
            {
                "operation": "http.get",
                "payload": {"url": f"https://example.com/api/{i}"},
                "correlation_id": uuid4(),
                "handler_type": "http",
            }
            for i in range(3)
        ]

        with patch.object(process, "_handlers", {"http": mock_handler}):
            await process.start()

            try:
                # Process envelopes - should be sequential
                for envelope in envelopes:
                    await process._handle_envelope(envelope)

                # All should be processed in order
                assert len(mock_handler.calls) == 3
                for i, call in enumerate(mock_handler.calls):
                    payload = call["payload"]
                    assert isinstance(payload, dict)
                    url = payload["url"]
                    assert isinstance(url, str)
                    assert f"api/{i}" in url
            finally:
                await process.stop()

    @pytest.mark.asyncio
    async def test_error_produces_failure_envelope(
        self,
        mock_failing_handler: MockFailingHandler,
    ) -> None:
        """Test that handler errors produce success=False response envelopes.

        When a handler raises an exception, the process should produce
        a failure envelope with success=False and error details.
        """

        process = RuntimeHostProcess(config={"output_topic": "test.output"})
        envelope = {
            "operation": "failing.execute",
            "payload": {},
            "correlation_id": uuid4(),
            "handler_type": "failing",
        }

        with patch.object(process, "_handlers", {"failing": mock_failing_handler}):
            await process.start()

            try:
                with patch.object(
                    process.event_bus, "publish_envelope", new_callable=AsyncMock
                ) as mock_publish:
                    await process._handle_envelope(envelope)

                    # Should have published a failure response
                    mock_publish.assert_called_once()
                    published_envelope = mock_publish.call_args[0][0]

                    # Verify it's a failure envelope
                    if hasattr(published_envelope, "model_dump"):
                        data = published_envelope.model_dump()
                    else:
                        data = published_envelope

                    assert data.get("success") is False or data.get("status") == "error"
            finally:
                await process.stop()


# =============================================================================
# TestRuntimeHostProcessErrorHandling
# =============================================================================


class TestRuntimeHostProcessErrorHandling:
    """Test error handling patterns."""

    @pytest.mark.asyncio
    async def test_handler_error_returns_failure_envelope(
        self,
        mock_failing_handler: MockFailingHandler,
    ) -> None:
        """Test that handler execution errors return failure envelopes.

        Failure envelopes should contain:
        - success: False (or status: "error")
        - error: Description of the error
        - correlation_id: Preserved from the original envelope
        """

        process = RuntimeHostProcess()
        correlation_id = uuid4()
        envelope = {
            "operation": "failing.execute",
            "payload": {},
            "correlation_id": correlation_id,
            "handler_type": "failing",
        }

        with patch.object(process, "_handlers", {"failing": mock_failing_handler}):
            await process.start()

            try:
                with patch.object(
                    process.event_bus, "publish_envelope", new_callable=AsyncMock
                ) as mock_publish:
                    await process._handle_envelope(envelope)

                    published_envelope = mock_publish.call_args[0][0]
                    if hasattr(published_envelope, "model_dump"):
                        data = published_envelope.model_dump()
                    else:
                        data = published_envelope

                    # Should preserve correlation_id (as string after serialization)
                    assert data.get("correlation_id") == str(correlation_id)
                    # Should indicate failure
                    assert data.get("success") is False or data.get("status") == "error"
                    # Should include error information
                    assert "error" in data or "error_message" in data
            finally:
                await process.stop()

    @pytest.mark.asyncio
    async def test_unknown_handler_type_returns_failure(self) -> None:
        """Test that unknown handler types return failure envelopes.

        When an envelope specifies a handler_type that is not registered,
        the process should return a failure envelope indicating the error.
        """

        process = RuntimeHostProcess()
        envelope = {
            "operation": "unknown.execute",
            "payload": {},
            "correlation_id": uuid4(),
            "handler_type": "unknown_handler",
        }

        with patch.object(process, "_handlers", {}):  # No handlers registered
            await process.start()

            try:
                with patch.object(
                    process.event_bus, "publish_envelope", new_callable=AsyncMock
                ) as mock_publish:
                    await process._handle_envelope(envelope)

                    published_envelope = mock_publish.call_args[0][0]
                    if hasattr(published_envelope, "model_dump"):
                        data = published_envelope.model_dump()
                    else:
                        data = published_envelope

                    assert data.get("success") is False or data.get("status") == "error"
                    # Error should mention unknown handler
                    error_msg = str(data.get("error", data.get("error_message", "")))
                    assert (
                        "unknown" in error_msg.lower()
                        or "not found" in error_msg.lower()
                        or "not registered" in error_msg.lower()
                    )
            finally:
                await process.stop()

    @pytest.mark.asyncio
    async def test_invalid_envelope_returns_failure(self) -> None:
        """Test that invalid envelopes return failure responses.

        Envelopes missing required fields (operation, handler_type) should
        result in failure responses rather than exceptions.
        """

        process = RuntimeHostProcess()
        invalid_envelope = {
            # Missing operation and handler_type
            "payload": {},
            "correlation_id": uuid4(),
        }

        await process.start()

        try:
            with patch.object(
                process.event_bus, "publish_envelope", new_callable=AsyncMock
            ) as mock_publish:
                await process._handle_envelope(invalid_envelope)

                published_envelope = mock_publish.call_args[0][0]
                if hasattr(published_envelope, "model_dump"):
                    data = published_envelope.model_dump()
                else:
                    data = published_envelope

                assert data.get("success") is False or data.get("status") == "error"
        finally:
            await process.stop()

    @pytest.mark.asyncio
    async def test_errors_include_correlation_id(
        self,
        deterministic_id_gen: DeterministicIdGenerator,
    ) -> None:
        """Test that error responses include the original correlation_id.

        Correlation IDs must be preserved in error responses for proper
        request tracking and debugging.
        """

        process = RuntimeHostProcess()
        correlation_id = deterministic_id_gen.next_uuid()

        # Invalid envelope with correlation_id
        invalid_envelope = {
            "payload": {},
            "correlation_id": correlation_id,
        }

        await process.start()

        try:
            with patch.object(
                process.event_bus, "publish_envelope", new_callable=AsyncMock
            ) as mock_publish:
                await process._handle_envelope(invalid_envelope)

                published_envelope = mock_publish.call_args[0][0]
                if hasattr(published_envelope, "model_dump"):
                    data = published_envelope.model_dump()
                else:
                    data = published_envelope

                # Correlation ID must be preserved (as string after serialization)
                assert data.get("correlation_id") == str(correlation_id)
        finally:
            await process.stop()


# =============================================================================
# TestRuntimeHostProcessHealthCheck
# =============================================================================


class TestRuntimeHostProcessHealthCheck:
    """Test health check functionality."""

    @pytest.mark.asyncio
    async def test_health_check_returns_status(self) -> None:
        """Test that health_check returns a status dict.

        Health check should return a dictionary with at minimum:
        - healthy: boolean indicating overall health
        - is_running: boolean indicating if process is running
        """

        process = RuntimeHostProcess()

        health = await process.health_check()

        assert isinstance(health, dict)
        assert "healthy" in health
        assert "is_running" in health

    @pytest.mark.asyncio
    async def test_health_check_includes_event_bus_status(self) -> None:
        """Test that health_check includes event bus status.

        Health check should aggregate the event bus health status
        into the overall health response.
        """

        process = RuntimeHostProcess()

        # Patch _populate_handlers_from_registry to prevent handler instantiation
        # failures from affecting health status (singleton registry may have
        # handlers from other tests that fail without proper config)
        async def noop_populate() -> None:
            pass

        with patch.object(process, "_populate_handlers_from_registry", noop_populate):
            await process.start()

            try:
                health = await process.health_check()

                # Should include event_bus status
                assert "event_bus" in health or "event_bus_healthy" in health

                # Should include failed_handlers and registered_handlers
                assert "failed_handlers" in health
                assert "registered_handlers" in health

                # When running with no failed handlers, should be healthy
                assert health["healthy"] is True
                assert health["is_running"] is True
            finally:
                await process.stop()

    @pytest.mark.asyncio
    async def test_health_check_reflects_stopped_state(self) -> None:
        """Test that health_check reflects stopped state accurately."""

        process = RuntimeHostProcess()

        # Before starting
        health = await process.health_check()
        assert health["is_running"] is False

        # Patch _populate_handlers_from_registry to prevent handler instantiation
        # failures from affecting health status
        async def noop_populate() -> None:
            pass

        with patch.object(process, "_populate_handlers_from_registry", noop_populate):
            # After starting
            await process.start()
            health = await process.health_check()
            assert health["is_running"] is True

            # After stopping
            await process.stop()
            health = await process.health_check()
            assert health["is_running"] is False

    @pytest.mark.asyncio
    async def test_health_check_includes_degraded_field(self) -> None:
        """Test that health_check includes degraded field.

        The degraded field indicates partial functionality:
        - degraded=False: Fully operational (no handler failures)
        - degraded=True: Running with reduced functionality (some handlers failed)
        """

        process = RuntimeHostProcess()

        # Patch _populate_handlers_from_registry to prevent handler instantiation
        async def noop_populate() -> None:
            pass

        with patch.object(process, "_populate_handlers_from_registry", noop_populate):
            await process.start()

            try:
                health = await process.health_check()

                # Should include degraded field
                assert "degraded" in health
                # With no failed handlers, should not be degraded
                assert health["degraded"] is False
                assert health["healthy"] is True
            finally:
                await process.stop()

    @pytest.mark.asyncio
    async def test_health_check_degraded_when_handlers_fail(self) -> None:
        """Test that health_check shows degraded=True when handlers fail.

        When handlers fail to instantiate during start(), the process
        should report as degraded (running but with reduced functionality).
        """

        process = RuntimeHostProcess()

        # Patch _populate_handlers_from_registry to prevent handler instantiation
        async def noop_populate() -> None:
            pass

        with patch.object(process, "_populate_handlers_from_registry", noop_populate):
            await process.start()

            try:
                # Simulate failed handlers by directly setting _failed_handlers
                process._failed_handlers = {"test_handler": "Mock failure"}

                health = await process.health_check()

                # Should be degraded since handlers failed
                assert health["degraded"] is True
                # Should NOT be healthy since handlers failed
                assert health["healthy"] is False
                # Should still be running
                assert health["is_running"] is True
                # Failed handlers should be reported
                assert "test_handler" in health["failed_handlers"]
            finally:
                await process.stop()

    @pytest.mark.asyncio
    async def test_health_check_not_degraded_when_stopped(self) -> None:
        """Test that health_check is not degraded when process is stopped.

        Degraded state requires the process to be running. A stopped
        process with failed handlers is not degraded, just not running.
        """

        process = RuntimeHostProcess()

        # Simulate failed handlers even though not started
        process._failed_handlers = {"test_handler": "Mock failure"}

        health = await process.health_check()

        # Should NOT be degraded since not running
        assert health["degraded"] is False
        # Should NOT be healthy since not running
        assert health["healthy"] is False
        # Should not be running
        assert health["is_running"] is False

    @pytest.mark.asyncio
    async def test_health_check_includes_handler_health(
        self,
        mock_handler: MockHandler,
    ) -> None:
        """Test that health_check aggregates handler health status.

        Health check should iterate all registered handlers and call
        their health_check() method, aggregating results into the response.
        """

        process = RuntimeHostProcess()
        mock_handler.initialized = True  # Mark as healthy

        # Patch _populate_handlers_from_registry to prevent handler instantiation
        async def noop_populate() -> None:
            pass

        with patch.object(process, "_populate_handlers_from_registry", noop_populate):
            with patch.object(process, "_handlers", {"http": mock_handler}):
                await process.start()

                try:
                    health = await process.health_check()

                    # Should include handlers key
                    assert "handlers" in health
                    # Should include http handler health
                    assert "http" in health["handlers"]
                    # Handler should be healthy (initialized=True)
                    assert health["handlers"]["http"]["healthy"] is True
                    # Overall health should be True
                    assert health["healthy"] is True
                finally:
                    await process.stop()

    @pytest.mark.asyncio
    async def test_health_check_unhealthy_handler(
        self,
        mock_handler: MockHandler,
    ) -> None:
        """Test that unhealthy handler makes overall health False.

        When a registered handler reports unhealthy status, the overall
        health check should report healthy=False.
        """

        process = RuntimeHostProcess()
        mock_handler.initialized = False  # Mark as unhealthy

        # Patch _populate_handlers_from_registry to prevent handler instantiation
        async def noop_populate() -> None:
            pass

        with patch.object(process, "_populate_handlers_from_registry", noop_populate):
            with patch.object(process, "_handlers", {"http": mock_handler}):
                await process.start()

                try:
                    health = await process.health_check()

                    # Handler should be unhealthy
                    assert health["handlers"]["http"]["healthy"] is False
                    # Overall health should be False due to unhealthy handler
                    assert health["healthy"] is False
                    # Process should still be running
                    assert health["is_running"] is True
                finally:
                    await process.stop()

    @pytest.mark.asyncio
    async def test_health_check_handler_error_caught(self) -> None:
        """Test that handler health_check errors are caught and reported.

        When a handler's health_check() raises an exception, the error
        should be caught, reported in the response, and not crash the
        overall health check.
        """

        process = RuntimeHostProcess()

        class ErrorHandler:
            """Handler that raises an error during health check."""

            async def health_check(self) -> dict[str, object]:
                raise RuntimeError("Health check failed")

        # Patch _populate_handlers_from_registry to prevent handler instantiation
        async def noop_populate() -> None:
            pass

        with patch.object(process, "_populate_handlers_from_registry", noop_populate):
            with patch.object(process, "_handlers", {"error": ErrorHandler()}):
                await process.start()

                try:
                    # Should not crash
                    health = await process.health_check()

                    # Error handler should be reported as unhealthy
                    assert "error" in health["handlers"]
                    assert health["handlers"]["error"]["healthy"] is False
                    # Error message should be captured
                    assert "error" in health["handlers"]["error"]
                    assert "Health check failed" in health["handlers"]["error"]["error"]
                    # Overall health should be False
                    assert health["healthy"] is False
                finally:
                    await process.stop()

    @pytest.mark.asyncio
    async def test_health_check_handler_without_health_check_method(self) -> None:
        """Test that handlers without health_check method are assumed healthy.

        When a handler does not implement a health_check() method, it should
        be assumed healthy with a note indicating no health_check method.
        """

        process = RuntimeHostProcess()

        class SimpleHandler:
            """Handler without health_check method."""

            async def execute(self, envelope: dict[str, object]) -> dict[str, object]:
                return {"status": "success"}

        # Patch _populate_handlers_from_registry to prevent handler instantiation
        async def noop_populate() -> None:
            pass

        with patch.object(process, "_populate_handlers_from_registry", noop_populate):
            with patch.object(process, "_handlers", {"simple": SimpleHandler()}):
                await process.start()

                try:
                    health = await process.health_check()

                    # Simple handler should be reported as healthy
                    assert "simple" in health["handlers"]
                    assert health["handlers"]["simple"]["healthy"] is True
                    # Should have note about no health_check method
                    assert "note" in health["handlers"]["simple"]
                    assert (
                        "no health_check method" in health["handlers"]["simple"]["note"]
                    )
                    # Overall health should be True
                    assert health["healthy"] is True
                finally:
                    await process.stop()

    @pytest.mark.asyncio
    async def test_health_check_multiple_handlers_mixed_health(self) -> None:
        """Test health check with multiple handlers of mixed health status.

        When multiple handlers are registered with different health statuses,
        the overall health should be False if any handler is unhealthy.
        """

        process = RuntimeHostProcess()

        healthy_handler = MockHandler(handler_type="healthy")
        healthy_handler.initialized = True

        unhealthy_handler = MockHandler(handler_type="unhealthy")
        unhealthy_handler.initialized = False

        handlers = {
            "healthy": healthy_handler,
            "unhealthy": unhealthy_handler,
        }

        # Patch _populate_handlers_from_registry to prevent handler instantiation
        async def noop_populate() -> None:
            pass

        with patch.object(process, "_populate_handlers_from_registry", noop_populate):
            with patch.object(process, "_handlers", handlers):
                await process.start()

                try:
                    health = await process.health_check()

                    # Both handlers should be reported
                    assert "healthy" in health["handlers"]
                    assert "unhealthy" in health["handlers"]
                    # Healthy handler should report healthy
                    assert health["handlers"]["healthy"]["healthy"] is True
                    # Unhealthy handler should report unhealthy
                    assert health["handlers"]["unhealthy"]["healthy"] is False
                    # Overall health should be False (due to unhealthy handler)
                    assert health["healthy"] is False
                finally:
                    await process.stop()


# =============================================================================
# TestRuntimeHostProcessIntegration
# =============================================================================


class TestRuntimeHostProcessIntegration:
    """Integration tests for RuntimeHostProcess with real event bus."""

    @pytest.mark.asyncio
    async def test_full_request_response_cycle(
        self,
        mock_handler: MockHandler,
    ) -> None:
        """Test complete request/response cycle through event bus.

        This test verifies the full flow:
        1. Process starts and subscribes to input topic
        2. Message published to input topic
        3. Handler receives and processes envelope
        4. Response published to output topic
        """

        process = RuntimeHostProcess(
            config={
                "input_topic": "test.input",
                "output_topic": "test.output",
            }
        )

        with patch.object(process, "_handlers", {"http": mock_handler}):
            await process.start()

            try:
                # Simulate receiving an envelope through the event bus
                envelope = {
                    "operation": "http.get",
                    "payload": {"url": "https://example.com/api"},
                    "correlation_id": uuid4(),
                    "handler_type": "http",
                }

                # Process the envelope
                await process._handle_envelope(envelope)

                # Verify handler was called
                assert len(mock_handler.calls) == 1
                assert mock_handler.calls[0]["operation"] == "http.get"
            finally:
                await process.stop()

    @pytest.mark.asyncio
    async def test_multiple_handlers_registered(self) -> None:
        """Test that multiple handlers can be registered and used."""

        process = RuntimeHostProcess()

        http_handler = MockHandler(handler_type="http")
        db_handler = MockHandler(handler_type="db")

        handlers = {
            "http": http_handler,
            "db": db_handler,
        }

        with patch.object(process, "_handlers", handlers):
            await process.start()

            try:
                # Send to HTTP handler
                await process._handle_envelope(
                    {
                        "operation": "http.get",
                        "payload": {},
                        "correlation_id": uuid4(),
                        "handler_type": "http",
                    }
                )

                # Send to DB handler
                await process._handle_envelope(
                    {
                        "operation": "db.query",
                        "payload": {},
                        "correlation_id": uuid4(),
                        "handler_type": "db",
                    }
                )

                assert len(http_handler.calls) == 1
                assert len(db_handler.calls) == 1
            finally:
                await process.stop()


# =============================================================================
# TestRuntimeHostProcessDeterministic
# =============================================================================


class TestRuntimeHostProcessDeterministic:
    """Tests using deterministic utilities for reproducible behavior."""

    @pytest.mark.asyncio
    async def test_deterministic_correlation_id_tracking(
        self,
        deterministic_id_gen: DeterministicIdGenerator,
        mock_handler: MockHandler,
    ) -> None:
        """Test correlation ID handling with deterministic IDs."""

        process = RuntimeHostProcess()

        # Generate predictable correlation IDs
        correlation_ids = [deterministic_id_gen.next_uuid() for _ in range(3)]

        # Verify they are deterministic
        assert correlation_ids[0].int == 101
        assert correlation_ids[1].int == 102
        assert correlation_ids[2].int == 103

        with patch.object(process, "_handlers", {"http": mock_handler}):
            await process.start()

            try:
                for i, corr_id in enumerate(correlation_ids):
                    await process._handle_envelope(
                        {
                            "operation": "http.get",
                            "payload": {"index": i},
                            "correlation_id": corr_id,
                            "handler_type": "http",
                        }
                    )

                # Verify all were processed with correct correlation IDs
                assert len(mock_handler.calls) == 3
                for i, call in enumerate(mock_handler.calls):
                    assert call["correlation_id"] == correlation_ids[i]
            finally:
                await process.stop()


# =============================================================================
# TestRuntimeHostProcessLogWarnings
# =============================================================================


class TestRuntimeHostProcessLogWarnings:
    """Test log warning assertions (following OMN-252 patterns)."""

    RUNTIME_MODULE = "omnibase_infra.runtime.runtime_host_process"

    @pytest.mark.asyncio
    async def test_no_unexpected_warnings_during_normal_operation(
        self,
        mock_handler: MockHandler,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test that normal operations produce no unexpected warnings."""
        import logging

        from tests.helpers import filter_handler_warnings

        process = RuntimeHostProcess()

        with caplog.at_level(logging.WARNING):
            # Patch _populate_handlers_from_registry to prevent it from trying to
            # instantiate handlers from the singleton registry (which may have handlers
            # registered from previous tests, and would fail without proper config)
            async def noop_populate() -> None:
                pass

            with patch.object(
                process, "_populate_handlers_from_registry", noop_populate
            ):
                with patch.object(process, "_handlers", {"http": mock_handler}):
                    await process.start()

                    try:
                        # Normal operation - process an envelope
                        await process._handle_envelope(
                            {
                                "operation": "http.get",
                                "payload": {"url": "https://example.com"},
                                "correlation_id": uuid4(),
                                "handler_type": "http",
                            }
                        )
                    finally:
                        await process.stop()

        # Filter for warnings from our module
        runtime_warnings = filter_handler_warnings(caplog.records, self.RUNTIME_MODULE)
        assert (
            len(runtime_warnings) == 0
        ), f"Unexpected warnings: {[w.message for w in runtime_warnings]}"


# =============================================================================
# Shutdown Priority Tests
# =============================================================================


class MockHandlerWithPriority(MockHandler):
    """Mock handler with configurable shutdown priority."""

    def __init__(
        self,
        handler_type: str = "mock",
        priority: int = 0,
    ) -> None:
        """Initialize mock handler with priority.

        Args:
            handler_type: The type identifier for this handler.
            priority: Shutdown priority (higher = shutdown first).
        """
        super().__init__(handler_type=handler_type)
        self._priority = priority
        self.shutdown_order: int | None = None  # Track when shutdown was called

    def shutdown_priority(self) -> int:
        """Return shutdown priority."""
        return self._priority


class MockHandlerWithInvalidPriority:
    """Mock handler with invalid shutdown_priority return type."""

    def __init__(self, handler_type: str = "invalid_priority") -> None:
        """Initialize mock handler."""
        self.handler_type = handler_type
        self.shutdown_called = False

    def shutdown_priority(self) -> str:  # type: ignore[return-value]
        """Return invalid priority type."""
        return "not_an_int"  # type: ignore[return-value]

    async def shutdown(self) -> None:
        """Shutdown the mock handler."""
        self.shutdown_called = True


class MockHandlerWithFailingPriority:
    """Mock handler where shutdown_priority raises an exception."""

    def __init__(self, handler_type: str = "failing_priority") -> None:
        """Initialize mock handler."""
        self.handler_type = handler_type
        self.shutdown_called = False

    def shutdown_priority(self) -> int:
        """Raise exception when getting priority."""
        raise RuntimeError("Priority check failed")

    async def shutdown(self) -> None:
        """Shutdown the mock handler."""
        self.shutdown_called = True


class TestRuntimeHostProcessShutdownPriority:
    """Tests for shutdown priority-based handler ordering."""

    @pytest.mark.asyncio
    async def test_get_shutdown_priority_returns_default_for_handler_without_method(
        self,
    ) -> None:
        """Test that get_shutdown_priority returns 0 for handlers without the method."""
        handler = MockHandler(handler_type="no_priority")

        # Use ProtocolLifecycleExecutor static method directly
        priority = ProtocolLifecycleExecutor.get_shutdown_priority(handler)  # type: ignore[arg-type]

        assert priority == 0

    @pytest.mark.asyncio
    async def test_get_shutdown_priority_returns_handler_priority(self) -> None:
        """Test that get_shutdown_priority returns the handler's priority value."""
        handler = MockHandlerWithPriority(handler_type="with_priority", priority=100)

        # Use ProtocolLifecycleExecutor static method directly
        priority = ProtocolLifecycleExecutor.get_shutdown_priority(handler)  # type: ignore[arg-type]

        assert priority == 100

    @pytest.mark.asyncio
    async def test_get_shutdown_priority_returns_default_for_invalid_return_type(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test that get_shutdown_priority returns 0 when handler returns non-int."""
        import logging

        handler = MockHandlerWithInvalidPriority()

        with caplog.at_level(logging.WARNING):
            # Use ProtocolLifecycleExecutor static method directly
            priority = ProtocolLifecycleExecutor.get_shutdown_priority(handler)  # type: ignore[arg-type]

        assert priority == 0
        assert any(
            "shutdown_priority() returned non-int" in r.message for r in caplog.records
        )

    @pytest.mark.asyncio
    async def test_get_shutdown_priority_returns_default_when_method_raises(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test that get_shutdown_priority returns 0 when handler raises exception."""
        import logging

        handler = MockHandlerWithFailingPriority()

        with caplog.at_level(logging.WARNING):
            # Use ProtocolLifecycleExecutor static method directly
            priority = ProtocolLifecycleExecutor.get_shutdown_priority(handler)  # type: ignore[arg-type]

        assert priority == 0
        assert any(
            "Error calling handler shutdown_priority()" in r.message
            for r in caplog.records
        )

    @pytest.mark.asyncio
    async def test_stop_shuts_down_higher_priority_handlers_first(self) -> None:
        """Test that handlers with higher priority are shutdown before lower priority."""
        process = RuntimeHostProcess()

        # Track shutdown order globally
        shutdown_order: list[str] = []

        # Create handlers with different priorities
        consumer = MockHandlerWithPriority(handler_type="consumer", priority=100)
        producer = MockHandlerWithPriority(handler_type="producer", priority=50)
        pool = MockHandlerWithPriority(handler_type="pool", priority=0)

        # Override shutdown to track order
        original_consumer_shutdown = consumer.shutdown

        async def consumer_shutdown() -> None:
            shutdown_order.append("consumer")
            await original_consumer_shutdown()

        consumer.shutdown = consumer_shutdown  # type: ignore[method-assign]

        original_producer_shutdown = producer.shutdown

        async def producer_shutdown() -> None:
            shutdown_order.append("producer")
            await original_producer_shutdown()

        producer.shutdown = producer_shutdown  # type: ignore[method-assign]

        original_pool_shutdown = pool.shutdown

        async def pool_shutdown() -> None:
            shutdown_order.append("pool")
            await original_pool_shutdown()

        pool.shutdown = pool_shutdown  # type: ignore[method-assign]

        # Patch to prevent auto-population
        async def noop_populate() -> None:
            pass

        with patch.object(process, "_populate_handlers_from_registry", noop_populate):
            with patch.object(
                process,
                "_handlers",
                {"consumer": consumer, "producer": producer, "pool": pool},
            ):
                await process.start()
                await process.stop()

        # Verify shutdown order: consumer (100) -> producer (50) -> pool (0)
        assert shutdown_order == ["consumer", "producer", "pool"]
        assert consumer.shutdown_called is True
        assert producer.shutdown_called is True
        assert pool.shutdown_called is True

    @pytest.mark.asyncio
    async def test_stop_shuts_down_same_priority_handlers_in_parallel(self) -> None:
        """Test that handlers with same priority are shutdown in parallel."""
        process = RuntimeHostProcess()

        # Create multiple handlers with same priority
        handler_a = MockHandlerWithPriority(handler_type="handler_a", priority=50)
        handler_b = MockHandlerWithPriority(handler_type="handler_b", priority=50)

        # Track concurrent execution
        execution_times: dict[str, tuple[float, float]] = {}
        import time

        async def make_timed_shutdown(
            handler: MockHandlerWithPriority, name: str
        ) -> AsyncMock:
            original = handler.shutdown

            async def timed_shutdown() -> None:
                start = time.monotonic()
                await asyncio.sleep(0.05)  # Small delay to observe parallelism
                await original()
                end = time.monotonic()
                execution_times[name] = (start, end)

            return timed_shutdown  # type: ignore[return-value]

        handler_a.shutdown = await make_timed_shutdown(handler_a, "handler_a")  # type: ignore[method-assign]
        handler_b.shutdown = await make_timed_shutdown(handler_b, "handler_b")  # type: ignore[method-assign]

        async def noop_populate() -> None:
            pass

        with patch.object(process, "_populate_handlers_from_registry", noop_populate):
            with patch.object(
                process,
                "_handlers",
                {"handler_a": handler_a, "handler_b": handler_b},
            ):
                await process.start()

                start_time = time.monotonic()
                await process.stop()
                total_time = time.monotonic() - start_time

        # If run in parallel, total time should be close to single handler time (~0.05s)
        # If sequential, it would be ~0.1s
        # Allow some margin for test overhead
        assert (
            total_time < 0.15
        ), f"Parallel shutdown took too long: {total_time}s (expected < 0.15s)"

        # Verify both were called
        assert handler_a.shutdown_called is True
        assert handler_b.shutdown_called is True

    @pytest.mark.asyncio
    async def test_stop_handles_mixed_priority_handlers(self) -> None:
        """Test stop with handlers that have and don't have shutdown_priority."""
        process = RuntimeHostProcess()

        shutdown_order: list[str] = []

        # Handler with priority
        high_priority = MockHandlerWithPriority(
            handler_type="high_priority", priority=100
        )

        # Handler without priority method (default 0)
        no_priority = MockHandler(handler_type="no_priority")

        async def high_shutdown() -> None:
            shutdown_order.append("high_priority")
            high_priority.shutdown_called = True

        high_priority.shutdown = high_shutdown  # type: ignore[method-assign]

        async def no_priority_shutdown() -> None:
            shutdown_order.append("no_priority")
            no_priority.shutdown_called = True

        no_priority.shutdown = no_priority_shutdown  # type: ignore[method-assign]

        async def noop_populate() -> None:
            pass

        with patch.object(process, "_populate_handlers_from_registry", noop_populate):
            with patch.object(
                process,
                "_handlers",
                {"high_priority": high_priority, "no_priority": no_priority},
            ):
                await process.start()
                await process.stop()

        # High priority (100) should shutdown before no priority (0)
        assert shutdown_order == ["high_priority", "no_priority"]

    @pytest.mark.asyncio
    async def test_stop_logs_priority_groups(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test that stop logs priority group information."""
        import logging

        process = RuntimeHostProcess()

        handler_a = MockHandlerWithPriority(handler_type="handler_a", priority=100)
        handler_b = MockHandlerWithPriority(handler_type="handler_b", priority=50)

        async def noop_populate() -> None:
            pass

        with patch.object(process, "_populate_handlers_from_registry", noop_populate):
            with patch.object(
                process,
                "_handlers",
                {"handler_a": handler_a, "handler_b": handler_b},
            ):
                await process.start()

                with caplog.at_level(logging.INFO):
                    await process.stop()

        # Check that priority-based shutdown message was logged
        assert any(
            "Priority-based handler shutdown completed" in r.message
            for r in caplog.records
        )

    @pytest.mark.asyncio
    async def test_stop_continues_with_next_priority_group_on_failure(self) -> None:
        """Test that failure in one priority group doesn't prevent shutdown of next."""
        process = RuntimeHostProcess()

        shutdown_order: list[str] = []

        # High priority handler that fails
        failing_handler = MockHandlerWithPriority(handler_type="failing", priority=100)

        async def failing_shutdown() -> None:
            shutdown_order.append("failing")
            raise RuntimeError("Shutdown failed")

        failing_handler.shutdown = failing_shutdown  # type: ignore[method-assign]

        # Lower priority handler that should still be shutdown
        normal_handler = MockHandlerWithPriority(handler_type="normal", priority=50)

        async def normal_shutdown() -> None:
            shutdown_order.append("normal")
            normal_handler.shutdown_called = True

        normal_handler.shutdown = normal_shutdown  # type: ignore[method-assign]

        async def noop_populate() -> None:
            pass

        with patch.object(process, "_populate_handlers_from_registry", noop_populate):
            with patch.object(
                process,
                "_handlers",
                {"failing": failing_handler, "normal": normal_handler},
            ):
                await process.start()
                await process.stop()

        # Both should have been attempted, regardless of failure
        assert "failing" in shutdown_order
        assert "normal" in shutdown_order
        assert shutdown_order.index("failing") < shutdown_order.index("normal")
        assert normal_handler.shutdown_called is True


# =============================================================================
# Module Exports
# =============================================================================


__all__: list[str] = [
    "TestRuntimeHostProcessInitialization",
    "TestRuntimeHostProcessTimeoutValidation",
    "TestRuntimeHostProcessLifecycle",
    "TestRuntimeHostProcessEnvelopeRouting",
    "TestRuntimeHostProcessErrorHandling",
    "TestRuntimeHostProcessHealthCheck",
    "TestRuntimeHostProcessIntegration",
    "TestRuntimeHostProcessDeterministic",
    "TestRuntimeHostProcessLogWarnings",
    "TestRuntimeHostProcessShutdownPriority",
    "MockHandler",
    "MockFailingHandler",
    "MockEventBus",
    "MockHandlerWithPriority",
    "MockHandlerWithInvalidPriority",
    "MockHandlerWithFailingPriority",
]
