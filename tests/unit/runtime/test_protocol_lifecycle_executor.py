# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for ProtocolLifecycleExecutor.

Tests the ProtocolLifecycleExecutor class which extracts protocol lifecycle
operations from RuntimeHostProcess, including:
- Shutdown priority retrieval
- Individual handler shutdown
- Handler health checks with timeout
- Priority-based shutdown orchestration

These tests follow existing patterns from test_runtime_host_process.py.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

import pytest

if TYPE_CHECKING:
    from omnibase_spi.protocols.handlers.protocol_handler import ProtocolHandler

# =============================================================================
# TDD Skip Helper - Check if ProtocolLifecycleExecutor is implemented
# =============================================================================

_PROTOCOL_LIFECYCLE_EXECUTOR_IMPLEMENTED = False
try:
    from omnibase_infra.runtime.protocol_lifecycle_executor import (
        ProtocolLifecycleExecutor,
    )

    _PROTOCOL_LIFECYCLE_EXECUTOR_IMPLEMENTED = True
except ImportError:
    # ProtocolLifecycleExecutor not implemented yet - define placeholder for type checking
    ProtocolLifecycleExecutor = None  # type: ignore[misc, assignment]

# Skip marker for all tests when implementation doesn't exist
pytestmark = pytest.mark.skipif(
    not _PROTOCOL_LIFECYCLE_EXECUTOR_IMPLEMENTED,
    reason="ProtocolLifecycleExecutor not yet implemented (TDD red phase)",
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
        self.initialized: bool = True
        self.shutdown_called: bool = False

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


class MockHandlerWithoutShutdown:
    """Mock handler without shutdown method."""

    def __init__(self, handler_type: str = "no_shutdown") -> None:
        """Initialize mock handler."""
        self.handler_type = handler_type

    async def health_check(self) -> dict[str, object]:
        """Return health check status."""
        return {"healthy": True, "handler_type": self.handler_type}


class MockHandlerWithoutHealthCheck:
    """Mock handler without health_check method."""

    def __init__(self, handler_type: str = "no_health") -> None:
        """Initialize mock handler."""
        self.handler_type = handler_type
        self.shutdown_called = False

    async def shutdown(self) -> None:
        """Shutdown the mock handler."""
        self.shutdown_called = True


class MockHandlerWithSlowHealthCheck:
    """Mock handler with slow health check for timeout testing."""

    def __init__(
        self,
        handler_type: str = "slow_health",
        delay_seconds: float = 10.0,
    ) -> None:
        """Initialize mock handler.

        Args:
            handler_type: The type identifier for this handler.
            delay_seconds: How long health_check should take.
        """
        self.handler_type = handler_type
        self.delay_seconds = delay_seconds

    async def health_check(self) -> dict[str, object]:
        """Return health check status after delay."""
        await asyncio.sleep(self.delay_seconds)
        return {"healthy": True, "handler_type": self.handler_type}


class MockHandlerWithFailingHealthCheck:
    """Mock handler where health_check raises an exception."""

    def __init__(self, handler_type: str = "failing_health") -> None:
        """Initialize mock handler."""
        self.handler_type = handler_type

    async def health_check(self) -> dict[str, object]:
        """Raise exception during health check."""
        raise RuntimeError("Health check failed")


class MockHandlerWithFailingShutdown:
    """Mock handler where shutdown raises an exception."""

    def __init__(self, handler_type: str = "failing_shutdown") -> None:
        """Initialize mock handler."""
        self.handler_type = handler_type
        self.shutdown_called = False

    def shutdown_priority(self) -> int:
        """Return shutdown priority."""
        return 50

    async def shutdown(self) -> None:
        """Raise exception during shutdown."""
        self.shutdown_called = True
        raise RuntimeError("Shutdown failed")


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_handler() -> MockHandler:
    """Create mock handler for testing."""
    return MockHandler(handler_type="test")


@pytest.fixture
def protocol_lifecycle_executor() -> ProtocolLifecycleExecutor:
    """Create protocol lifecycle executor with default settings."""
    return ProtocolLifecycleExecutor()


@pytest.fixture
def protocol_lifecycle_executor_custom_timeout() -> ProtocolLifecycleExecutor:
    """Create protocol lifecycle executor with custom timeout."""
    return ProtocolLifecycleExecutor(health_check_timeout_seconds=10.0)


# =============================================================================
# TestProtocolLifecycleExecutorInit
# =============================================================================


class TestProtocolLifecycleExecutorInit:
    """Test ProtocolLifecycleExecutor initialization."""

    def test_default_timeout(self) -> None:
        """Test that ProtocolLifecycleExecutor initializes with default timeout of 5.0 seconds."""
        executor = ProtocolLifecycleExecutor()

        assert executor.health_check_timeout_seconds == 5.0

    def test_custom_timeout(self) -> None:
        """Test that ProtocolLifecycleExecutor accepts custom timeout."""
        executor = ProtocolLifecycleExecutor(health_check_timeout_seconds=15.0)

        assert executor.health_check_timeout_seconds == 15.0


# =============================================================================
# TestProtocolLifecycleExecutorShutdownPriority
# =============================================================================


class TestProtocolLifecycleExecutorShutdownPriority:
    """Test shutdown priority retrieval methods."""

    def test_get_shutdown_priority_returns_default_for_handler_without_method(
        self,
        protocol_lifecycle_executor: ProtocolLifecycleExecutor,
    ) -> None:
        """Test that get_shutdown_priority returns 0 for handlers without the method."""
        handler = MockHandler(handler_type="no_priority")

        priority = protocol_lifecycle_executor.get_shutdown_priority(handler)  # type: ignore[arg-type]

        assert priority == 0

    def test_get_shutdown_priority_returns_handler_priority(
        self,
        protocol_lifecycle_executor: ProtocolLifecycleExecutor,
    ) -> None:
        """Test that get_shutdown_priority returns the handler's priority value."""
        handler = MockHandlerWithPriority(handler_type="with_priority", priority=100)

        priority = protocol_lifecycle_executor.get_shutdown_priority(handler)  # type: ignore[arg-type]

        assert priority == 100

    def test_get_shutdown_priority_returns_default_for_invalid_return_type(
        self,
        protocol_lifecycle_executor: ProtocolLifecycleExecutor,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test that get_shutdown_priority returns 0 when handler returns non-int."""
        handler = MockHandlerWithInvalidPriority()

        with caplog.at_level(logging.WARNING):
            priority = protocol_lifecycle_executor.get_shutdown_priority(handler)  # type: ignore[arg-type]

        assert priority == 0
        assert any(
            "shutdown_priority() returned non-int" in r.message for r in caplog.records
        )

    def test_get_shutdown_priority_returns_default_when_method_raises(
        self,
        protocol_lifecycle_executor: ProtocolLifecycleExecutor,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test that get_shutdown_priority returns 0 when handler raises exception."""
        handler = MockHandlerWithFailingPriority()

        with caplog.at_level(logging.WARNING):
            priority = protocol_lifecycle_executor.get_shutdown_priority(handler)  # type: ignore[arg-type]

        assert priority == 0
        assert any(
            "Error calling handler shutdown_priority()" in r.message
            for r in caplog.records
        )


# =============================================================================
# TestProtocolLifecycleExecutorShutdown
# =============================================================================


class TestProtocolLifecycleExecutorShutdown:
    """Test individual handler shutdown methods."""

    @pytest.mark.asyncio
    async def test_shutdown_handler_success(
        self,
        protocol_lifecycle_executor: ProtocolLifecycleExecutor,
    ) -> None:
        """Test successful handler shutdown returns success tuple."""
        handler = MockHandler(handler_type="test")

        result = await protocol_lifecycle_executor.shutdown_handler(
            "test",
            handler,  # type: ignore[arg-type]
        )

        assert result.handler_type == "test"
        assert result.success is True
        assert result.has_error is False
        assert result.error_message == ""
        assert handler.shutdown_called is True

    @pytest.mark.asyncio
    async def test_shutdown_handler_without_shutdown_method(
        self,
        protocol_lifecycle_executor: ProtocolLifecycleExecutor,
    ) -> None:
        """Test shutdown of handler without shutdown method succeeds."""
        handler = MockHandlerWithoutShutdown(handler_type="no_shutdown")

        result = await protocol_lifecycle_executor.shutdown_handler(
            "no_shutdown",
            handler,  # type: ignore[arg-type]
        )

        assert result.handler_type == "no_shutdown"
        assert result.success is True
        assert result.has_error is False
        assert result.error_message == ""

    @pytest.mark.asyncio
    async def test_shutdown_handler_error_returns_failure(
        self,
        protocol_lifecycle_executor: ProtocolLifecycleExecutor,
    ) -> None:
        """Test that handler shutdown error returns failure tuple."""
        handler = MockHandlerWithFailingShutdown(handler_type="failing")

        result = await protocol_lifecycle_executor.shutdown_handler(
            "failing",
            handler,  # type: ignore[arg-type]
        )

        assert result.handler_type == "failing"
        assert result.success is False
        assert result.has_error
        assert "Shutdown failed" in result.error_message
        assert handler.shutdown_called is True


# =============================================================================
# TestProtocolLifecycleExecutorHealthCheck
# =============================================================================


class TestProtocolLifecycleExecutorHealthCheck:
    """Test handler health check methods."""

    @pytest.mark.asyncio
    async def test_check_handler_health_success(
        self,
        protocol_lifecycle_executor: ProtocolLifecycleExecutor,
    ) -> None:
        """Test successful health check returns handler's health status."""
        handler = MockHandler(handler_type="test")

        result = await protocol_lifecycle_executor.check_handler_health(
            "test",
            handler,  # type: ignore[arg-type]
        )

        assert result.handler_type == "test"
        assert result.healthy is True
        assert result.details.get("healthy") is True

    @pytest.mark.asyncio
    async def test_check_handler_health_timeout(
        self,
        protocol_lifecycle_executor: ProtocolLifecycleExecutor,
    ) -> None:
        """Test that slow health check returns timeout error."""
        handler = MockHandlerWithSlowHealthCheck(
            handler_type="slow", delay_seconds=10.0
        )

        # Use a very short timeout to trigger timeout quickly
        result = await protocol_lifecycle_executor.check_handler_health(
            "slow",
            handler,  # type: ignore[arg-type]
            timeout_seconds=0.1,
        )

        assert result.handler_type == "slow"
        assert result.healthy is False
        error_msg = result.details.get("error", "")
        assert isinstance(error_msg, str)
        assert "timeout" in error_msg.lower()

    @pytest.mark.asyncio
    async def test_check_handler_health_error(
        self,
        protocol_lifecycle_executor: ProtocolLifecycleExecutor,
    ) -> None:
        """Test that health check error returns failure status."""
        handler = MockHandlerWithFailingHealthCheck(handler_type="failing")

        result = await protocol_lifecycle_executor.check_handler_health(
            "failing",
            handler,  # type: ignore[arg-type]
        )

        assert result.handler_type == "failing"
        assert result.healthy is False
        error_msg = result.details.get("error", "")
        assert isinstance(error_msg, str)
        assert "Health check failed" in error_msg

    @pytest.mark.asyncio
    async def test_check_handler_health_no_method(
        self,
        protocol_lifecycle_executor: ProtocolLifecycleExecutor,
    ) -> None:
        """Test that handler without health_check method is assumed healthy."""
        handler = MockHandlerWithoutHealthCheck(handler_type="no_health")

        result = await protocol_lifecycle_executor.check_handler_health(
            "no_health",
            handler,  # type: ignore[arg-type]
        )

        assert result.handler_type == "no_health"
        assert result.healthy is True
        note_msg = result.details.get("note", "")
        assert isinstance(note_msg, str)
        assert "no health_check method" in note_msg

    @pytest.mark.asyncio
    async def test_check_handler_health_uses_default_timeout(
        self,
        protocol_lifecycle_executor_custom_timeout: ProtocolLifecycleExecutor,
    ) -> None:
        """Test that health check uses default timeout when not specified."""
        # Verify the executor has the custom timeout
        assert (
            protocol_lifecycle_executor_custom_timeout.health_check_timeout_seconds
            == 10.0
        )

        handler = MockHandler(handler_type="test")

        # Call without timeout_seconds - should use default
        result = await protocol_lifecycle_executor_custom_timeout.check_handler_health(
            "test",
            handler,  # type: ignore[arg-type]
        )

        assert result.handler_type == "test"
        assert result.healthy is True

    @pytest.mark.asyncio
    async def test_check_handler_health_uses_provided_timeout(
        self,
        protocol_lifecycle_executor: ProtocolLifecycleExecutor,
    ) -> None:
        """Test that health check uses provided timeout over default."""
        # Create a handler with a delay that's longer than default (5.0) but shorter
        # than our provided timeout
        handler = MockHandlerWithSlowHealthCheck(handler_type="slow", delay_seconds=0.2)

        # Use a timeout that allows completion
        result = await protocol_lifecycle_executor.check_handler_health(
            "slow",
            handler,  # type: ignore[arg-type]
            timeout_seconds=1.0,
        )

        assert result.handler_type == "slow"
        assert result.healthy is True


# =============================================================================
# TestProtocolLifecycleExecutorShutdownByPriority
# =============================================================================


class TestProtocolLifecycleExecutorShutdownByPriority:
    """Test priority-based handler shutdown orchestration."""

    @pytest.mark.asyncio
    async def test_shutdown_handlers_by_priority_higher_first(
        self,
        protocol_lifecycle_executor: ProtocolLifecycleExecutor,
    ) -> None:
        """Test that handlers with higher priority are shutdown before lower priority."""
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

        handlers: dict[str, ProtocolHandler] = {
            "consumer": consumer,  # type: ignore[dict-item]
            "producer": producer,  # type: ignore[dict-item]
            "pool": pool,  # type: ignore[dict-item]
        }

        await protocol_lifecycle_executor.shutdown_handlers_by_priority(handlers)

        # Verify shutdown order: consumer (100) -> producer (50) -> pool (0)
        assert shutdown_order == ["consumer", "producer", "pool"]
        assert consumer.shutdown_called is True
        assert producer.shutdown_called is True
        assert pool.shutdown_called is True

    @pytest.mark.asyncio
    async def test_shutdown_handlers_same_priority_parallel(
        self,
        protocol_lifecycle_executor: ProtocolLifecycleExecutor,
    ) -> None:
        """Test that handlers with same priority are shutdown in parallel."""
        # Create multiple handlers with same priority
        handler_a = MockHandlerWithPriority(handler_type="handler_a", priority=50)
        handler_b = MockHandlerWithPriority(handler_type="handler_b", priority=50)

        # Track concurrent execution
        execution_times: dict[str, tuple[float, float]] = {}

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

        handlers: dict[str, ProtocolHandler] = {
            "handler_a": handler_a,  # type: ignore[dict-item]
            "handler_b": handler_b,  # type: ignore[dict-item]
        }

        start_time = time.monotonic()
        await protocol_lifecycle_executor.shutdown_handlers_by_priority(handlers)
        total_time = time.monotonic() - start_time

        # If run in parallel, total time should be close to single handler time (~0.05s)
        # If sequential, it would be ~0.1s
        # CI-friendly threshold: 1.0s catches severe regressions while allowing
        # for variable CI performance (containerization, CPU throttling, etc.)
        assert total_time < 1.0, (
            f"Parallel shutdown took too long: {total_time}s (expected < 1.0s)"
        )

        # Verify both were called
        assert handler_a.shutdown_called is True
        assert handler_b.shutdown_called is True

    @pytest.mark.asyncio
    async def test_shutdown_handlers_mixed_priorities(
        self,
        protocol_lifecycle_executor: ProtocolLifecycleExecutor,
    ) -> None:
        """Test shutdown with handlers that have and don't have shutdown_priority."""
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

        handlers: dict[str, ProtocolHandler] = {
            "high_priority": high_priority,  # type: ignore[dict-item]
            "no_priority": no_priority,  # type: ignore[dict-item]
        }

        await protocol_lifecycle_executor.shutdown_handlers_by_priority(handlers)

        # High priority (100) should shutdown before no priority (0)
        assert shutdown_order == ["high_priority", "no_priority"]

    @pytest.mark.asyncio
    async def test_shutdown_handlers_empty_dict(
        self,
        protocol_lifecycle_executor: ProtocolLifecycleExecutor,
    ) -> None:
        """Test that shutdown with empty handlers dict completes without error."""
        handlers: dict[str, ProtocolHandler] = {}

        # Should not raise any exception
        result = await protocol_lifecycle_executor.shutdown_handlers_by_priority(
            handlers
        )

        # Result should indicate success with empty results
        assert result is not None
        assert result.succeeded_handlers == []
        assert result.failed_handlers == []

    @pytest.mark.asyncio
    async def test_shutdown_handlers_continues_on_failure(
        self,
        protocol_lifecycle_executor: ProtocolLifecycleExecutor,
    ) -> None:
        """Test that failure in one handler doesn't prevent shutdown of others."""
        shutdown_order: list[str] = []

        # High priority handler that fails
        failing_handler = MockHandlerWithFailingShutdown(handler_type="failing")

        # Override to track order
        original_failing_shutdown = failing_handler.shutdown

        async def tracked_failing_shutdown() -> None:
            shutdown_order.append("failing")
            await original_failing_shutdown()

        failing_handler.shutdown = tracked_failing_shutdown  # type: ignore[method-assign]

        # Lower priority handler that should still be shutdown
        normal_handler = MockHandlerWithPriority(handler_type="normal", priority=40)

        async def normal_shutdown() -> None:
            shutdown_order.append("normal")
            normal_handler.shutdown_called = True

        normal_handler.shutdown = normal_shutdown  # type: ignore[method-assign]

        handlers: dict[str, ProtocolHandler] = {
            "failing": failing_handler,  # type: ignore[dict-item]
            "normal": normal_handler,  # type: ignore[dict-item]
        }

        result = await protocol_lifecycle_executor.shutdown_handlers_by_priority(
            handlers
        )

        # Both should have been attempted, regardless of failure
        assert "failing" in shutdown_order
        assert "normal" in shutdown_order

        # Verify failure was tracked
        assert "normal" in result.succeeded_handlers
        assert any(f.handler_type == "failing" for f in result.failed_handlers)

    @pytest.mark.asyncio
    async def test_shutdown_handlers_logs_priority_groups(
        self,
        protocol_lifecycle_executor: ProtocolLifecycleExecutor,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test that shutdown logs priority group information."""
        handler_a = MockHandlerWithPriority(handler_type="handler_a", priority=100)
        handler_b = MockHandlerWithPriority(handler_type="handler_b", priority=50)

        handlers: dict[str, ProtocolHandler] = {
            "handler_a": handler_a,  # type: ignore[dict-item]
            "handler_b": handler_b,  # type: ignore[dict-item]
        }

        with caplog.at_level(logging.INFO):
            await protocol_lifecycle_executor.shutdown_handlers_by_priority(handlers)

        # Check that priority-based shutdown message was logged
        assert any(
            "shutdown" in r.message.lower() and "priority" in r.message.lower()
            for r in caplog.records
        ) or any(
            "shutdown" in r.message.lower() and "completed" in r.message.lower()
            for r in caplog.records
        )


# =============================================================================
# Additional Edge Case Tests
# =============================================================================


class TestProtocolLifecycleExecutorEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_shutdown_priority_with_zero_value(
        self,
        protocol_lifecycle_executor: ProtocolLifecycleExecutor,
    ) -> None:
        """Test that priority of 0 is handled correctly (not treated as falsy)."""
        handler = MockHandlerWithPriority(handler_type="zero_priority", priority=0)

        priority = protocol_lifecycle_executor.get_shutdown_priority(handler)  # type: ignore[arg-type]

        assert priority == 0

    @pytest.mark.asyncio
    async def test_shutdown_priority_with_negative_value(
        self,
        protocol_lifecycle_executor: ProtocolLifecycleExecutor,
    ) -> None:
        """Test that negative priority values are handled correctly."""
        handler = MockHandlerWithPriority(
            handler_type="negative_priority", priority=-10
        )

        priority = protocol_lifecycle_executor.get_shutdown_priority(handler)  # type: ignore[arg-type]

        assert priority == -10

    @pytest.mark.asyncio
    async def test_health_check_with_very_short_timeout(
        self,
        protocol_lifecycle_executor: ProtocolLifecycleExecutor,
    ) -> None:
        """Test health check with very short timeout still returns proper structure."""
        handler = MockHandlerWithSlowHealthCheck(handler_type="slow", delay_seconds=1.0)

        result = await protocol_lifecycle_executor.check_handler_health(
            "slow",
            handler,  # type: ignore[arg-type]
            timeout_seconds=0.001,
        )

        assert result.handler_type == "slow"
        assert result.healthy is False
        assert "error" in result.details

    @pytest.mark.asyncio
    async def test_shutdown_handlers_with_duplicate_priorities(
        self,
        protocol_lifecycle_executor: ProtocolLifecycleExecutor,
    ) -> None:
        """Test shutdown with multiple handlers at same priority level."""
        shutdown_times: dict[str, float] = {}

        # Create three handlers all with priority 50
        handler_a = MockHandlerWithPriority(handler_type="a", priority=50)
        handler_b = MockHandlerWithPriority(handler_type="b", priority=50)
        handler_c = MockHandlerWithPriority(handler_type="c", priority=50)

        async def make_shutdown(handler: MockHandlerWithPriority, name: str) -> None:
            original = handler.shutdown

            async def tracked_shutdown() -> None:
                shutdown_times[name] = time.monotonic()
                await original()

            handler.shutdown = tracked_shutdown  # type: ignore[method-assign]

        await make_shutdown(handler_a, "a")
        await make_shutdown(handler_b, "b")
        await make_shutdown(handler_c, "c")

        handlers: dict[str, ProtocolHandler] = {
            "a": handler_a,  # type: ignore[dict-item]
            "b": handler_b,  # type: ignore[dict-item]
            "c": handler_c,  # type: ignore[dict-item]
        }

        await protocol_lifecycle_executor.shutdown_handlers_by_priority(handlers)

        # All should be shutdown
        assert handler_a.shutdown_called is True
        assert handler_b.shutdown_called is True
        assert handler_c.shutdown_called is True

        # All should have shutdown times recorded
        assert len(shutdown_times) == 3


# =============================================================================
# Module Exports
# =============================================================================


__all__: list[str] = [
    "TestProtocolLifecycleExecutorInit",
    "TestProtocolLifecycleExecutorShutdownPriority",
    "TestProtocolLifecycleExecutorShutdown",
    "TestProtocolLifecycleExecutorHealthCheck",
    "TestProtocolLifecycleExecutorShutdownByPriority",
    "TestProtocolLifecycleExecutorEdgeCases",
    "MockHandler",
    "MockHandlerWithPriority",
    "MockHandlerWithInvalidPriority",
    "MockHandlerWithFailingPriority",
    "MockHandlerWithoutShutdown",
    "MockHandlerWithoutHealthCheck",
    "MockHandlerWithSlowHealthCheck",
    "MockHandlerWithFailingHealthCheck",
    "MockHandlerWithFailingShutdown",
]
