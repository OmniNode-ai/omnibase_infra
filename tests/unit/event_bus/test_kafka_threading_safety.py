# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Threading safety tests for KafkaEventBus race condition fixes.

Test Isolation and Cleanup Patterns
====================================

This test suite demonstrates proper test isolation patterns for async tests:

1. **Cleanup Order**: Always call `await bus.close()` inside the patch context
   to ensure mocks are still active during cleanup. This prevents AttributeError
   during producer.stop() calls.

2. **Exception Handling**: Tests that simulate failures properly catch and
   suppress expected exceptions using try/except blocks with pass statements.

3. **Timing-Sensitive Tests**: Tests using asyncio.sleep() include documentation
   noting that timing may vary under system load. These tests verify thread
   safety, not strict timing behavior.

4. **Resource Management**: Each test creates its own bus instance and ensures
   cleanup within the test method, preventing resource leaks between tests.

5. **Idempotency Testing**: Tests like concurrent_close_operations verify that
   operations can be safely called multiple times without additional cleanup.

6. **Documentation**: Inline comments explain cleanup patterns and expected
   behaviors to aid future maintenance and parallel test execution.

These patterns ensure tests can run reliably in parallel (pytest-xdist) without
interference or resource contention.
"""

import asyncio
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest

from omnibase_infra.event_bus.kafka_event_bus import KafkaEventBus


class SimulatedProducerError(Exception):
    """Custom exception for simulating producer failures in tests."""


@pytest.mark.asyncio
class TestKafkaEventBusThreadingSafety:
    """Test suite for KafkaEventBus threading safety and race condition fixes."""

    async def test_concurrent_publish_operations_thread_safe(self) -> None:
        """Test that concurrent publish operations don't cause race conditions."""
        # Create event bus with mocked producer
        bus = KafkaEventBus.default()

        # Mock the producer
        mock_producer = AsyncMock()
        mock_future: asyncio.Future[Any] = asyncio.Future()
        mock_future.set_result(Mock(partition=0, offset=0))
        mock_producer.send.return_value = mock_future

        # Start the bus (this will fail to connect but we'll mock it)
        with patch(
            "omnibase_infra.event_bus.kafka_event_bus.AIOKafkaProducer"
        ) as MockProducer:
            MockProducer.return_value = mock_producer
            mock_producer.start = AsyncMock()

            await bus.start()

            # Now simulate concurrent publish operations from multiple "threads"
            async def publish_task(i: int) -> None:
                try:
                    await bus.publish(
                        topic="test-topic",
                        key=f"key-{i}".encode(),
                        value=f"value-{i}".encode(),
                    )
                except Exception:
                    pass  # Expected for some races

            # Launch 10 concurrent publish operations
            tasks = [publish_task(i) for i in range(10)]
            await asyncio.gather(*tasks, return_exceptions=True)

            # Verify producer was called (may be less than 10 due to some failures)
            assert mock_producer.send.called

            # Cleanup: Close bus within patch context to ensure proper cleanup order
            await bus.close()

    async def test_initialize_start_race_condition_fixed(self) -> None:
        """Test that initialize() doesn't race with start()."""
        bus = KafkaEventBus.default()

        with patch(
            "omnibase_infra.event_bus.kafka_event_bus.AIOKafkaProducer"
        ) as MockProducer:
            mock_producer = AsyncMock()
            mock_producer.start = AsyncMock()
            MockProducer.return_value = mock_producer

            # Simulate concurrent initialize and config updates
            async def init_task() -> None:
                try:
                    await bus.initialize(
                        {
                            "environment": "test-env",
                            "group": "test-group",
                        }
                    )
                except Exception:
                    pass

            async def update_task() -> None:
                # Try to read environment during initialization
                _ = bus.environment

            # Run both concurrently
            await asyncio.gather(init_task(), update_task(), return_exceptions=True)

            # Verify final state is consistent
            assert bus.environment == "test-env"
            assert bus.group == "test-group"

            # Cleanup: Close bus within patch context to ensure proper cleanup order
            await bus.close()

    async def test_producer_access_during_retry_thread_safe(self) -> None:
        """Test that producer field access during retry is thread-safe."""
        bus = KafkaEventBus.default()

        with patch(
            "omnibase_infra.event_bus.kafka_event_bus.AIOKafkaProducer"
        ) as MockProducer:
            mock_producer = AsyncMock()
            MockProducer.return_value = mock_producer

            # Simulate producer that succeeds on both attempts (no timeout)
            call_count = 0

            async def send_success(*args: object, **kwargs: object) -> Any:
                nonlocal call_count
                call_count += 1
                future: asyncio.Future[Any] = asyncio.Future()
                future.set_result(Mock(partition=0, offset=0))
                return future

            mock_producer.send = send_success
            mock_producer.start = AsyncMock()

            await bus.start()

            # Multiple concurrent publishes should all succeed with thread-safe producer access
            async def publish_task(i: int) -> None:
                await bus.publish(
                    topic="test-topic",
                    key=f"key-{i}".encode(),
                    value=f"value-{i}".encode(),
                )

            # Launch 5 concurrent publishes
            await asyncio.gather(*[publish_task(i) for i in range(5)])

            # Verify all publishes succeeded
            assert call_count == 5

            # Cleanup: Close bus within patch context to ensure proper cleanup order
            await bus.close()

    async def test_concurrent_close_operations_thread_safe(self) -> None:
        """Test that concurrent close operations don't cause race conditions.

        Note: This test verifies idempotent close behavior - calling close()
        multiple times concurrently should be safe and result in proper cleanup.
        """
        bus = KafkaEventBus.default()

        with patch(
            "omnibase_infra.event_bus.kafka_event_bus.AIOKafkaProducer"
        ) as MockProducer:
            mock_producer = AsyncMock()
            mock_producer.start = AsyncMock()
            mock_producer.stop = AsyncMock()
            MockProducer.return_value = mock_producer

            await bus.start()

            # Launch multiple concurrent close operations (tests idempotency)
            close_tasks = [bus.close() for _ in range(5)]
            await asyncio.gather(*close_tasks)

            # Verify bus is properly closed
            assert bus._shutdown is True
            assert bus._started is False
            # No additional cleanup needed - close() already called multiple times

    async def test_health_check_during_shutdown_thread_safe(self) -> None:
        """Test that health_check() during shutdown is thread-safe.

        Note: Uses sleep() for timing coordination which may be affected by
        system load. The test verifies thread-safe concurrent health checks
        during shutdown, not strict timing behavior.
        """
        bus = KafkaEventBus.default()

        with patch(
            "omnibase_infra.event_bus.kafka_event_bus.AIOKafkaProducer"
        ) as MockProducer:
            mock_producer = AsyncMock()
            mock_producer.start = AsyncMock()
            mock_producer.stop = AsyncMock()
            MockProducer.return_value = mock_producer

            await bus.start()

            # Start shutdown and health check concurrently
            async def health_task() -> None:
                """Background task that repeatedly calls health_check."""
                for _ in range(10):
                    await bus.health_check()
                    await asyncio.sleep(0.01)

            health = asyncio.create_task(health_task())
            # Allow some health checks to run before shutdown
            await asyncio.sleep(0.05)
            await bus.close()
            # Wait for all health checks to complete
            await health

            # Final health check should show bus is closed
            status = await bus.health_check()
            assert status["started"] is False
            # Cleanup complete - bus already closed within patch context

    async def test_circuit_breaker_concurrent_access_thread_safe(self) -> None:
        """Test that circuit breaker state is thread-safe under concurrent access.

        This test verifies thread-safe circuit breaker behavior under concurrent
        failure conditions. All operations properly catch and suppress exceptions.
        """
        bus = KafkaEventBus(
            config=None,
            bootstrap_servers="localhost:9092",
            circuit_breaker_threshold=3,
        )

        with patch(
            "omnibase_infra.event_bus.kafka_event_bus.AIOKafkaProducer"
        ) as MockProducer:
            mock_producer = AsyncMock()
            MockProducer.return_value = mock_producer

            # Simulate producer that always fails
            async def failing_send(*args: object, **kwargs: object) -> None:
                raise SimulatedProducerError("Simulated failure")

            mock_producer.send = failing_send
            mock_producer.start = AsyncMock()

            await bus.start()

            # Launch multiple concurrent publish operations that will fail
            async def failing_publish() -> None:
                """Publish operation that catches and suppresses expected failures."""
                try:
                    await bus.publish(topic="test", key=None, value=b"test")
                except Exception:
                    pass  # Expected - circuit breaker or producer failure

            # Launch enough to trigger circuit breaker (threshold=3, launching 5)
            tasks = [failing_publish() for _ in range(5)]
            await asyncio.gather(*tasks, return_exceptions=True)

            # Verify circuit breaker opened after threshold reached
            status = await bus.health_check()
            assert status["circuit_state"] == "open"

            # Cleanup: Close bus within patch context to ensure proper cleanup order
            await bus.close()
