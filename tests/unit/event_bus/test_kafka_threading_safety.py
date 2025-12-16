# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Threading safety tests for KafkaEventBus race condition fixes."""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest

from omnibase_infra.event_bus.kafka_event_bus import KafkaEventBus
from omnibase_infra.event_bus.models import ModelEventHeaders


class SimulatedProducerError(Exception):
    """Custom exception for simulating producer failures in tests."""


@pytest.mark.asyncio
class TestKafkaEventBusThreadingSafety:
    """Test suite for KafkaEventBus threading safety and race condition fixes."""

    async def test_concurrent_publish_operations_thread_safe(self):
        """Test that concurrent publish operations don't cause race conditions."""
        # Create event bus with mocked producer
        bus = KafkaEventBus.default()

        # Mock the producer
        mock_producer = AsyncMock()
        mock_future = asyncio.Future()
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
            async def publish_task(i: int):
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

        await bus.close()

    async def test_initialize_start_race_condition_fixed(self):
        """Test that initialize() doesn't race with start()."""
        bus = KafkaEventBus.default()

        with patch(
            "omnibase_infra.event_bus.kafka_event_bus.AIOKafkaProducer"
        ) as MockProducer:
            mock_producer = AsyncMock()
            mock_producer.start = AsyncMock()
            MockProducer.return_value = mock_producer

            # Simulate concurrent initialize and config updates
            async def init_task():
                try:
                    await bus.initialize(
                        {
                            "environment": "test-env",
                            "group": "test-group",
                        }
                    )
                except Exception:
                    pass

            async def update_task():
                # Try to read environment during initialization
                _ = bus.environment

            # Run both concurrently
            await asyncio.gather(init_task(), update_task(), return_exceptions=True)

            # Verify final state is consistent
            assert bus.environment == "test-env"
            assert bus.group == "test-group"

        await bus.close()

    async def test_producer_access_during_retry_thread_safe(self):
        """Test that producer field access during retry is thread-safe."""
        bus = KafkaEventBus.default()

        with patch(
            "omnibase_infra.event_bus.kafka_event_bus.AIOKafkaProducer"
        ) as MockProducer:
            mock_producer = AsyncMock()
            MockProducer.return_value = mock_producer

            # Simulate producer that succeeds on both attempts (no timeout)
            call_count = 0

            async def send_success(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                future = asyncio.Future()
                future.set_result(Mock(partition=0, offset=0))
                return future

            mock_producer.send = send_success
            mock_producer.start = AsyncMock()

            await bus.start()

            # Multiple concurrent publishes should all succeed with thread-safe producer access
            async def publish_task(i: int):
                await bus.publish(
                    topic="test-topic",
                    key=f"key-{i}".encode(),
                    value=f"value-{i}".encode(),
                )

            # Launch 5 concurrent publishes
            await asyncio.gather(*[publish_task(i) for i in range(5)])

            # Verify all publishes succeeded
            assert call_count == 5

        await bus.close()

    async def test_concurrent_close_operations_thread_safe(self):
        """Test that concurrent close operations don't cause race conditions."""
        bus = KafkaEventBus.default()

        with patch(
            "omnibase_infra.event_bus.kafka_event_bus.AIOKafkaProducer"
        ) as MockProducer:
            mock_producer = AsyncMock()
            mock_producer.start = AsyncMock()
            mock_producer.stop = AsyncMock()
            MockProducer.return_value = mock_producer

            await bus.start()

            # Launch multiple concurrent close operations
            close_tasks = [bus.close() for _ in range(5)]
            await asyncio.gather(*close_tasks)

            # Verify bus is properly closed
            assert bus._shutdown is True
            assert bus._started is False

    async def test_health_check_during_shutdown_thread_safe(self):
        """Test that health_check() during shutdown is thread-safe."""
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
            async def health_task():
                for _ in range(10):
                    await bus.health_check()
                    await asyncio.sleep(0.01)

            health = asyncio.create_task(health_task())
            await asyncio.sleep(0.05)
            await bus.close()
            await health

            # Final health check should show bus is closed
            status = await bus.health_check()
            assert status["started"] is False

    async def test_circuit_breaker_concurrent_access_thread_safe(self):
        """Test that circuit breaker state is thread-safe under concurrent access."""
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
            async def failing_send(*args, **kwargs):
                raise SimulatedProducerError("Simulated failure")

            mock_producer.send = failing_send
            mock_producer.start = AsyncMock()

            await bus.start()

            # Launch multiple concurrent publish operations that will fail
            async def failing_publish():
                try:
                    await bus.publish(topic="test", key=None, value=b"test")
                except Exception:
                    pass

            # Launch enough to trigger circuit breaker
            tasks = [failing_publish() for _ in range(5)]
            await asyncio.gather(*tasks, return_exceptions=True)

            # Verify circuit breaker opened
            status = await bus.health_check()
            assert status["circuit_state"] == "open"

        await bus.close()
