# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Race condition and concurrent access tests for InMemoryEventBus.

This module provides comprehensive async race condition tests for:
- Concurrent publish operations
- Concurrent subscribe/unsubscribe operations
- Circuit breaker state transitions under concurrent load
- Event history consistency under concurrent access
- Topic offset integrity under concurrent publishes

Test Categories:
1. Concurrent Publish: Multiple coroutines publishing simultaneously
2. Concurrent Subscribe/Unsubscribe: Subscription management under load
3. Circuit Breaker Race Conditions: State transitions under concurrent failures
4. History Consistency: Event history integrity under concurrent access
5. Offset Consistency: Topic offset atomicity under concurrent writes

All tests use asyncio.gather() for true concurrent execution
and are designed to be deterministic.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

import pytest

from omnibase_infra.errors import InfraUnavailableError
from omnibase_infra.event_bus.inmemory_event_bus import InMemoryEventBus
from omnibase_infra.event_bus.models import ModelEventMessage

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def event_bus() -> InMemoryEventBus:
    """Create a fresh InMemoryEventBus instance for each test."""
    return InMemoryEventBus(
        environment="test",
        group="test-group",
        max_history=10000,  # Large history to avoid truncation during tests
        circuit_breaker_threshold=5,
    )


# =============================================================================
# Concurrent Publish Tests
# =============================================================================


class TestConcurrentPublishOperations:
    """Tests for concurrent publish operation thread safety."""

    @pytest.mark.asyncio
    async def test_concurrent_publish_same_topic(
        self, event_bus: InMemoryEventBus
    ) -> None:
        """Test concurrent publishes to the same topic maintain data integrity."""
        await event_bus.start()

        num_publishers = 10
        messages_per_publisher = 100

        async def publish_batch(publisher_id: int) -> None:
            for i in range(messages_per_publisher):
                await event_bus.publish(
                    topic="concurrent-topic",
                    key=f"key-{publisher_id}-{i}".encode(),
                    value=f"value-{publisher_id}-{i}".encode(),
                )

        # Launch all publishers concurrently
        await asyncio.gather(*[publish_batch(i) for i in range(num_publishers)])

        # Verify all messages were published
        history = await event_bus.get_event_history(limit=10000)
        assert len(history) == num_publishers * messages_per_publisher

        # Verify topic offset is correct
        offset = await event_bus.get_topic_offset("concurrent-topic")
        assert offset == num_publishers * messages_per_publisher

        await event_bus.close()

    @pytest.mark.asyncio
    async def test_concurrent_publish_multiple_topics(
        self, event_bus: InMemoryEventBus
    ) -> None:
        """Test concurrent publishes to different topics are isolated."""
        await event_bus.start()

        num_topics = 5
        messages_per_topic = 50

        async def publish_to_topic(topic_id: int) -> None:
            for i in range(messages_per_topic):
                await event_bus.publish(
                    topic=f"topic-{topic_id}",
                    key=None,
                    value=f"msg-{i}".encode(),
                )

        # Launch publishers for each topic concurrently
        await asyncio.gather(*[publish_to_topic(i) for i in range(num_topics)])

        # Verify each topic has correct offset
        for topic_id in range(num_topics):
            offset = await event_bus.get_topic_offset(f"topic-{topic_id}")
            assert offset == messages_per_topic

        # Verify total history
        history = await event_bus.get_event_history(limit=10000)
        assert len(history) == num_topics * messages_per_topic

        await event_bus.close()

    @pytest.mark.asyncio
    async def test_concurrent_publish_with_subscribers(
        self, event_bus: InMemoryEventBus
    ) -> None:
        """Test concurrent publish with active subscribers receiving all messages."""
        await event_bus.start()

        received_messages: list[ModelEventMessage] = []
        lock = asyncio.Lock()

        async def handler(msg: ModelEventMessage) -> None:
            async with lock:
                received_messages.append(msg)

        # Set up subscriber
        await event_bus.subscribe("concurrent-topic", "group1", handler)

        num_publishers = 5
        messages_per_publisher = 50

        async def publish_batch(publisher_id: int) -> None:
            for i in range(messages_per_publisher):
                await event_bus.publish(
                    topic="concurrent-topic",
                    key=None,
                    value=f"pub-{publisher_id}-msg-{i}".encode(),
                )

        # Launch publishers concurrently
        await asyncio.gather(*[publish_batch(i) for i in range(num_publishers)])

        # Verify all messages were received
        assert len(received_messages) == num_publishers * messages_per_publisher

        await event_bus.close()

    @pytest.mark.asyncio
    async def test_concurrent_publish_offset_uniqueness(
        self, event_bus: InMemoryEventBus
    ) -> None:
        """Test that concurrent publishes get unique sequential offsets."""
        await event_bus.start()

        num_concurrent = 100
        offsets: list[str] = []
        lock = asyncio.Lock()

        async def publish_and_record() -> None:
            await event_bus.publish(
                topic="offset-test",
                key=None,
                value=b"test",
            )

        # Launch all publishes concurrently
        await asyncio.gather(*[publish_and_record() for _ in range(num_concurrent)])

        # Get all messages and check offsets
        history = await event_bus.get_event_history(limit=1000)
        offsets = [msg.offset for msg in history if msg.topic == "offset-test"]

        # All offsets should be unique
        assert len(offsets) == len(set(offsets))
        # Offsets should be sequential (0 through num_concurrent-1)
        offset_ints = sorted(int(o) for o in offsets)
        assert offset_ints == list(range(num_concurrent))

        await event_bus.close()


# =============================================================================
# Concurrent Subscribe/Unsubscribe Tests
# =============================================================================


class TestConcurrentSubscribeUnsubscribe:
    """Tests for concurrent subscribe and unsubscribe operations."""

    @pytest.mark.asyncio
    async def test_concurrent_subscribe_same_topic(
        self, event_bus: InMemoryEventBus
    ) -> None:
        """Test concurrent subscriptions to the same topic."""
        await event_bus.start()

        num_subscribers = 50
        unsubscribes: list[Callable[[], Awaitable[None]]] = []
        lock = asyncio.Lock()

        async def handler(msg: ModelEventMessage) -> None:
            pass

        async def subscribe_task(sub_id: int) -> None:
            unsub = await event_bus.subscribe(
                topic="shared-topic",
                group_id=f"group-{sub_id}",
                on_message=handler,
            )
            async with lock:
                unsubscribes.append(unsub)

        # Subscribe concurrently
        await asyncio.gather(*[subscribe_task(i) for i in range(num_subscribers)])

        # Verify all subscriptions registered
        count = await event_bus.get_subscriber_count(topic="shared-topic")
        assert count == num_subscribers

        await event_bus.close()

    @pytest.mark.asyncio
    async def test_concurrent_unsubscribe(self, event_bus: InMemoryEventBus) -> None:
        """Test concurrent unsubscription operations."""
        await event_bus.start()

        num_subscribers = 50
        unsubscribes: list[Callable[[], Awaitable[None]]] = []

        async def handler(msg: ModelEventMessage) -> None:
            pass

        # Subscribe all first (sequentially to collect unsubscribe functions)
        for i in range(num_subscribers):
            unsub = await event_bus.subscribe(
                topic="shared-topic",
                group_id=f"group-{i}",
                on_message=handler,
            )
            unsubscribes.append(unsub)

        # Verify all subscribed
        assert (
            await event_bus.get_subscriber_count(topic="shared-topic")
            == num_subscribers
        )

        # Unsubscribe all concurrently
        await asyncio.gather(*[unsub() for unsub in unsubscribes])

        # Verify all unsubscribed
        count = await event_bus.get_subscriber_count(topic="shared-topic")
        assert count == 0

        await event_bus.close()

    @pytest.mark.asyncio
    async def test_concurrent_subscribe_unsubscribe_interleaved(
        self, event_bus: InMemoryEventBus
    ) -> None:
        """Test interleaved subscribe and unsubscribe operations."""
        await event_bus.start()

        async def handler(msg: ModelEventMessage) -> None:
            pass

        async def subscribe_unsubscribe_cycle(sub_id: int) -> None:
            for _ in range(10):
                unsub = await event_bus.subscribe(
                    topic="interleaved-topic",
                    group_id=f"group-{sub_id}",
                    on_message=handler,
                )
                await asyncio.sleep(0)  # Yield to other tasks
                await unsub()

        # Run many subscribe/unsubscribe cycles concurrently
        await asyncio.gather(*[subscribe_unsubscribe_cycle(i) for i in range(20)])

        # Final state should have 0 subscribers (all unsubscribed)
        count = await event_bus.get_subscriber_count(topic="interleaved-topic")
        assert count == 0

        await event_bus.close()

    @pytest.mark.asyncio
    async def test_concurrent_double_unsubscribe_safety(
        self, event_bus: InMemoryEventBus
    ) -> None:
        """Test that concurrent double unsubscribe is safe."""
        await event_bus.start()

        async def handler(msg: ModelEventMessage) -> None:
            pass

        unsub = await event_bus.subscribe(
            topic="double-unsub-topic",
            group_id="group1",
            on_message=handler,
        )

        # Call unsubscribe concurrently multiple times
        await asyncio.gather(*[unsub() for _ in range(10)])

        # Should not raise and subscriber count should be 0
        count = await event_bus.get_subscriber_count(topic="double-unsub-topic")
        assert count == 0

        await event_bus.close()


# =============================================================================
# Circuit Breaker Race Condition Tests
# =============================================================================


class TestCircuitBreakerRaceConditions:
    """Tests for circuit breaker state transitions under concurrent load."""

    @pytest.mark.asyncio
    async def test_concurrent_failures_trigger_circuit_open(self) -> None:
        """Test that concurrent failures properly trigger circuit breaker."""
        event_bus = InMemoryEventBus(
            environment="test",
            group="test-group",
            circuit_breaker_threshold=5,
        )
        await event_bus.start()

        failure_count = 0
        lock = asyncio.Lock()

        async def failing_handler(msg: ModelEventMessage) -> None:
            nonlocal failure_count
            async with lock:
                failure_count += 1
            raise ValueError("Intentional failure")

        await event_bus.subscribe("circuit-topic", "fail-group", failing_handler)

        # Send messages concurrently to trigger failures
        async def send_message(i: int) -> None:
            await event_bus.publish(
                topic="circuit-topic",
                key=None,
                value=f"msg-{i}".encode(),
            )

        # Send more messages than threshold to ensure circuit opens
        await asyncio.gather(*[send_message(i) for i in range(10)])

        # Verify circuit breaker opened (handler should have stopped being called)
        status = await event_bus.get_circuit_breaker_status()
        assert len(status["open_circuits"]) == 1
        assert failure_count == 5  # Only 5 failures before circuit opened

        await event_bus.close()

    @pytest.mark.asyncio
    async def test_concurrent_success_resets_circuit_breaker(self) -> None:
        """Test that successful operations reset failure count correctly."""
        event_bus = InMemoryEventBus(
            environment="test",
            group="test-group",
            circuit_breaker_threshold=5,
        )
        await event_bus.start()

        call_count = 0
        should_fail = True
        lock = asyncio.Lock()

        async def flaky_handler(msg: ModelEventMessage) -> None:
            nonlocal call_count
            async with lock:
                call_count += 1
                if should_fail and call_count <= 3:
                    raise ValueError("Intentional failure")

        await event_bus.subscribe("reset-topic", "flaky-group", flaky_handler)

        # Send 3 failing messages
        for i in range(3):
            await event_bus.publish("reset-topic", None, f"fail-{i}".encode())

        # Verify failure count (should be 3)
        status = await event_bus.get_circuit_breaker_status()
        assert status["failure_counts"]["reset-topic:flaky-group"] == 3

        # Now send success
        should_fail = False
        await event_bus.publish("reset-topic", None, b"success")

        # Verify failure count reset
        status = await event_bus.get_circuit_breaker_status()
        assert "reset-topic:flaky-group" not in status["failure_counts"]

        await event_bus.close()

    @pytest.mark.asyncio
    async def test_concurrent_circuit_reset_operations(self) -> None:
        """Test concurrent circuit reset operations are safe."""
        event_bus = InMemoryEventBus(
            environment="test",
            group="test-group",
            circuit_breaker_threshold=3,
        )
        await event_bus.start()

        async def failing_handler(msg: ModelEventMessage) -> None:
            raise ValueError("Always fails")

        await event_bus.subscribe("reset-test", "fail-group", failing_handler)

        # Trigger circuit open
        for i in range(5):
            await event_bus.publish("reset-test", None, f"msg-{i}".encode())

        # Verify circuit is open
        status = await event_bus.get_circuit_breaker_status()
        assert len(status["open_circuits"]) == 1

        # Concurrent resets should all be safe
        results = await asyncio.gather(
            *[
                event_bus.reset_subscriber_circuit("reset-test", "fail-group")
                for _ in range(10)
            ]
        )

        # First reset should return True, others False
        assert results.count(True) == 1
        assert results.count(False) == 9

        await event_bus.close()


# =============================================================================
# Event History Consistency Tests
# =============================================================================


class TestEventHistoryConsistency:
    """Tests for event history integrity under concurrent access."""

    @pytest.mark.asyncio
    async def test_concurrent_publish_and_history_read(
        self, event_bus: InMemoryEventBus
    ) -> None:
        """Test reading history while publishing doesn't cause issues."""
        await event_bus.start()

        publish_count = 0
        read_count = 0
        lock = asyncio.Lock()

        async def publish_task() -> None:
            nonlocal publish_count
            for i in range(100):
                await event_bus.publish("history-topic", None, f"msg-{i}".encode())
                async with lock:
                    publish_count += 1

        async def read_task() -> None:
            nonlocal read_count
            for _ in range(50):
                _ = await event_bus.get_event_history(limit=100)
                async with lock:
                    read_count += 1
                await asyncio.sleep(0)  # Yield to other tasks

        # Run publishers and readers concurrently
        await asyncio.gather(
            publish_task(),
            publish_task(),
            read_task(),
            read_task(),
        )

        assert publish_count == 200
        assert read_count == 100

        await event_bus.close()

    @pytest.mark.asyncio
    async def test_concurrent_clear_history(self, event_bus: InMemoryEventBus) -> None:
        """Test concurrent history clear operations."""
        await event_bus.start()

        # Publish some messages
        for i in range(100):
            await event_bus.publish("clear-topic", None, f"msg-{i}".encode())

        # Concurrent clears should be safe
        await asyncio.gather(*[event_bus.clear_event_history() for _ in range(10)])

        # History should be empty
        history = await event_bus.get_event_history(limit=1000)
        assert len(history) == 0

        await event_bus.close()

    @pytest.mark.asyncio
    async def test_history_max_limit_under_concurrent_writes(self) -> None:
        """Test that max_history is respected under concurrent writes."""
        max_history = 100
        event_bus = InMemoryEventBus(
            environment="test",
            group="test-group",
            max_history=max_history,
        )
        await event_bus.start()

        num_publishers = 10
        messages_per_publisher = 50  # Total 500 messages, but max_history is 100

        async def publish_batch(publisher_id: int) -> None:
            for i in range(messages_per_publisher):
                await event_bus.publish(
                    topic="max-history-topic",
                    key=None,
                    value=f"pub-{publisher_id}-msg-{i}".encode(),
                )

        await asyncio.gather(*[publish_batch(i) for i in range(num_publishers)])

        # History should be limited to max_history
        history = await event_bus.get_event_history(limit=1000)
        assert len(history) == max_history

        await event_bus.close()


# =============================================================================
# Start/Stop/Lifecycle Race Condition Tests
# =============================================================================


class TestLifecycleRaceConditions:
    """Tests for lifecycle method race conditions."""

    @pytest.mark.asyncio
    async def test_concurrent_start_calls(self, event_bus: InMemoryEventBus) -> None:
        """Test concurrent start calls are idempotent."""
        # Multiple concurrent starts should be safe
        await asyncio.gather(*[event_bus.start() for _ in range(10)])

        health = await event_bus.health_check()
        assert health["started"] is True

        await event_bus.close()

    @pytest.mark.asyncio
    async def test_concurrent_close_calls(self, event_bus: InMemoryEventBus) -> None:
        """Test concurrent close calls are safe."""
        await event_bus.start()

        # Multiple concurrent closes should be safe
        await asyncio.gather(*[event_bus.close() for _ in range(10)])

        health = await event_bus.health_check()
        assert health["started"] is False

    @pytest.mark.asyncio
    async def test_publish_during_close(self, event_bus: InMemoryEventBus) -> None:
        """Test publish operations during close are handled gracefully."""
        await event_bus.start()

        errors: list[Exception] = []
        success_count = 0
        lock = asyncio.Lock()

        async def publish_task() -> None:
            nonlocal success_count
            for i in range(50):
                try:
                    await event_bus.publish(
                        topic="during-close",
                        key=None,
                        value=f"msg-{i}".encode(),
                    )
                    async with lock:
                        success_count += 1
                except InfraUnavailableError:
                    pass  # Expected after close
                except Exception as e:
                    errors.append(e)
                await asyncio.sleep(0)  # Yield

        async def close_task() -> None:
            await asyncio.sleep(0.01)  # Let some publishes start
            await event_bus.close()

        await asyncio.gather(publish_task(), close_task())

        # No unexpected errors
        assert len(errors) == 0
        # Some publishes should have succeeded before close
        assert success_count > 0

    @pytest.mark.asyncio
    async def test_subscribe_during_close(self, event_bus: InMemoryEventBus) -> None:
        """Test subscribe operations during close."""
        await event_bus.start()

        async def handler(msg: ModelEventMessage) -> None:
            pass

        errors: list[Exception] = []
        subscribe_count = 0
        lock = asyncio.Lock()

        async def subscribe_task() -> None:
            nonlocal subscribe_count
            for i in range(20):
                try:
                    await event_bus.subscribe(
                        topic=f"topic-{i}",
                        group_id="group1",
                        on_message=handler,
                    )
                    async with lock:
                        subscribe_count += 1
                except Exception as e:
                    errors.append(e)
                await asyncio.sleep(0)

        async def close_task() -> None:
            await asyncio.sleep(0.005)
            await event_bus.close()

        await asyncio.gather(subscribe_task(), close_task())

        # No unexpected errors (subscribing during/after close is valid operation)
        # Close clears subscribers, but subscription calls should not error
        assert len(errors) == 0


# =============================================================================
# Health Check Race Condition Tests
# =============================================================================


class TestHealthCheckRaceConditions:
    """Tests for health check consistency under concurrent operations."""

    @pytest.mark.asyncio
    async def test_concurrent_health_checks(self, event_bus: InMemoryEventBus) -> None:
        """Test concurrent health checks return consistent data."""
        await event_bus.start()

        results: list[dict[str, object]] = []
        lock = asyncio.Lock()

        async def health_check_task() -> None:
            for _ in range(50):
                result = await event_bus.health_check()
                async with lock:
                    results.append(result)
                await asyncio.sleep(0)

        await asyncio.gather(*[health_check_task() for _ in range(5)])

        # All results should show started=True
        assert all(r["started"] is True for r in results)
        assert all(r["healthy"] is True for r in results)

        await event_bus.close()

    @pytest.mark.asyncio
    async def test_health_check_during_operations(
        self, event_bus: InMemoryEventBus
    ) -> None:
        """Test health checks during concurrent publish/subscribe operations."""
        await event_bus.start()

        health_results: list[dict[str, object]] = []
        lock = asyncio.Lock()

        async def handler(msg: ModelEventMessage) -> None:
            pass

        async def publish_task() -> None:
            for i in range(100):
                await event_bus.publish("ops-topic", None, f"msg-{i}".encode())

        async def subscribe_task() -> None:
            for i in range(50):
                unsub = await event_bus.subscribe(f"topic-{i}", f"group-{i}", handler)
                await asyncio.sleep(0)
                await unsub()

        async def health_task() -> None:
            for _ in range(50):
                result = await event_bus.health_check()
                async with lock:
                    health_results.append(result)
                await asyncio.sleep(0)

        await asyncio.gather(
            publish_task(),
            subscribe_task(),
            health_task(),
        )

        # All health checks should succeed without error
        assert len(health_results) == 50
        assert all(r["started"] is True for r in health_results)

        await event_bus.close()


# =============================================================================
# Stress Tests
# =============================================================================


class TestInMemoryEventBusStress:
    """High-volume stress tests for InMemoryEventBus."""

    @pytest.mark.asyncio
    async def test_high_volume_publish_stress(
        self, event_bus: InMemoryEventBus
    ) -> None:
        """Stress test with high volume of concurrent publishes."""
        await event_bus.start()

        num_publishers = 20
        messages_per_publisher = 100

        async def publish_batch(publisher_id: int) -> None:
            for i in range(messages_per_publisher):
                await event_bus.publish(
                    topic="stress-topic",
                    key=f"key-{publisher_id}-{i}".encode(),
                    value=f"value-{publisher_id}-{i}".encode(),
                )

        await asyncio.gather(*[publish_batch(i) for i in range(num_publishers)])

        # Verify all messages
        history = await event_bus.get_event_history(limit=10000)
        assert len(history) == num_publishers * messages_per_publisher

        await event_bus.close()

    @pytest.mark.asyncio
    async def test_mixed_operations_stress(self, event_bus: InMemoryEventBus) -> None:
        """Stress test with mixed concurrent operations."""
        await event_bus.start()

        received: list[ModelEventMessage] = []
        lock = asyncio.Lock()

        async def handler(msg: ModelEventMessage) -> None:
            async with lock:
                received.append(msg)

        errors: list[Exception] = []

        async def publisher(pub_id: int) -> None:
            try:
                for i in range(50):
                    await event_bus.publish(
                        topic="mixed-topic",
                        key=None,
                        value=f"pub-{pub_id}-{i}".encode(),
                    )
            except Exception as e:
                errors.append(e)

        async def subscriber(sub_id: int) -> None:
            try:
                unsub = await event_bus.subscribe(
                    topic="mixed-topic",
                    group_id=f"group-{sub_id}",
                    on_message=handler,
                )
                await asyncio.sleep(0.1)  # Let publishers run
                await unsub()
            except Exception as e:
                errors.append(e)

        async def reader() -> None:
            try:
                for _ in range(20):
                    await event_bus.get_event_history(limit=100)
                    await event_bus.health_check()
                    await asyncio.sleep(0)
            except Exception as e:
                errors.append(e)

        # Launch all operations concurrently
        tasks: list[asyncio.Task[None]] = []
        tasks.extend([asyncio.create_task(publisher(i)) for i in range(10)])
        tasks.extend([asyncio.create_task(subscriber(i)) for i in range(5)])
        tasks.extend([asyncio.create_task(reader()) for _ in range(3)])

        await asyncio.gather(*tasks)

        assert len(errors) == 0, f"Errors: {errors}"

        await event_bus.close()
