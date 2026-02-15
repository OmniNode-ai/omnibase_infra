# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Load and stress tests for Event Bus.

This test suite validates event bus behavior under sustained load including:
- Sustained high-volume publishing
- Memory stability under load
- Multiple subscriber load testing
- Resource cleanup and recovery

Performance Thresholds:
    These tests validate stability under load, not raw performance.
    They ensure the system doesn't degrade, leak memory, or fail
    under sustained operation.

Usage:
    Run load tests:
        poetry run pytest tests/performance/event_bus/test_event_bus_load.py -v

    Skip in normal CI (use marker):
        poetry run pytest -m "not performance" tests/

Related:
    - OMN-57: Event bus performance testing (Phase 9)
    - EventBusInmemory: Primary implementation under test
"""

from __future__ import annotations

import asyncio
import gc
import time
from collections.abc import Awaitable, Callable
from statistics import mean

import pytest

from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.event_bus.models import ModelEventMessage
from tests.conftest import make_test_node_identity
from tests.performance.event_bus.conftest import generate_unique_topic

# Mark all tests in this module as performance tests
pytestmark = [pytest.mark.performance]

# -----------------------------------------------------------------------------
# Sustained Load Tests
# -----------------------------------------------------------------------------


class TestSustainedLoad:
    """Test sustained high-volume operation."""

    @pytest.mark.asyncio
    async def test_sustained_5_second_load(
        self,
        sample_message_bytes: bytes,
    ) -> None:
        """Test sustained publishing for 5 seconds.

        Validates that throughput remains stable over extended operation.

        Stability Target:
            Throughput variance < 75% between intervals (relaxed from 50%
            to handle GC pauses and CPU contention in full suite runs)
        """
        topic = generate_unique_topic()
        bus = EventBusInmemory(
            environment="sustained",
            group="load",
            max_history=10000,
        )
        await bus.start()

        interval_duration = 1.0
        num_intervals = 5
        interval_counts: list[int] = []

        overall_start = time.perf_counter()

        for interval in range(num_intervals):
            interval_start = time.perf_counter()
            count = 0

            while time.perf_counter() - interval_start < interval_duration:
                await bus.publish(
                    topic=topic,
                    key=f"sustained-{interval}-{count}".encode(),
                    value=sample_message_bytes,
                )
                count += 1

            interval_counts.append(count)

        total_time = time.perf_counter() - overall_start
        total_ops = sum(interval_counts)

        await bus.close()

        # Calculate throughput variance (guard against division by zero)
        avg_count = mean(interval_counts)
        if avg_count > 0:
            max_variance = max(abs(c - avg_count) / avg_count for c in interval_counts)
        else:
            max_variance = 0.0

        print(f"\nSustained Load ({num_intervals} x {interval_duration}s):")
        print(f"  Interval counts: {interval_counts}")
        print(f"  Total ops:       {total_ops}")
        print(f"  Total time:      {total_time:.2f}s")
        print(f"  Avg throughput:  {total_ops / total_time:.0f} ops/sec")
        print(f"  Max variance:    {max_variance * 100:.1f}%")

        # Variance threshold set at 75% to account for GC pauses, CPU contention,
        # and resource sharing when running in full test suite. The test passes
        # at 50% when run individually but is flaky under shared resource conditions.
        assert max_variance < 0.75, (
            f"Throughput variance {max_variance * 100:.1f}%, expected < 75%"
        )

    @pytest.mark.asyncio
    async def test_sustained_with_subscribers(
        self,
        sample_message_bytes: bytes,
    ) -> None:
        """Test sustained load with active subscribers.

        Validates that subscriber processing doesn't cause accumulation
        or memory issues over time.
        """
        topic = generate_unique_topic()
        bus = EventBusInmemory(
            environment="sub-load",
            group="sustained",
            max_history=5000,
        )
        await bus.start()

        received_count = 0
        lock = asyncio.Lock()

        async def handler(msg: ModelEventMessage) -> None:
            nonlocal received_count
            async with lock:
                received_count += 1

        await bus.subscribe(topic, make_test_node_identity("sustained-group"), handler)

        # Run for 3 seconds
        duration = 3.0
        start = time.perf_counter()
        published = 0

        while time.perf_counter() - start < duration:
            await bus.publish(
                topic=topic,
                key=f"msg-{published}".encode(),
                value=sample_message_bytes,
            )
            published += 1

        elapsed = time.perf_counter() - start

        await bus.close()

        # All published should be received
        assert received_count == published, (
            f"Received {received_count}/{published} messages"
        )

        print(f"\nSustained with Subscriber ({duration}s):")
        print(f"  Published: {published}")
        print(f"  Received:  {received_count}")
        print(f"  Rate:      {published / elapsed:.0f} msg/sec")


# -----------------------------------------------------------------------------
# Memory Stability Tests
# -----------------------------------------------------------------------------


class TestMemoryStability:
    """Test memory stability under sustained load."""

    @pytest.mark.asyncio
    async def test_memory_bounded_with_history_limit(
        self,
        sample_message_bytes: bytes,
    ) -> None:
        """Test that memory stays bounded due to history limit.

        Validates that the max_history setting prevents unbounded growth.
        """
        topic = generate_unique_topic()
        max_history = 1000
        bus = EventBusInmemory(
            environment="memory",
            group="bounded",
            max_history=max_history,
        )
        await bus.start()

        # Publish 10x the history limit
        num_messages = max_history * 10
        for i in range(num_messages):
            await bus.publish(
                topic=topic,
                key=f"mem-{i}".encode(),
                value=sample_message_bytes,
            )

        # Check history size is bounded
        history = await bus.get_event_history(limit=max_history + 100)
        assert len(history) <= max_history, (
            f"History size {len(history)}, expected <= {max_history}"
        )

        await bus.close()

        print("\nMemory Bounded Test:")
        print(f"  Published:     {num_messages}")
        print(f"  History limit: {max_history}")
        print(f"  History size:  {len(history)}")

    @pytest.mark.asyncio
    async def test_memory_after_subscriber_unsubscribe(
        self,
        sample_message_bytes: bytes,
    ) -> None:
        """Test memory cleanup after unsubscribing.

        Validates that unsubscribed handlers are properly cleaned up.
        """
        topic = generate_unique_topic()
        bus = EventBusInmemory(environment="cleanup", group="test")
        await bus.start()

        # Add and remove many subscribers
        for iteration in range(100):
            unsubscribes = []

            # Create 10 subscribers
            for i in range(10):

                async def handler(msg: ModelEventMessage) -> None:
                    pass

                unsub = await bus.subscribe(
                    topic, make_test_node_identity(f"group-{iteration}-{i}"), handler
                )
                unsubscribes.append(unsub)

            # Unsubscribe all
            for unsub in unsubscribes:
                await unsub()

        # Check subscriber count is 0
        sub_count = await bus.get_subscriber_count(topic)
        assert sub_count == 0, f"Subscriber count {sub_count}, expected 0"

        await bus.close()

        print("\nSubscriber Cleanup Test:")
        print("  Iterations:        100 x 10 subscribers")
        print(f"  Final subscribers: {sub_count}")

    @pytest.mark.asyncio
    async def test_no_memory_leak_under_load(
        self,
        sample_message_bytes: bytes,
    ) -> None:
        """Test for memory leaks during sustained operation.

        Runs multiple iterations and checks that memory growth is bounded.
        """
        topic = generate_unique_topic()
        bus = EventBusInmemory(
            environment="leak-test",
            group="memory",
            max_history=100,  # Small history
        )
        await bus.start()

        # Force GC before measurement
        gc.collect()
        initial_objects = len(gc.get_objects())

        # Run many operations
        for iteration in range(10):
            for i in range(1000):
                await bus.publish(
                    topic=topic,
                    key=f"leak-{iteration}-{i}".encode(),
                    value=sample_message_bytes,
                )

        # Force GC after operations
        gc.collect()
        final_objects = len(gc.get_objects())

        await bus.close()

        # Object count should not grow excessively
        # Allow some growth for internal structures, but not proportional to operations
        growth = final_objects - initial_objects
        growth_per_op = growth / 10000

        print("\nMemory Leak Check:")
        print(f"  Initial objects: {initial_objects}")
        print(f"  Final objects:   {final_objects}")
        print(f"  Growth:          {growth}")
        print(f"  Growth per op:   {growth_per_op:.4f}")

        # Less than 0.1 objects per operation (very conservative)
        assert growth_per_op < 0.1, (
            f"Object growth {growth_per_op:.4f} per op, expected < 0.1"
        )


# -----------------------------------------------------------------------------
# Multiple Subscriber Load Tests
# -----------------------------------------------------------------------------


class TestMultipleSubscriberLoad:
    """Test load with multiple subscribers."""

    @pytest.mark.asyncio
    async def test_fanout_to_100_subscribers(
        self,
        sample_message_bytes: bytes,
    ) -> None:
        """Test fanout to 100 subscribers.

        Validates that the event bus can handle high subscriber counts.
        """
        topic = generate_unique_topic()
        bus = EventBusInmemory(
            environment="fanout",
            group="100-subs",
            max_history=1000,
        )
        await bus.start()

        num_subscribers = 100
        counters = [0] * num_subscribers
        locks = [asyncio.Lock() for _ in range(num_subscribers)]

        # Create subscribers
        for idx in range(num_subscribers):

            def make_handler(
                i: int,
            ) -> Callable[[ModelEventMessage], Awaitable[None]]:
                async def handler(msg: ModelEventMessage) -> None:
                    async with locks[i]:
                        counters[i] += 1

                return handler

            h = make_handler(idx)
            await bus.subscribe(topic, make_test_node_identity(f"fanout-{idx}"), h)

        # Publish messages
        num_messages = 100
        start = time.perf_counter()
        for i in range(num_messages):
            await bus.publish(
                topic=topic,
                key=f"fan-{i}".encode(),
                value=sample_message_bytes,
            )
        elapsed = time.perf_counter() - start

        await bus.close()

        # All subscribers should receive all messages
        total_deliveries = sum(counters)
        expected_deliveries = num_subscribers * num_messages

        assert total_deliveries == expected_deliveries, (
            f"Deliveries {total_deliveries}, expected {expected_deliveries}"
        )

        deliveries_per_sec = total_deliveries / elapsed

        print(f"\nFanout to {num_subscribers} Subscribers:")
        print(f"  Messages:        {num_messages}")
        print(f"  Subscribers:     {num_subscribers}")
        print(f"  Total deliveries: {total_deliveries}")
        print(f"  Duration:        {elapsed:.3f}s")
        print(f"  Delivery rate:   {deliveries_per_sec:.0f}/sec")

    @pytest.mark.asyncio
    async def test_subscribers_across_topics(
        self,
        sample_message_bytes: bytes,
    ) -> None:
        """Test subscribers distributed across multiple topics.

        Validates that multi-topic operation scales correctly.
        """
        num_topics = 20
        subscribers_per_topic = 5
        messages_per_topic = 50

        bus = EventBusInmemory(
            environment="multi-topic",
            group="load",
            max_history=10000,
        )
        await bus.start()

        topics = [generate_unique_topic() for _ in range(num_topics)]
        counters: dict[str, int] = dict.fromkeys(topics, 0)
        locks: dict[str, asyncio.Lock] = {t: asyncio.Lock() for t in topics}

        # Create subscribers for each topic
        for topic in topics:
            for i in range(subscribers_per_topic):

                def make_handler(
                    t: str,
                ) -> Callable[[ModelEventMessage], Awaitable[None]]:
                    async def handler(msg: ModelEventMessage) -> None:
                        async with locks[t]:
                            counters[t] += 1

                    return handler

                h = make_handler(topic)
                await bus.subscribe(topic, make_test_node_identity(f"sub-{i}"), h)

        # Publish to all topics
        start = time.perf_counter()
        for topic in topics:
            for i in range(messages_per_topic):
                await bus.publish(
                    topic=topic,
                    key=f"msg-{i}".encode(),
                    value=sample_message_bytes,
                )
        elapsed = time.perf_counter() - start

        await bus.close()

        total_deliveries = sum(counters.values())
        expected = num_topics * messages_per_topic * subscribers_per_topic

        assert total_deliveries == expected, (
            f"Deliveries {total_deliveries}, expected {expected}"
        )

        print("\nMulti-Topic Load:")
        print(f"  Topics:          {num_topics}")
        print(f"  Subs per topic:  {subscribers_per_topic}")
        print(f"  Msgs per topic:  {messages_per_topic}")
        print(f"  Total deliveries: {total_deliveries}")
        print(f"  Duration:        {elapsed:.3f}s")
        print(f"  Rate:            {total_deliveries / elapsed:.0f}/sec")


# -----------------------------------------------------------------------------
# Recovery and Resilience Tests
# -----------------------------------------------------------------------------


class TestRecoveryResilience:
    """Test recovery and resilience under load."""

    @pytest.mark.asyncio
    async def test_recovery_after_subscriber_error(
        self,
        sample_message_bytes: bytes,
    ) -> None:
        """Test that bus recovers from subscriber errors.

        Validates that failing subscribers don't break the bus.
        """
        topic = generate_unique_topic()
        bus = EventBusInmemory(
            environment="recovery",
            group="error",
            circuit_breaker_threshold=100,  # High threshold to not trip
        )
        await bus.start()

        good_count = 0
        error_count = 0
        good_lock = asyncio.Lock()
        error_lock = asyncio.Lock()

        async def good_handler(msg: ModelEventMessage) -> None:
            nonlocal good_count
            async with good_lock:
                good_count += 1

        async def bad_handler(msg: ModelEventMessage) -> None:
            nonlocal error_count
            async with error_lock:
                error_count += 1
            raise ValueError("Intentional error")

        await bus.subscribe(topic, make_test_node_identity("good"), good_handler)
        await bus.subscribe(topic, make_test_node_identity("bad"), bad_handler)

        # Publish messages
        num_messages = 100
        for i in range(num_messages):
            await bus.publish(
                topic=topic,
                key=f"msg-{i}".encode(),
                value=sample_message_bytes,
            )

        await bus.close()

        # Good handler should receive all messages despite bad handler failing
        assert good_count == num_messages, (
            f"Good handler got {good_count}/{num_messages}"
        )

        print("\nRecovery After Subscriber Error:")
        print(f"  Messages:       {num_messages}")
        print(f"  Good received:  {good_count}")
        print(f"  Error count:    {error_count}")

    @pytest.mark.asyncio
    async def test_circuit_breaker_under_load(
        self,
        sample_message_bytes: bytes,
    ) -> None:
        """Test circuit breaker behavior under load.

        Validates that circuit breaker properly isolates failing subscribers.
        """
        topic = generate_unique_topic()
        threshold = 5
        bus = EventBusInmemory(
            environment="circuit",
            group="breaker",
            circuit_breaker_threshold=threshold,
        )
        await bus.start()

        fail_count = 0

        async def failing_handler(msg: ModelEventMessage) -> None:
            nonlocal fail_count
            fail_count += 1
            raise RuntimeError("Always fails")

        await bus.subscribe(topic, make_test_node_identity("failing"), failing_handler)

        # Publish enough to trip circuit breaker
        for i in range(threshold + 10):
            await bus.publish(
                topic=topic,
                key=f"msg-{i}".encode(),
                value=sample_message_bytes,
            )

        # Check circuit breaker status
        status = await bus.get_circuit_breaker_status()

        await bus.close()

        # Circuit should be open
        assert len(status["open_circuits"]) > 0, "Circuit breaker should be open"

        # Handler should have been called threshold times before circuit opened
        assert fail_count >= threshold, (
            f"Fail count {fail_count}, expected >= {threshold}"
        )

        print("\nCircuit Breaker Under Load:")
        print(f"  Threshold:     {threshold}")
        print(f"  Fail count:    {fail_count}")
        print(f"  Open circuits: {status['open_circuits']}")

    @pytest.mark.asyncio
    async def test_graceful_shutdown_under_load(
        self,
        sample_message_bytes: bytes,
    ) -> None:
        """Test graceful shutdown while under load.

        Validates that shutdown completes cleanly during active publishing.
        """
        topic = generate_unique_topic()
        bus = EventBusInmemory(environment="shutdown", group="test")
        await bus.start()

        published = 0
        shutdown_requested = False

        async def publisher() -> None:
            nonlocal published
            while not shutdown_requested:
                await bus.publish(
                    topic=topic,
                    key=f"msg-{published}".encode(),
                    value=sample_message_bytes,
                )
                published += 1
                await asyncio.sleep(0.001)  # Small delay

        # Start publishing task
        task = asyncio.create_task(publisher())

        # Let it run briefly
        await asyncio.sleep(0.1)

        # Request shutdown
        shutdown_requested = True
        await task

        # Shutdown should complete without error
        start = time.perf_counter()
        await bus.close()
        shutdown_time = time.perf_counter() - start

        print("\nGraceful Shutdown Under Load:")
        print(f"  Published before shutdown: {published}")
        print(f"  Shutdown time:            {shutdown_time * 1000:.1f}ms")

        # Shutdown should be fast (< 1 second)
        assert shutdown_time < 1.0, f"Shutdown took {shutdown_time:.2f}s, expected < 1s"
