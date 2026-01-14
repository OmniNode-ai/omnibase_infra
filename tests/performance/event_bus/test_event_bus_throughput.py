# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Throughput performance tests for Event Bus.

This test suite measures event throughput characteristics including:
- Messages per second for publish operations
- Batch publishing performance
- Concurrent publisher throughput
- Subscriber delivery throughput

Performance Thresholds:
    These thresholds are intentionally lenient for CI environments
    where resources may be constrained. Adjust for dedicated perf testing.

    - Single publisher: > 1000 events/sec
    - Batch publishing: > 5000 events/sec
    - Concurrent publishers: > 2000 events/sec total

Usage:
    Run throughput tests:
        poetry run pytest tests/performance/event_bus/test_event_bus_throughput.py -v

    Skip in normal CI (use marker):
        poetry run pytest -m "not performance" tests/

Related:
    - OMN-57: Event bus performance testing (Phase 9)
    - InMemoryEventBus: Primary implementation under test
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable

import pytest

from omnibase_infra.event_bus.inmemory_event_bus import InMemoryEventBus
from omnibase_infra.event_bus.models import ModelEventMessage
from tests.performance.event_bus.conftest import generate_unique_topic

# -----------------------------------------------------------------------------
# Single Publisher Throughput Tests
# -----------------------------------------------------------------------------


class TestSinglePublisherThroughput:
    """Test single publisher throughput characteristics."""

    @pytest.mark.asyncio
    async def test_1000_sequential_publishes(
        self,
        event_bus: InMemoryEventBus,
        sample_message_bytes: bytes,
    ) -> None:
        """Test 1000 sequential publish operations.

        Validates that sequential publishing achieves > 1000 events/sec
        with the InMemoryEventBus.

        Performance Target:
            > 1000 events/sec (lenient for CI)
        """
        topic = generate_unique_topic()

        start = time.perf_counter()
        for i in range(1000):
            await event_bus.publish(
                topic=topic,
                key=f"key-{i}".encode(),
                value=sample_message_bytes,
            )
        elapsed = time.perf_counter() - start

        events_per_sec = 1000 / elapsed

        # Performance assertion
        assert events_per_sec > 1000, (
            f"Throughput {events_per_sec:.0f} events/sec, expected > 1000"
        )

        # Verify all messages were stored
        offset = await event_bus.get_topic_offset(topic)
        assert offset == 1000

        print("\nSequential Publish (1000 events):")
        print(f"  Duration:   {elapsed:.3f}s")
        print(f"  Throughput: {events_per_sec:.0f} events/sec")

    @pytest.mark.asyncio
    @pytest.mark.xfail(
        reason="Environment-dependent throughput: CI runners may not achieve 5000 events/sec "
        "due to variable CPU/memory resources. Test still runs to capture metrics.",
        strict=False,
    )
    async def test_10000_sequential_publishes(
        self,
        high_volume_event_bus: InMemoryEventBus,
        sample_message_bytes: bytes,
    ) -> None:
        """Test 10000 sequential publish operations.

        Higher volume test for sustained throughput measurement.

        Performance Target:
            > 5000 events/sec (with no subscribers)
        """
        topic = generate_unique_topic()

        start = time.perf_counter()
        for i in range(10000):
            await high_volume_event_bus.publish(
                topic=topic,
                key=f"key-{i}".encode(),
                value=sample_message_bytes,
            )
        elapsed = time.perf_counter() - start

        events_per_sec = 10000 / elapsed

        assert events_per_sec > 5000, (
            f"Throughput {events_per_sec:.0f} events/sec, expected > 5000"
        )

        print("\nSequential Publish (10000 events):")
        print(f"  Duration:   {elapsed:.3f}s")
        print(f"  Throughput: {events_per_sec:.0f} events/sec")

    @pytest.mark.asyncio
    async def test_sustained_throughput_1_second(
        self,
        event_bus: InMemoryEventBus,
        sample_message_bytes: bytes,
    ) -> None:
        """Measure sustained throughput over 1 second.

        Runs continuous publishes for a fixed duration to measure
        real-world sustained throughput.

        Performance Target:
            > 1000 events/sec sustained
        """
        topic = generate_unique_topic()
        target_duration = 1.0
        operations = 0

        start = time.perf_counter()
        while time.perf_counter() - start < target_duration:
            await event_bus.publish(
                topic=topic,
                key=f"key-{operations}".encode(),
                value=sample_message_bytes,
            )
            operations += 1

        actual_duration = time.perf_counter() - start
        events_per_sec = operations / actual_duration

        assert events_per_sec > 1000, (
            f"Sustained throughput {events_per_sec:.0f} events/sec, expected > 1000"
        )

        print("\nSustained Throughput (1 second):")
        print(f"  Operations: {operations}")
        print(f"  Duration:   {actual_duration:.3f}s")
        print(f"  Throughput: {events_per_sec:.0f} events/sec")


# -----------------------------------------------------------------------------
# Batch Publishing Tests
# -----------------------------------------------------------------------------


class TestBatchPublishing:
    """Test batch publishing performance."""

    @pytest.mark.asyncio
    async def test_batch_publish_100_messages(
        self,
        event_bus: InMemoryEventBus,
        sample_message_bytes: bytes,
    ) -> None:
        """Test publishing 100 messages in rapid succession.

        Performance Target:
            Complete in < 100ms
        """
        topic = generate_unique_topic()

        start = time.perf_counter()
        tasks = [
            event_bus.publish(
                topic=topic,
                key=f"key-{i}".encode(),
                value=sample_message_bytes,
            )
            for i in range(100)
        ]
        # Execute sequentially (batch pattern)
        for task in tasks:
            await task
        elapsed = time.perf_counter() - start

        assert elapsed < 0.1, f"Batch 100 took {elapsed:.3f}s, expected < 0.1s"

        print("\nBatch Publish (100 messages):")
        print(f"  Duration:   {elapsed * 1000:.1f}ms")
        print(f"  Throughput: {100 / elapsed:.0f} events/sec")

    @pytest.mark.asyncio
    @pytest.mark.xfail(
        reason="Environment-dependent throughput: CI runners may not achieve 5000 events/sec "
        "due to variable CPU/memory resources (observed 3220/sec in CI). "
        "Test still runs to capture metrics.",
        strict=False,
    )
    async def test_batch_publish_1000_messages(
        self,
        event_bus: InMemoryEventBus,
        sample_message_bytes: bytes,
    ) -> None:
        """Test publishing 1000 messages in rapid succession.

        Performance Target:
            > 5000 events/sec
        """
        topic = generate_unique_topic()

        start = time.perf_counter()
        for i in range(1000):
            await event_bus.publish(
                topic=topic,
                key=f"key-{i}".encode(),
                value=sample_message_bytes,
            )
        elapsed = time.perf_counter() - start

        events_per_sec = 1000 / elapsed

        assert events_per_sec > 5000, (
            f"Batch throughput {events_per_sec:.0f} events/sec, expected > 5000"
        )

        print("\nBatch Publish (1000 messages):")
        print(f"  Duration:   {elapsed:.3f}s")
        print(f"  Throughput: {events_per_sec:.0f} events/sec")

    @pytest.mark.asyncio
    async def test_large_message_throughput(
        self,
        event_bus: InMemoryEventBus,
        large_message_bytes: bytes,
    ) -> None:
        """Test throughput with larger (~1KB) messages.

        Validates that message size doesn't significantly impact throughput.

        Performance Target:
            > 500 events/sec with 1KB messages
        """
        topic = generate_unique_topic()

        start = time.perf_counter()
        for i in range(1000):
            await event_bus.publish(
                topic=topic,
                key=f"key-{i}".encode(),
                value=large_message_bytes,
            )
        elapsed = time.perf_counter() - start

        events_per_sec = 1000 / elapsed

        assert events_per_sec > 500, (
            f"Large message throughput {events_per_sec:.0f} events/sec, expected > 500"
        )

        print("\nLarge Message Throughput (1KB x 1000):")
        print(f"  Duration:   {elapsed:.3f}s")
        print(f"  Throughput: {events_per_sec:.0f} events/sec")
        print(f"  Data rate:  {events_per_sec / 1024:.2f} MB/sec")


# -----------------------------------------------------------------------------
# Concurrent Publisher Tests
# -----------------------------------------------------------------------------


class TestConcurrentPublishers:
    """Test concurrent publisher throughput."""

    @pytest.mark.asyncio
    async def test_10_concurrent_publishers(
        self,
        event_bus: InMemoryEventBus,
        sample_message_bytes: bytes,
    ) -> None:
        """Test 10 concurrent publishers each publishing 100 messages.

        Validates that concurrent publishing works correctly
        and achieves good aggregate throughput.

        Performance Target:
            > 2000 events/sec aggregate
        """
        topic = generate_unique_topic()
        num_publishers = 10
        msgs_per_publisher = 100

        async def publisher(publisher_id: int) -> int:
            count = 0
            for i in range(msgs_per_publisher):
                await event_bus.publish(
                    topic=topic,
                    key=f"pub-{publisher_id}-{i}".encode(),
                    value=sample_message_bytes,
                )
                count += 1
            return count

        start = time.perf_counter()
        results = await asyncio.gather(*[publisher(i) for i in range(num_publishers)])
        elapsed = time.perf_counter() - start

        total_messages = sum(results)
        events_per_sec = total_messages / elapsed

        # Verify all published
        assert total_messages == num_publishers * msgs_per_publisher
        offset = await event_bus.get_topic_offset(topic)
        assert offset == total_messages

        # Performance assertion
        assert events_per_sec > 2000, (
            f"Concurrent throughput {events_per_sec:.0f} events/sec, expected > 2000"
        )

        print(f"\nConcurrent Publishers ({num_publishers} x {msgs_per_publisher}):")
        print(f"  Total:      {total_messages} messages")
        print(f"  Duration:   {elapsed:.3f}s")
        print(f"  Throughput: {events_per_sec:.0f} events/sec")

    @pytest.mark.asyncio
    @pytest.mark.xfail(
        reason="Environment-dependent throughput: CI runners may not achieve 5000 events/sec "
        "aggregate due to variable CPU/memory resources. Test still runs to capture metrics.",
        strict=False,
    )
    async def test_50_concurrent_publishers(
        self,
        high_volume_event_bus: InMemoryEventBus,
        sample_message_bytes: bytes,
    ) -> None:
        """Test 50 concurrent publishers (stress test).

        Higher concurrency stress test to validate lock contention
        and memory safety under heavy concurrent load.

        Performance Target:
            > 5000 events/sec aggregate
        """
        topic = generate_unique_topic()
        num_publishers = 50
        msgs_per_publisher = 200

        async def publisher(publisher_id: int) -> int:
            count = 0
            for i in range(msgs_per_publisher):
                await high_volume_event_bus.publish(
                    topic=topic,
                    key=f"pub-{publisher_id}-{i}".encode(),
                    value=sample_message_bytes,
                )
                count += 1
            return count

        start = time.perf_counter()
        results = await asyncio.gather(*[publisher(i) for i in range(num_publishers)])
        elapsed = time.perf_counter() - start

        total_messages = sum(results)
        events_per_sec = total_messages / elapsed

        assert total_messages == num_publishers * msgs_per_publisher
        assert events_per_sec > 5000, (
            f"High concurrency throughput {events_per_sec:.0f} events/sec"
        )

        print(
            f"\nHigh Concurrency Publishers ({num_publishers} x {msgs_per_publisher}):"
        )
        print(f"  Total:      {total_messages} messages")
        print(f"  Duration:   {elapsed:.3f}s")
        print(f"  Throughput: {events_per_sec:.0f} events/sec")

    @pytest.mark.asyncio
    async def test_concurrent_multi_topic(
        self,
        event_bus: InMemoryEventBus,
        sample_message_bytes: bytes,
    ) -> None:
        """Test concurrent publishing to multiple topics.

        Validates that concurrent publishing to different topics
        doesn't cause contention issues.

        Performance Target:
            > 3000 events/sec across topics
        """
        num_topics = 5
        msgs_per_topic = 200
        topics = [generate_unique_topic() for _ in range(num_topics)]

        async def topic_publisher(topic: str) -> int:
            count = 0
            for i in range(msgs_per_topic):
                await event_bus.publish(
                    topic=topic,
                    key=f"key-{i}".encode(),
                    value=sample_message_bytes,
                )
                count += 1
            return count

        start = time.perf_counter()
        results = await asyncio.gather(*[topic_publisher(topic) for topic in topics])
        elapsed = time.perf_counter() - start

        total_messages = sum(results)
        events_per_sec = total_messages / elapsed

        assert total_messages == num_topics * msgs_per_topic
        assert events_per_sec > 3000, (
            f"Multi-topic throughput {events_per_sec:.0f} events/sec"
        )

        print(f"\nMulti-Topic Publishers ({num_topics} topics x {msgs_per_topic}):")
        print(f"  Total:      {total_messages} messages")
        print(f"  Duration:   {elapsed:.3f}s")
        print(f"  Throughput: {events_per_sec:.0f} events/sec")


# -----------------------------------------------------------------------------
# Publisher with Subscriber Throughput Tests
# -----------------------------------------------------------------------------


class TestPublishWithSubscribers:
    """Test publishing throughput with active subscribers."""

    @pytest.mark.asyncio
    async def test_throughput_with_single_subscriber(
        self,
        event_bus: InMemoryEventBus,
        sample_message_bytes: bytes,
    ) -> None:
        """Test throughput with one active subscriber.

        Validates that subscriber callbacks don't significantly
        impact publishing throughput.

        Performance Target:
            > 500 events/sec with subscriber processing
        """
        topic = generate_unique_topic()
        received_count = 0
        lock = asyncio.Lock()

        async def handler(msg: ModelEventMessage) -> None:
            nonlocal received_count
            async with lock:
                received_count += 1

        await event_bus.subscribe(topic, "perf-group", handler)

        start = time.perf_counter()
        for i in range(1000):
            await event_bus.publish(
                topic=topic,
                key=f"key-{i}".encode(),
                value=sample_message_bytes,
            )
        elapsed = time.perf_counter() - start

        events_per_sec = 1000 / elapsed

        # All messages should be delivered
        assert received_count == 1000

        assert events_per_sec > 500, (
            f"Throughput with subscriber {events_per_sec:.0f} events/sec"
        )

        print("\nThroughput with Subscriber (1000 events):")
        print(f"  Received:   {received_count}")
        print(f"  Duration:   {elapsed:.3f}s")
        print(f"  Throughput: {events_per_sec:.0f} events/sec")

    @pytest.mark.asyncio
    async def test_throughput_with_multiple_subscribers(
        self,
        event_bus: InMemoryEventBus,
        sample_message_bytes: bytes,
    ) -> None:
        """Test throughput with multiple subscribers on same topic.

        Validates that multiple subscribers scale reasonably.

        Performance Target:
            > 200 events/sec with 5 subscribers
        """
        topic = generate_unique_topic()
        counters: list[int] = [0] * 5
        locks = [asyncio.Lock() for _ in range(5)]

        def make_handler(
            index: int,
        ) -> Callable[[ModelEventMessage], Awaitable[None]]:
            async def handler(msg: ModelEventMessage) -> None:
                async with locks[index]:
                    counters[index] += 1

            return handler

        for i in range(5):
            await event_bus.subscribe(topic, f"group-{i}", make_handler(i))

        start = time.perf_counter()
        for i in range(500):
            await event_bus.publish(
                topic=topic,
                key=f"key-{i}".encode(),
                value=sample_message_bytes,
            )
        elapsed = time.perf_counter() - start

        events_per_sec = 500 / elapsed

        # All subscribers should receive all messages
        assert all(c == 500 for c in counters)

        assert events_per_sec > 200, (
            f"Multi-subscriber throughput {events_per_sec:.0f} events/sec"
        )

        print("\nThroughput with 5 Subscribers (500 events):")
        print(f"  Per subscriber: {counters}")
        print(f"  Duration:       {elapsed:.3f}s")
        print(f"  Throughput:     {events_per_sec:.0f} events/sec")

    @pytest.mark.asyncio
    async def test_fan_out_throughput(
        self,
        event_bus: InMemoryEventBus,
        sample_message_bytes: bytes,
    ) -> None:
        """Test fan-out pattern with many subscribers.

        Measures overhead of delivering to 10 subscribers per message.

        Performance Target:
            > 100 events/sec with 10 subscribers (1000 deliveries/sec)
        """
        topic = generate_unique_topic()
        total_received = 0
        lock = asyncio.Lock()

        async def handler(msg: ModelEventMessage) -> None:
            nonlocal total_received
            async with lock:
                total_received += 1

        # Subscribe 10 handlers
        for i in range(10):
            await event_bus.subscribe(topic, f"fanout-{i}", handler)

        start = time.perf_counter()
        for i in range(200):
            await event_bus.publish(
                topic=topic,
                key=f"key-{i}".encode(),
                value=sample_message_bytes,
            )
        elapsed = time.perf_counter() - start

        events_per_sec = 200 / elapsed
        deliveries_per_sec = total_received / elapsed

        # Each of 10 subscribers should receive 200 messages
        assert total_received == 2000

        print("\nFan-out Throughput (10 subscribers x 200 events):")
        print(f"  Total deliveries: {total_received}")
        print(f"  Duration:         {elapsed:.3f}s")
        print(f"  Publish rate:     {events_per_sec:.0f} events/sec")
        print(f"  Delivery rate:    {deliveries_per_sec:.0f} deliveries/sec")
