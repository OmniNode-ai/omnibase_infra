# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Latency performance tests for Event Bus.

This test suite measures latency characteristics including:
- Publish operation latency (p50, p95, p99)
- End-to-end message delivery latency
- Latency under varying load conditions
- Latency distribution analysis

Performance Thresholds:
    Target latencies are intentionally lenient for CI environments.
    Adjust for dedicated performance testing infrastructure.

    - Publish p95: < 100ms
    - End-to-end p95: < 100ms
    - Publish p99: < 200ms

Usage:
    Run latency tests:
        poetry run pytest tests/performance/event_bus/test_event_bus_latency.py -v

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
from statistics import mean, median, quantiles, stdev

import pytest

from omnibase_infra.event_bus.inmemory_event_bus import InMemoryEventBus
from omnibase_infra.event_bus.models import ModelEventHeaders, ModelEventMessage
from tests.performance.event_bus.conftest import generate_unique_topic

# -----------------------------------------------------------------------------
# Publish Latency Tests
# -----------------------------------------------------------------------------


class TestPublishLatency:
    """Test publish operation latency characteristics."""

    @pytest.mark.asyncio
    async def test_publish_latency_distribution_1000(
        self,
        event_bus: InMemoryEventBus,
        sample_message_bytes: bytes,
    ) -> None:
        """Measure p50, p95, p99 publish latencies for 1000 operations.

        Collects timing data for 1000 publish operations and calculates
        latency percentiles to understand performance distribution.

        Performance Target:
            - p50: < 10ms
            - p95: < 50ms
            - p99: < 100ms
        """
        topic = generate_unique_topic()
        latencies: list[float] = []

        for i in range(1000):
            start = time.perf_counter()
            await event_bus.publish(
                topic=topic,
                key=f"key-{i}".encode(),
                value=sample_message_bytes,
            )
            latencies.append(time.perf_counter() - start)

        # Calculate statistics
        avg_latency = mean(latencies)
        std_latency = stdev(latencies)
        med_latency = median(latencies)

        # Calculate percentiles
        percentiles = quantiles(latencies, n=100)
        p50 = percentiles[49]
        p95 = percentiles[94]
        p99 = percentiles[98]

        # Performance assertions (in seconds)
        assert p50 < 0.01, f"p50 latency {p50 * 1000:.2f}ms, expected < 10ms"
        assert p95 < 0.05, f"p95 latency {p95 * 1000:.2f}ms, expected < 50ms"
        assert p99 < 0.10, f"p99 latency {p99 * 1000:.2f}ms, expected < 100ms"

        print("\nPublish Latency Distribution (1000 ops):")
        print(f"  Mean:   {avg_latency * 1000:.3f}ms")
        print(f"  Median: {med_latency * 1000:.3f}ms")
        print(f"  Std:    {std_latency * 1000:.3f}ms")
        print(f"  p50:    {p50 * 1000:.3f}ms")
        print(f"  p95:    {p95 * 1000:.3f}ms")
        print(f"  p99:    {p99 * 1000:.3f}ms")

    @pytest.mark.asyncio
    async def test_cold_vs_warm_publish_latency(
        self,
        sample_message_bytes: bytes,
    ) -> None:
        """Compare cold start vs warm publish latencies.

        Measures the first publish (cold) vs subsequent publishes (warm)
        to understand initialization overhead.
        """
        topic = generate_unique_topic()

        # Create fresh bus for cold start measurement
        cold_bus = InMemoryEventBus(environment="cold-test", group="latency")
        await cold_bus.start()

        # Cold publish (first operation)
        cold_start = time.perf_counter()
        await cold_bus.publish(topic=topic, key=b"cold", value=sample_message_bytes)
        cold_latency = time.perf_counter() - cold_start

        # Warm publishes (subsequent operations)
        warm_latencies: list[float] = []
        for i in range(100):
            start = time.perf_counter()
            await cold_bus.publish(
                topic=topic,
                key=f"warm-{i}".encode(),
                value=sample_message_bytes,
            )
            warm_latencies.append(time.perf_counter() - start)

        await cold_bus.close()

        avg_warm = mean(warm_latencies)

        # Cold should not be dramatically slower (< 50x warm)
        # Note: In CI environments, cold start can be significantly impacted
        # by first-time asyncio loop initialization, dict creation, etc.
        # This is a lenient threshold to avoid flaky tests in CI.
        ratio = cold_latency / avg_warm if avg_warm > 0 else 1.0
        assert ratio < 50, f"Cold/warm ratio {ratio:.1f}x, expected < 50x"

        print("\nCold vs Warm Latency:")
        print(f"  Cold:  {cold_latency * 1000:.3f}ms")
        print(f"  Warm:  {avg_warm * 1000:.3f}ms (avg of 100)")
        print(f"  Ratio: {ratio:.1f}x")

    @pytest.mark.asyncio
    async def test_publish_latency_with_headers(
        self,
        event_bus: InMemoryEventBus,
        sample_message_bytes: bytes,
        sample_headers: ModelEventHeaders,
    ) -> None:
        """Test publish latency with custom headers.

        Validates that header processing doesn't add significant latency.
        """
        topic = generate_unique_topic()

        # Without headers
        no_header_latencies: list[float] = []
        for i in range(500):
            start = time.perf_counter()
            await event_bus.publish(
                topic=topic,
                key=f"no-header-{i}".encode(),
                value=sample_message_bytes,
            )
            no_header_latencies.append(time.perf_counter() - start)

        # With headers
        with_header_latencies: list[float] = []
        for i in range(500):
            start = time.perf_counter()
            await event_bus.publish(
                topic=topic,
                key=f"with-header-{i}".encode(),
                value=sample_message_bytes,
                headers=sample_headers,
            )
            with_header_latencies.append(time.perf_counter() - start)

        avg_no_header = mean(no_header_latencies)
        avg_with_header = mean(with_header_latencies)

        # Headers should add < 50% overhead
        overhead = (
            (avg_with_header - avg_no_header) / avg_no_header
            if avg_no_header > 0
            else 0
        )
        assert overhead < 0.5, f"Header overhead {overhead * 100:.1f}%, expected < 50%"

        print("\nHeader Impact on Latency:")
        print(f"  Without headers: {avg_no_header * 1000:.3f}ms")
        print(f"  With headers:    {avg_with_header * 1000:.3f}ms")
        print(f"  Overhead:        {overhead * 100:.1f}%")


# -----------------------------------------------------------------------------
# End-to-End Latency Tests
# -----------------------------------------------------------------------------


class TestEndToEndLatency:
    """Test end-to-end message delivery latency."""

    @pytest.mark.asyncio
    async def test_publish_to_receive_latency(
        self,
        event_bus: InMemoryEventBus,
        sample_message_bytes: bytes,
    ) -> None:
        """Measure publish-to-receive latency.

        Calculates the time from publish call to subscriber callback invocation.

        Performance Target:
            p95 end-to-end < 100ms
        """
        topic = generate_unique_topic()
        e2e_latencies: list[float] = []
        lock = asyncio.Lock()
        publish_times: dict[int, float] = {}

        async def handler(msg: ModelEventMessage) -> None:
            receive_time = time.perf_counter()
            # Extract index from key
            key_str = msg.key.decode() if msg.key else "0"
            index = int(key_str.split("-")[-1])
            async with lock:
                if index in publish_times:
                    e2e_latencies.append(receive_time - publish_times[index])

        await event_bus.subscribe(topic, "e2e-group", handler)

        # Publish with timing
        for i in range(1000):
            publish_times[i] = time.perf_counter()
            await event_bus.publish(
                topic=topic,
                key=f"msg-{i}".encode(),
                value=sample_message_bytes,
            )

        # Calculate statistics
        assert len(e2e_latencies) == 1000, f"Only {len(e2e_latencies)} received"

        avg_e2e = mean(e2e_latencies)
        percentiles = quantiles(e2e_latencies, n=100)
        p50 = percentiles[49]
        p95 = percentiles[94]
        p99 = percentiles[98]

        # Performance assertions
        assert p95 < 0.1, f"E2E p95 {p95 * 1000:.2f}ms, expected < 100ms"

        print("\nEnd-to-End Latency (1000 messages):")
        print(f"  Mean: {avg_e2e * 1000:.3f}ms")
        print(f"  p50:  {p50 * 1000:.3f}ms")
        print(f"  p95:  {p95 * 1000:.3f}ms")
        print(f"  p99:  {p99 * 1000:.3f}ms")

    @pytest.mark.asyncio
    @pytest.mark.xfail(
        reason="Flaky in CI: subscriber latency ratio varies with shared resources. "
        "Observed 31.1x ratio in CI vs expected <5x. Test provides value locally.",
        strict=False,
    )
    async def test_e2e_latency_with_multiple_subscribers(
        self,
        event_bus: InMemoryEventBus,
        sample_message_bytes: bytes,
    ) -> None:
        """Measure E2E latency with multiple subscribers.

        Validates that additional subscribers add predictable latency.
        """
        topic = generate_unique_topic()
        latencies_by_subscriber: dict[int, list[float]] = {i: [] for i in range(5)}
        locks = [asyncio.Lock() for _ in range(5)]
        publish_times: dict[int, float] = {}

        def make_handler(
            idx: int,
            locks_ref: list[asyncio.Lock],
            publish_times_ref: dict[int, float],
            latencies_ref: dict[int, list[float]],
        ) -> Callable[[ModelEventMessage], Awaitable[None]]:
            async def handler(msg: ModelEventMessage) -> None:
                receive_time = time.perf_counter()
                key_str = msg.key.decode() if msg.key else "0"
                msg_idx = int(key_str.split("-")[-1])
                async with locks_ref[idx]:
                    if msg_idx in publish_times_ref:
                        latencies_ref[idx].append(
                            receive_time - publish_times_ref[msg_idx]
                        )

            return handler

        for sub_idx in range(5):
            handler = make_handler(
                sub_idx, locks, publish_times, latencies_by_subscriber
            )
            await event_bus.subscribe(topic, f"multi-e2e-{sub_idx}", handler)

        # Publish messages
        for i in range(500):
            publish_times[i] = time.perf_counter()
            await event_bus.publish(
                topic=topic,
                key=f"msg-{i}".encode(),
                value=sample_message_bytes,
            )

        # Analyze per-subscriber latency
        print("\nE2E Latency by Subscriber Position (500 messages):")
        for idx in range(5):
            lats = latencies_by_subscriber[idx]
            if lats:
                avg = mean(lats)
                print(f"  Subscriber {idx}: {avg * 1000:.3f}ms avg ({len(lats)} msgs)")

        # First subscriber should always be fastest
        avg_first = (
            mean(latencies_by_subscriber[0]) if latencies_by_subscriber[0] else 0
        )
        avg_last = mean(latencies_by_subscriber[4]) if latencies_by_subscriber[4] else 0

        # Last subscriber should not be more than 5x slower than first
        if avg_first > 0:
            ratio = avg_last / avg_first
            assert ratio < 5, f"Last/first ratio {ratio:.1f}x, expected < 5x"

    @pytest.mark.asyncio
    @pytest.mark.xfail(
        reason="Flaky in CI: latency varies significantly with shared resources. "
        "Observed 113.7x degradation in CI vs expected <2x. Test provides value locally.",
        strict=False,
    )
    async def test_latency_consistency_over_time(
        self,
        event_bus: InMemoryEventBus,
        sample_message_bytes: bytes,
    ) -> None:
        """Test that latency remains consistent over extended operation.

        Runs multiple batches and compares latency between early and late batches
        to detect performance degradation.

        Note:
            This test is marked xfail for CI environments due to variable resource
            availability causing latency spikes >100x. The test still runs and
            provides value for local development where it should pass consistently.
        """
        topic = generate_unique_topic()
        batch_size = 200
        num_batches = 5
        batch_latencies: list[list[float]] = []

        for batch in range(num_batches):
            latencies: list[float] = []
            for i in range(batch_size):
                start = time.perf_counter()
                await event_bus.publish(
                    topic=topic,
                    key=f"batch-{batch}-{i}".encode(),
                    value=sample_message_bytes,
                )
                latencies.append(time.perf_counter() - start)
            batch_latencies.append(latencies)

        # Compare first and last batch
        first_avg = mean(batch_latencies[0])
        last_avg = mean(batch_latencies[-1])

        # Latency should not degrade more than 2x over time
        degradation = last_avg / first_avg if first_avg > 0 else 1.0
        assert degradation < 2.0, (
            f"Latency degradation {degradation:.1f}x, expected < 2x"
        )

        print("\nLatency Consistency Over Time:")
        for i, lats in enumerate(batch_latencies):
            avg = mean(lats)
            print(f"  Batch {i + 1}: {avg * 1000:.3f}ms avg")
        print(f"  Degradation: {degradation:.2f}x")


# -----------------------------------------------------------------------------
# Latency Under Load Tests
# -----------------------------------------------------------------------------


class TestLatencyUnderLoad:
    """Test latency characteristics under varying load conditions."""

    @pytest.mark.asyncio
    async def test_latency_vs_concurrency(
        self,
        sample_message_bytes: bytes,
    ) -> None:
        """Measure how latency changes with concurrent publishers.

        Tests with 1, 5, 10, and 20 concurrent publishers.
        """
        topic = generate_unique_topic()
        bus = InMemoryEventBus(environment="load-test", group="concurrency")
        await bus.start()

        concurrency_levels = [1, 5, 10, 20]
        results: dict[int, tuple[float, float]] = {}  # level -> (avg, p99)

        async def make_publisher(
            ops: int,
            conc: int,
            lock_ref: asyncio.Lock,
            latencies_ref: list[float],
            topic_ref: str,
            msg_bytes: bytes,
            bus_ref: InMemoryEventBus,
        ) -> Callable[[int], Awaitable[None]]:
            async def publisher(pub_id: int) -> None:
                for i in range(ops):
                    start = time.perf_counter()
                    await bus_ref.publish(
                        topic=topic_ref,
                        key=f"c{conc}-p{pub_id}-{i}".encode(),
                        value=msg_bytes,
                    )
                    latency = time.perf_counter() - start
                    async with lock_ref:
                        latencies_ref.append(latency)

            return publisher

        for concurrency in concurrency_levels:
            all_latencies: list[float] = []
            lock = asyncio.Lock()
            ops_per_publisher = 100

            publisher = await make_publisher(
                ops_per_publisher,
                concurrency,
                lock,
                all_latencies,
                topic,
                sample_message_bytes,
                bus,
            )

            await asyncio.gather(*[publisher(i) for i in range(concurrency)])

            avg = mean(all_latencies)
            p99 = quantiles(all_latencies, n=100)[98]
            results[concurrency] = (avg, p99)

        await bus.close()

        print("\nLatency vs Concurrency:")
        for level, (avg, p99) in results.items():
            print(
                f"  {level:2d} publishers: avg={avg * 1000:.3f}ms, p99={p99 * 1000:.3f}ms"
            )

        # p99 should stay under 100ms even at high concurrency
        assert results[20][1] < 0.1, (
            f"p99 at 20 concurrency: {results[20][1] * 1000:.1f}ms, expected < 100ms"
        )

    @pytest.mark.asyncio
    async def test_latency_with_history_pressure(
        self,
        sample_message_bytes: bytes,
    ) -> None:
        """Test latency when event history is near capacity.

        Validates that LRU eviction doesn't impact latency significantly.
        """
        topic = generate_unique_topic()

        # Create bus with small history to trigger eviction
        bus = InMemoryEventBus(
            environment="history-test",
            group="pressure",
            max_history=100,  # Small history
        )
        await bus.start()

        # Fill history first
        for i in range(100):
            await bus.publish(topic=topic, key=f"fill-{i}".encode(), value=b"fill")

        # Now measure latency with eviction happening
        eviction_latencies: list[float] = []
        for i in range(500):
            start = time.perf_counter()
            await bus.publish(
                topic=topic,
                key=f"evict-{i}".encode(),
                value=sample_message_bytes,
            )
            eviction_latencies.append(time.perf_counter() - start)

        await bus.close()

        avg = mean(eviction_latencies)
        p99 = quantiles(eviction_latencies, n=100)[98]

        # Eviction should be O(1), so latency should remain low
        assert p99 < 0.05, f"p99 with eviction: {p99 * 1000:.1f}ms, expected < 50ms"

        print("\nLatency Under History Pressure:")
        print(f"  Avg: {avg * 1000:.3f}ms")
        print(f"  p99: {p99 * 1000:.3f}ms")

    @pytest.mark.asyncio
    async def test_subscriber_processing_impact(
        self,
        event_bus: InMemoryEventBus,
        sample_message_bytes: bytes,
    ) -> None:
        """Measure impact of slow subscriber on publish latency.

        Uses a subscriber with artificial delay to understand backpressure.
        """
        topic_fast = generate_unique_topic()
        topic_slow = generate_unique_topic()

        # Fast handler
        async def fast_handler(msg: ModelEventMessage) -> None:
            pass  # No-op

        # Slow handler (1ms delay)
        async def slow_handler(msg: ModelEventMessage) -> None:
            await asyncio.sleep(0.001)

        await event_bus.subscribe(topic_fast, "fast-group", fast_handler)
        await event_bus.subscribe(topic_slow, "slow-group", slow_handler)

        # Measure fast topic latency
        fast_latencies: list[float] = []
        for i in range(200):
            start = time.perf_counter()
            await event_bus.publish(
                topic=topic_fast, key=f"fast-{i}".encode(), value=sample_message_bytes
            )
            fast_latencies.append(time.perf_counter() - start)

        # Measure slow topic latency
        slow_latencies: list[float] = []
        for i in range(200):
            start = time.perf_counter()
            await event_bus.publish(
                topic=topic_slow, key=f"slow-{i}".encode(), value=sample_message_bytes
            )
            slow_latencies.append(time.perf_counter() - start)

        avg_fast = mean(fast_latencies)
        avg_slow = mean(slow_latencies)

        # Slow subscriber should add ~1ms to latency
        overhead = avg_slow - avg_fast
        assert overhead < 0.01, (
            f"Slow subscriber overhead {overhead * 1000:.1f}ms, expected < 10ms"
        )

        print("\nSubscriber Processing Impact:")
        print(f"  Fast handler: {avg_fast * 1000:.3f}ms")
        print(f"  Slow handler: {avg_slow * 1000:.3f}ms")
        print(f"  Overhead:     {overhead * 1000:.3f}ms")
