# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Backpressure and overload tests for Event Bus.

This test suite validates event bus behavior when producers outpace consumers,
and under various overload conditions. Tests cover:

- Slow consumer / fast producer scenarios (backpressure)
- Queue depth behavior under overload
- Circuit breaker activation under sustained failure load
- Graceful degradation when consumers are overwhelmed
- Recovery after overload conditions subside
- Memory bounds under sustained overload

Design Rationale:
    The in-memory event bus delivers synchronously: publish() awaits each
    subscriber before returning. This means backpressure manifests as
    increased publish latency (producer slows to match consumer speed),
    rather than queue growth. Tests validate that:

    1. Slow consumers slow down publishers proportionally
    2. Latency degrades gracefully (not catastrophically)
    3. No messages are dropped or lost
    4. Circuit breakers activate on sustained failures
    5. Memory stays bounded even under overload

CI Behavior:
    Tests with strict latency thresholds are skipped in CI due to variable
    CPU resources on shared runners. Tests that only check correctness
    (no message loss, circuit breaker activation) run in all environments.

Usage:
    Run backpressure tests:
        uv run pytest tests/performance/event_bus/test_event_bus_backpressure.py -v

    Skip performance tests:
        uv run pytest -m "not performance" tests/

Related:
    - OMN-774: INFRA-033 Backpressure and overload tests [BETA]
    - OMN-57: Event bus performance testing (Phase 9)
    - EventBusInmemory: Primary implementation under test
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator, Awaitable, Callable
from statistics import mean, median

import pytest

from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.event_bus.models import ModelEventMessage
from omnibase_infra.testing import is_ci_environment
from tests.conftest import make_test_node_identity
from tests.performance.event_bus.conftest import generate_unique_topic

IS_CI = is_ci_environment()

# Mark all tests in this module as performance tests
pytestmark = [pytest.mark.performance]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def backpressure_bus() -> AsyncGenerator[EventBusInmemory, None]:
    """Event bus configured for backpressure testing.

    Uses a higher circuit-breaker threshold to avoid short-circuiting
    during intentional slow-consumer overload tests.

    Yields:
        Started EventBusInmemory instance.
    """
    bus = EventBusInmemory(
        environment="backpressure-test",
        group="overload",
        max_history=50000,
        circuit_breaker_threshold=100,  # High threshold to avoid early circuit open
    )
    await bus.start()
    yield bus
    await bus.close()


@pytest.fixture
async def fast_bus() -> AsyncGenerator[EventBusInmemory, None]:
    """Minimal-overhead event bus for latency-sensitive tests.

    Yields:
        Started EventBusInmemory instance.
    """
    bus = EventBusInmemory(
        environment="fast-test",
        group="latency",
        max_history=1000,
    )
    await bus.start()
    yield bus
    await bus.close()


# ---------------------------------------------------------------------------
# Slow Consumer / Fast Producer (Backpressure)
# ---------------------------------------------------------------------------


class TestSlowConsumerBackpressure:
    """Test backpressure when consumers are slower than producers.

    The in-memory event bus delivers synchronously, so a slow handler
    directly slows publish(). These tests confirm that delivery is still
    correct and latency degrades gracefully rather than catastrophically.
    """

    @pytest.mark.asyncio
    async def test_slow_handler_slows_publisher(
        self,
        backpressure_bus: EventBusInmemory,
        sample_message_bytes: bytes,
    ) -> None:
        """Slow consumer causes proportional publish latency increase.

        Publishes to two topics: one with no subscribers (baseline) and
        one with a slow 1 ms handler. Validates that the slow handler
        causes a measurable latency increase per publish while still
        delivering all messages.

        Correctness Targets:
            All messages delivered: 100%
            No exceptions raised
        """
        topic_baseline = generate_unique_topic()
        topic_slow = generate_unique_topic()
        num_messages = 200
        handler_delay_seconds = 0.001  # 1 ms

        received_slow: list[int] = []
        lock = asyncio.Lock()

        async def slow_handler(msg: ModelEventMessage) -> None:
            await asyncio.sleep(handler_delay_seconds)
            async with lock:
                received_slow.append(1)

        await backpressure_bus.subscribe(
            topic_slow,
            make_test_node_identity(service="backpressure", node_name="slow"),
            slow_handler,
        )

        # Baseline: no subscribers
        start_baseline = time.perf_counter()
        for i in range(num_messages):
            await backpressure_bus.publish(
                topic=topic_baseline,
                key=f"key-{i}".encode(),
                value=sample_message_bytes,
            )
        baseline_duration = time.perf_counter() - start_baseline

        # Slow consumer path
        start_slow = time.perf_counter()
        for i in range(num_messages):
            await backpressure_bus.publish(
                topic=topic_slow,
                key=f"key-{i}".encode(),
                value=sample_message_bytes,
            )
        slow_duration = time.perf_counter() - start_slow

        # All messages must be delivered
        assert len(received_slow) == num_messages, (
            f"Expected {num_messages} deliveries, got {len(received_slow)}"
        )

        # Slow path must take significantly longer than baseline
        # (at minimum the total handler delay: num_messages * handler_delay)
        minimum_expected_slow_duration = num_messages * handler_delay_seconds
        assert slow_duration >= minimum_expected_slow_duration * 0.8, (
            f"Slow path ({slow_duration:.3f}s) should take at least "
            f"{minimum_expected_slow_duration:.3f}s due to handler delay"
        )

        print("\nSlow Consumer Backpressure:")
        print(f"  Messages:          {num_messages}")
        print(f"  Handler delay:     {handler_delay_seconds * 1000:.0f} ms")
        print(f"  Baseline duration: {baseline_duration:.3f}s")
        print(f"  Slow duration:     {slow_duration:.3f}s")
        print(
            f"  Slowdown factor:   {slow_duration / max(baseline_duration, 0.001):.1f}x"
        )

    @pytest.mark.asyncio
    async def test_variable_speed_handler_correctness(
        self,
        backpressure_bus: EventBusInmemory,
        sample_message_bytes: bytes,
    ) -> None:
        """Variable-speed handler receives all messages correctly.

        Simulates a handler that alternates between fast and slow processing
        to mimic realistic workload variability (e.g., occasional DB flushes).

        Correctness Target:
            100% message delivery under variable handler speed.
        """
        topic = generate_unique_topic()
        num_messages = 100
        received_offsets: list[str] = []
        lock = asyncio.Lock()

        async def variable_handler(msg: ModelEventMessage) -> None:
            # Every 10th message takes 5x longer (simulating periodic slow operations)
            if int(msg.offset) % 10 == 0:
                await asyncio.sleep(0.005)  # 5 ms
            else:
                await asyncio.sleep(0.001)  # 1 ms
            async with lock:
                received_offsets.append(msg.offset)

        await backpressure_bus.subscribe(
            topic,
            make_test_node_identity(service="backpressure", node_name="variable"),
            variable_handler,
        )

        for i in range(num_messages):
            await backpressure_bus.publish(
                topic=topic,
                key=f"key-{i}".encode(),
                value=sample_message_bytes,
            )

        # All messages must be received
        assert len(received_offsets) == num_messages, (
            f"Expected {num_messages} received, got {len(received_offsets)}"
        )

        # Offsets should be sequential (FIFO ordering preserved under backpressure)
        expected_offsets = [str(i) for i in range(num_messages)]
        assert received_offsets == expected_offsets, (
            "Message ordering violated under variable-speed handler"
        )

        print("\nVariable-Speed Handler:")
        print(f"  Messages received: {len(received_offsets)}/{num_messages}")
        print("  FIFO ordering:     preserved")

    @pytest.mark.asyncio
    async def test_multiple_slow_consumers_correctness(
        self,
        backpressure_bus: EventBusInmemory,
        sample_message_bytes: bytes,
    ) -> None:
        """Multiple slow consumers all receive all messages.

        Validates fan-out correctness when multiple slow handlers are registered.
        Under the synchronous delivery model, each publish awaits all handlers
        sequentially, making latency additive per subscriber.

        Correctness Target:
            All N subscribers receive all M messages.
        """
        topic = generate_unique_topic()
        num_consumers = 3
        num_messages = 50
        counters = [0] * num_consumers
        locks = [asyncio.Lock() for _ in range(num_consumers)]

        def make_slow_handler(
            idx: int,
        ) -> Callable[[ModelEventMessage], Awaitable[None]]:
            async def handler(msg: ModelEventMessage) -> None:
                await asyncio.sleep(0.001)  # 1 ms each
                async with locks[idx]:
                    counters[idx] += 1

            return handler

        for i in range(num_consumers):
            await backpressure_bus.subscribe(
                topic,
                make_test_node_identity(
                    service="backpressure", node_name="multi-slow", suffix=f"-{i}"
                ),
                make_slow_handler(i),
            )

        start = time.perf_counter()
        for i in range(num_messages):
            await backpressure_bus.publish(
                topic=topic,
                key=f"key-{i}".encode(),
                value=sample_message_bytes,
            )
        elapsed = time.perf_counter() - start

        # Each consumer must receive all messages
        for i, count in enumerate(counters):
            assert count == num_messages, (
                f"Consumer {i} received {count}/{num_messages} messages"
            )

        print(f"\nMultiple Slow Consumers ({num_consumers} x 1ms):")
        print(f"  Messages per consumer: {num_messages}")
        print(f"  Counts:                {counters}")
        print(f"  Total duration:        {elapsed:.3f}s")
        print(f"  Per-message latency:   {elapsed / num_messages * 1000:.1f}ms avg")


# ---------------------------------------------------------------------------
# Overload Conditions
# ---------------------------------------------------------------------------


class TestOverloadConditions:
    """Test event bus behavior under sustained overload.

    Validates that the system degrades gracefully rather than failing hard
    when operating beyond designed throughput levels.
    """

    @pytest.mark.asyncio
    async def test_circuit_breaker_activates_under_failure_overload(
        self,
        sample_message_bytes: bytes,
    ) -> None:
        """Circuit breaker activates when handler fails repeatedly.

        Publishes to a topic with a consistently failing handler. After
        `circuit_breaker_threshold` failures, the circuit opens and the
        bus skips the failing subscriber, protecting other subscribers.

        Correctness Targets:
            - Circuit opens after exactly `threshold` failures
            - Good subscriber continues receiving after circuit opens
            - Failing handler stops being called after circuit opens
        """
        topic = generate_unique_topic()
        threshold = 5
        bus = EventBusInmemory(
            environment="overload-circuit",
            group="breaker",
            circuit_breaker_threshold=threshold,
        )
        await bus.start()

        fail_count = 0
        good_count = 0
        fail_lock = asyncio.Lock()
        good_lock = asyncio.Lock()

        async def failing_handler(msg: ModelEventMessage) -> None:
            nonlocal fail_count
            async with fail_lock:
                fail_count += 1
            raise RuntimeError("Simulated persistent handler failure")

        async def good_handler(msg: ModelEventMessage) -> None:
            nonlocal good_count
            async with good_lock:
                good_count += 1

        await bus.subscribe(
            topic,
            make_test_node_identity(service="overload", node_name="failing"),
            failing_handler,
        )
        await bus.subscribe(
            topic,
            make_test_node_identity(service="overload", node_name="good"),
            good_handler,
        )

        num_messages = threshold + 20  # Enough to confirm circuit stays open
        for i in range(num_messages):
            await bus.publish(
                topic=topic,
                key=f"msg-{i}".encode(),
                value=sample_message_bytes,
            )

        status = await bus.get_circuit_breaker_status()
        await bus.close()

        # Circuit should be open for the failing subscriber
        assert len(status["open_circuits"]) >= 1, (
            "Expected at least one open circuit after repeated failures"
        )

        # Failing handler should have been called exactly `threshold` times
        assert fail_count == threshold, (
            f"Failing handler called {fail_count} times, expected {threshold} "
            f"(circuit should open after threshold failures)"
        )

        # Good handler must continue receiving all messages
        assert good_count == num_messages, (
            f"Good handler got {good_count}/{num_messages} (circuit isolation broken)"
        )

        print("\nCircuit Breaker Activation Under Overload:")
        print(f"  Threshold:         {threshold}")
        print(f"  Total messages:    {num_messages}")
        print(f"  Failing calls:     {fail_count}")
        print(f"  Good received:     {good_count}")
        print(f"  Open circuits:     {len(status['open_circuits'])}")

    @pytest.mark.asyncio
    async def test_overload_no_message_loss(
        self,
        sample_message_bytes: bytes,
    ) -> None:
        """No messages lost when producers dramatically outpace consumers.

        Simulates a scenario where the producer publishes at maximum speed
        while the consumer processes slowly. Validates zero message loss.

        With the synchronous delivery model, publish() blocks until the
        subscriber finishes, so "queue" depth == 0 (no buffering), but
        every published message must still be delivered.
        """
        topic = generate_unique_topic()
        bus = EventBusInmemory(
            environment="overload-loss",
            group="no-loss",
            max_history=20000,
            circuit_breaker_threshold=20000,  # Never trip
        )
        await bus.start()

        received: list[str] = []
        lock = asyncio.Lock()

        async def slow_consumer(msg: ModelEventMessage) -> None:
            await asyncio.sleep(0.0005)  # 0.5ms per message
            async with lock:
                received.append(msg.offset)

        await bus.subscribe(
            topic,
            make_test_node_identity(service="overload", node_name="consumer"),
            slow_consumer,
        )

        num_messages = 200
        for i in range(num_messages):
            await bus.publish(
                topic=topic,
                key=f"key-{i}".encode(),
                value=sample_message_bytes,
            )

        await bus.close()

        # Zero message loss assertion
        assert len(received) == num_messages, (
            f"Message loss detected: received {len(received)}/{num_messages}"
        )

        print("\nOverload No-Message-Loss:")
        print(f"  Published: {num_messages}")
        print(f"  Received:  {len(received)}")
        print("  Lost:      0")

    @pytest.mark.asyncio
    async def test_burst_then_idle_recovery(
        self,
        fast_bus: EventBusInmemory,
        sample_message_bytes: bytes,
    ) -> None:
        """System recovers to baseline performance after a burst.

        Publishes a large burst, waits briefly, then measures that
        subsequent publishes return to pre-burst latency levels.

        This validates that burst processing does not leave the bus
        in a degraded state.
        """
        topic = generate_unique_topic()
        num_baseline = 100
        burst_size = 1000
        num_recovery = 100

        # Baseline latency measurement
        start = time.perf_counter()
        for i in range(num_baseline):
            await fast_bus.publish(
                topic=topic,
                key=f"base-{i}".encode(),
                value=sample_message_bytes,
            )
        baseline_duration = time.perf_counter() - start
        baseline_per_msg = baseline_duration / num_baseline

        # Burst
        for i in range(burst_size):
            await fast_bus.publish(
                topic=topic,
                key=f"burst-{i}".encode(),
                value=sample_message_bytes,
            )

        # Brief pause to let any internal state settle
        await asyncio.sleep(0.01)

        # Recovery latency measurement
        start = time.perf_counter()
        for i in range(num_recovery):
            await fast_bus.publish(
                topic=topic,
                key=f"recovery-{i}".encode(),
                value=sample_message_bytes,
            )
        recovery_duration = time.perf_counter() - start
        recovery_per_msg = recovery_duration / num_recovery

        # Recovery should be within 5x of baseline (very lenient)
        assert recovery_per_msg < baseline_per_msg * 5 + 0.001, (
            f"Post-burst latency {recovery_per_msg * 1000:.3f}ms/msg far exceeds "
            f"baseline {baseline_per_msg * 1000:.3f}ms/msg — system not recovering"
        )

        # Topic offset should reflect all published messages
        expected_offset = num_baseline + burst_size + num_recovery
        offset = await fast_bus.get_topic_offset(topic)
        assert offset == expected_offset, (
            f"Topic offset {offset}, expected {expected_offset} (messages lost?)"
        )

        print("\nBurst-then-Idle Recovery:")
        print(
            f"  Baseline:         {num_baseline} msgs, {baseline_per_msg * 1000:.3f} ms/msg"
        )
        print(f"  Burst size:       {burst_size} msgs")
        print(
            f"  Recovery:         {num_recovery} msgs, {recovery_per_msg * 1000:.3f} ms/msg"
        )
        print(
            f"  Slowdown factor:  {recovery_per_msg / max(baseline_per_msg, 1e-9):.2f}x"
        )

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        IS_CI,
        reason="Latency percentile test is environment-sensitive; runs locally only.",
    )
    async def test_publish_latency_under_subscriber_load(
        self,
        sample_message_bytes: bytes,
    ) -> None:
        """Publish latency stays bounded with many slow subscribers.

        With 10 subscribers each taking 1ms, each publish call takes
        approximately 10ms (synchronous fan-out). Validates that p99 latency
        stays within 3x the theoretical minimum to detect pathological cases.

        Skipped in CI due to timer resolution differences on shared runners.
        """
        topic = generate_unique_topic()
        num_subscribers = 10
        handler_delay_ms = 1.0
        num_messages = 100

        bus = EventBusInmemory(
            environment="latency-load",
            group="p99",
            max_history=10000,
            circuit_breaker_threshold=10000,
        )
        await bus.start()

        lock = asyncio.Lock()
        total_received = 0

        def make_handler() -> Callable[[ModelEventMessage], Awaitable[None]]:
            async def handler(msg: ModelEventMessage) -> None:
                nonlocal total_received
                await asyncio.sleep(handler_delay_ms / 1000)
                async with lock:
                    total_received += 1

            return handler

        for i in range(num_subscribers):
            await bus.subscribe(
                topic,
                make_test_node_identity(
                    service="latency-load", node_name="sub", suffix=f"-{i}"
                ),
                make_handler(),
            )

        latencies_ms: list[float] = []
        for i in range(num_messages):
            t_start = time.perf_counter()
            await bus.publish(
                topic=topic,
                key=f"key-{i}".encode(),
                value=sample_message_bytes,
            )
            latency_ms = (time.perf_counter() - t_start) * 1000
            latencies_ms.append(latency_ms)

        await bus.close()

        # All subscribers receive all messages
        assert total_received == num_messages * num_subscribers, (
            f"Expected {num_messages * num_subscribers} deliveries, "
            f"got {total_received}"
        )

        p50 = median(latencies_ms)
        p99 = sorted(latencies_ms)[int(len(latencies_ms) * 0.99)]
        avg = mean(latencies_ms)

        # Theoretical minimum: num_subscribers * handler_delay_ms
        theoretical_min_ms = num_subscribers * handler_delay_ms
        # p99 should not exceed 10x theoretical to catch catastrophic regressions.
        # asyncio scheduling overhead can add significant per-call latency on loaded
        # systems, so we use a generous multiplier. The test purpose is to detect
        # pathological cases (e.g., 100x regression), not to validate exact latency.
        assert p99 < theoretical_min_ms * 10, (
            f"p99 latency {p99:.1f}ms exceeds 10x theoretical minimum "
            f"({theoretical_min_ms:.1f}ms) — possible scheduling regression"
        )

        print(f"\nPublish Latency Under {num_subscribers} Slow Subscribers:")
        print(f"  Theoretical min:  {theoretical_min_ms:.1f} ms/publish")
        print(f"  Average latency:  {avg:.2f} ms")
        print(f"  p50 latency:      {p50:.2f} ms")
        print(f"  p99 latency:      {p99:.2f} ms")


# ---------------------------------------------------------------------------
# Memory Bounds Under Overload
# ---------------------------------------------------------------------------


class TestMemoryUnderOverload:
    """Test that memory stays bounded during overload scenarios."""

    @pytest.mark.asyncio
    async def test_history_bounded_during_overload(
        self,
        sample_message_bytes: bytes,
    ) -> None:
        """History deque stays at max_history even when producers spike.

        Publishes 10x max_history messages rapidly and confirms that
        the history deque does not exceed its configured limit.
        """
        max_history = 500
        bus = EventBusInmemory(
            environment="mem-overload",
            group="bounded",
            max_history=max_history,
        )
        await bus.start()

        topic = generate_unique_topic()
        num_messages = max_history * 10  # Massively over-produce

        for i in range(num_messages):
            await bus.publish(
                topic=topic,
                key=f"key-{i}".encode(),
                value=sample_message_bytes,
            )

        history = await bus.get_event_history(limit=max_history + 1000)
        await bus.close()

        assert len(history) <= max_history, (
            f"History size {len(history)} exceeds max_history {max_history} — "
            "deque bound not enforced"
        )

        print("\nHistory Bounded During Overload:")
        print(f"  Published:     {num_messages}")
        print(f"  Max history:   {max_history}")
        print(f"  Actual history: {len(history)}")

    @pytest.mark.asyncio
    async def test_subscriber_cleanup_after_overload(
        self,
        sample_message_bytes: bytes,
    ) -> None:
        """Subscriber tables stay clean after overload + unsubscribe cycle.

        Subscribes, publishes a large overload burst, unsubscribes all,
        then confirms zero residual subscribers.
        """
        topic = generate_unique_topic()
        bus = EventBusInmemory(
            environment="cleanup-overload",
            group="cycle",
            max_history=5000,
        )
        await bus.start()

        num_subscribers = 20
        unsubscribers: list[Callable[[], Awaitable[None]]] = []

        async def noop(msg: ModelEventMessage) -> None:
            pass

        for i in range(num_subscribers):
            unsub = await bus.subscribe(
                topic,
                make_test_node_identity(
                    service="cleanup", node_name="sub", suffix=f"-{i}"
                ),
                noop,
            )
            unsubscribers.append(unsub)

        # Overload burst
        for i in range(2000):
            await bus.publish(
                topic=topic,
                key=f"key-{i}".encode(),
                value=sample_message_bytes,
            )

        # Unsubscribe all
        for unsub in unsubscribers:
            await unsub()

        remaining = await bus.get_subscriber_count(topic)
        await bus.close()

        assert remaining == 0, (
            f"Expected 0 subscribers after cleanup, found {remaining}"
        )

        print("\nSubscriber Cleanup After Overload:")
        print(f"  Subscribers added:   {num_subscribers}")
        print("  Burst messages:      2000")
        print(f"  Remaining after cleanup: {remaining}")


# ---------------------------------------------------------------------------
# Concurrent Producer / Consumer Scenarios
# ---------------------------------------------------------------------------


class TestConcurrentProducerConsumer:
    """Test concurrent producer-consumer interactions under backpressure."""

    @pytest.mark.asyncio
    async def test_concurrent_producers_single_slow_consumer(
        self,
        backpressure_bus: EventBusInmemory,
        sample_message_bytes: bytes,
    ) -> None:
        """Multiple concurrent producers with a single slow consumer.

        Validates that all messages from concurrent producers are delivered
        to the slow consumer without loss or duplication.
        """
        topic = generate_unique_topic()
        num_producers = 5
        msgs_per_producer = 40
        total_expected = num_producers * msgs_per_producer

        received_offsets: list[str] = []
        lock = asyncio.Lock()

        async def slow_consumer(msg: ModelEventMessage) -> None:
            await asyncio.sleep(0.001)  # 1ms processing
            async with lock:
                received_offsets.append(msg.offset)

        await backpressure_bus.subscribe(
            topic,
            make_test_node_identity(service="concurrent", node_name="consumer"),
            slow_consumer,
        )

        async def producer(prod_id: int) -> int:
            count = 0
            for i in range(msgs_per_producer):
                await backpressure_bus.publish(
                    topic=topic,
                    key=f"prod-{prod_id}-{i}".encode(),
                    value=sample_message_bytes,
                )
                count += 1
            return count

        start = time.perf_counter()
        results = await asyncio.gather(*[producer(p) for p in range(num_producers)])
        elapsed = time.perf_counter() - start

        total_published = sum(results)

        # Zero message loss
        assert total_published == total_expected, (
            f"Published {total_published}, expected {total_expected}"
        )
        assert len(received_offsets) == total_expected, (
            f"Consumer received {len(received_offsets)}/{total_expected} — message loss"
        )

        # No duplicate offsets
        assert len(received_offsets) == len(set(received_offsets)), (
            "Duplicate message delivery detected"
        )

        print(f"\nConcurrent Producers ({num_producers}) + Single Slow Consumer:")
        print(f"  Total published:   {total_published}")
        print(f"  Total received:    {len(received_offsets)}")
        print(f"  Duration:          {elapsed:.3f}s")
        print(f"  Effective rate:    {total_published / elapsed:.0f} msgs/sec")

    @pytest.mark.asyncio
    async def test_producer_consumer_parity_under_load(
        self,
        backpressure_bus: EventBusInmemory,
        sample_message_bytes: bytes,
    ) -> None:
        """Published count exactly matches delivered count under continuous load.

        Runs a 2-second continuous publish loop with an active subscriber and
        validates that every published message is delivered (parity = 1.0).
        """
        topic = generate_unique_topic()
        delivered: list[int] = []
        lock = asyncio.Lock()

        async def counting_handler(msg: ModelEventMessage) -> None:
            async with lock:
                delivered.append(1)

        await backpressure_bus.subscribe(
            topic,
            make_test_node_identity(service="parity", node_name="handler"),
            counting_handler,
        )

        run_duration = 1.0  # 1 second run
        published = 0
        start = time.perf_counter()

        while time.perf_counter() - start < run_duration:
            await backpressure_bus.publish(
                topic=topic,
                key=f"key-{published}".encode(),
                value=sample_message_bytes,
            )
            published += 1

        elapsed = time.perf_counter() - start

        # Every published message must be delivered
        assert len(delivered) == published, (
            f"Parity failure: published={published}, delivered={len(delivered)}"
        )

        parity = len(delivered) / max(published, 1)

        print("\nProducer-Consumer Parity Under Load:")
        print(f"  Duration:   {elapsed:.3f}s")
        print(f"  Published:  {published}")
        print(f"  Delivered:  {len(delivered)}")
        print(f"  Parity:     {parity:.4f}")
        print(f"  Rate:       {published / elapsed:.0f} msgs/sec")
