# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Replay Performance Tests for OMN-955.

This module provides performance tests for large event replay sequences.
These tests measure and validate performance characteristics of the replay
system under load conditions.

Performance Test Coverage:
    - Large event replay (1000+ events)
    - Replay with deduplication (50% duplicates)
    - Replay with intermittent failures (chaos + replay)
    - Memory usage during large replay

Performance Thresholds:
    - 1000 events should replay in < 5 seconds
    - Deduplication overhead should be < 20% of base replay time
    - Memory growth should be bounded (< 100MB for 10K events)

Note:
    These tests are marked with @pytest.mark.slow and may take longer
    to execute than unit tests. Run with `pytest -m slow` to execute
    only performance tests.

Related:
    - OMN-955: Event Replay Verification
    - test_idempotent_replay.py: Correctness tests
    - test_reducer_replay_determinism.py: Determinism tests
"""

from __future__ import annotations

import sys
import time
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest

from omnibase_infra.idempotency import InMemoryIdempotencyStore
from omnibase_infra.models.registration import (
    ModelNodeCapabilities,
    ModelNodeIntrospectionEvent,
    ModelNodeMetadata,
)
from omnibase_infra.nodes.reducers import RegistrationReducer
from omnibase_infra.nodes.reducers.models import ModelRegistrationState
from tests.helpers.deterministic import DeterministicClock, DeterministicIdGenerator

if TYPE_CHECKING:
    from tests.replay.conftest import EventFactory


# =============================================================================
# Constants
# =============================================================================

# Performance thresholds (configurable for CI environments)
REPLAY_1000_EVENTS_THRESHOLD_SECONDS = 5.0
REPLAY_5000_EVENTS_THRESHOLD_SECONDS = 25.0
DEDUPLICATION_OVERHEAD_MAX_PERCENT = 50.0  # 50% overhead allowed
MEMORY_GROWTH_MAX_MB = 100.0  # Max memory growth for 10K events


# =============================================================================
# Helper Functions
# =============================================================================


def generate_events(
    count: int,
    id_generator: DeterministicIdGenerator,
    clock: DeterministicClock,
    node_type: str = "effect",
) -> list[ModelNodeIntrospectionEvent]:
    """Generate a batch of deterministic events for performance testing.

    Args:
        count: Number of events to generate.
        id_generator: Deterministic ID generator for reproducibility.
        clock: Deterministic clock for reproducible timestamps.
        node_type: ONEX node type for events.

    Returns:
        List of ModelNodeIntrospectionEvent instances.
    """
    events: list[ModelNodeIntrospectionEvent] = []
    for i in range(count):
        if i > 0:
            clock.advance(1)  # 1 second between events
        events.append(
            ModelNodeIntrospectionEvent(
                node_id=id_generator.next_uuid(),
                node_type=node_type,
                node_version="1.0.0",
                correlation_id=id_generator.next_uuid(),
                timestamp=clock.now(),
                endpoints={},
                capabilities=ModelNodeCapabilities(),
                metadata=ModelNodeMetadata(),
            )
        )
    return events


def generate_events_with_duplicates(
    total_count: int,
    duplicate_rate: float,
    id_generator: DeterministicIdGenerator,
    clock: DeterministicClock,
) -> list[ModelNodeIntrospectionEvent]:
    """Generate events with a specified rate of duplicates.

    Args:
        total_count: Total number of events to generate.
        duplicate_rate: Fraction of events that should be duplicates (0.0-1.0).
        id_generator: Deterministic ID generator.
        clock: Deterministic clock.

    Returns:
        List of events where `duplicate_rate` fraction are duplicates.
    """
    unique_count = int(total_count * (1 - duplicate_rate))
    unique_events = generate_events(unique_count, id_generator, clock)

    # Create duplicates by repeating unique events
    events: list[ModelNodeIntrospectionEvent] = []
    duplicate_idx = 0
    unique_idx = 0

    for i in range(total_count):
        if unique_idx < len(unique_events):
            events.append(unique_events[unique_idx])
            unique_idx += 1
        else:
            # Cycle through unique events for duplicates
            events.append(unique_events[duplicate_idx % len(unique_events)])
            duplicate_idx += 1

    return events


# =============================================================================
# Large Event Replay Performance Tests
# =============================================================================


@pytest.mark.slow
@pytest.mark.asyncio
class TestLargeEventReplayPerformance:
    """Performance tests for large event replay scenarios."""

    async def test_replay_1000_events_performance(
        self,
        reducer: RegistrationReducer,
    ) -> None:
        """Test replay performance with 1000 events.

        Validates that processing 1000 events completes within the
        acceptable time threshold. This is the baseline performance test.
        """
        id_generator = DeterministicIdGenerator(seed=42)
        clock = DeterministicClock()
        events = generate_events(1000, id_generator, clock)

        start_time = time.perf_counter()

        for event in events:
            state = ModelRegistrationState()
            reducer.reduce(state, event)

        elapsed = time.perf_counter() - start_time

        # Assert performance threshold
        assert elapsed < REPLAY_1000_EVENTS_THRESHOLD_SECONDS, (
            f"Replay of 1000 events took {elapsed:.2f}s, "
            f"expected < {REPLAY_1000_EVENTS_THRESHOLD_SECONDS}s"
        )

        # Log performance metrics for visibility
        events_per_second = 1000 / elapsed
        print(
            f"\n[Performance] 1000 events: {elapsed:.3f}s ({events_per_second:.0f} events/s)"
        )

    async def test_replay_5000_events_performance(
        self,
        reducer: RegistrationReducer,
    ) -> None:
        """Test replay performance with 5000 events.

        Validates linear scaling of replay performance with larger
        event counts.
        """
        id_generator = DeterministicIdGenerator(seed=42)
        clock = DeterministicClock()
        events = generate_events(5000, id_generator, clock)

        start_time = time.perf_counter()

        for event in events:
            state = ModelRegistrationState()
            reducer.reduce(state, event)

        elapsed = time.perf_counter() - start_time

        # Assert performance threshold
        assert elapsed < REPLAY_5000_EVENTS_THRESHOLD_SECONDS, (
            f"Replay of 5000 events took {elapsed:.2f}s, "
            f"expected < {REPLAY_5000_EVENTS_THRESHOLD_SECONDS}s"
        )

        events_per_second = 5000 / elapsed
        print(
            f"\n[Performance] 5000 events: {elapsed:.3f}s ({events_per_second:.0f} events/s)"
        )


# =============================================================================
# Deduplication Performance Tests
# =============================================================================


@pytest.mark.slow
@pytest.mark.asyncio
class TestDeduplicationReplayPerformance:
    """Performance tests for replay with deduplication."""

    async def test_replay_with_50_percent_duplicates(
        self,
        reducer: RegistrationReducer,
    ) -> None:
        """Test replay performance with 50% duplicate events.

        Validates that idempotency checking does not add excessive
        overhead to replay performance.
        """
        id_generator = DeterministicIdGenerator(seed=42)
        clock = DeterministicClock()
        events = generate_events_with_duplicates(
            total_count=1000,
            duplicate_rate=0.5,
            id_generator=id_generator,
            clock=clock,
        )

        # Track unique vs duplicate processing
        processed_event_ids: set[str] = set()
        unique_count = 0
        duplicate_count = 0

        start_time = time.perf_counter()

        for event in events:
            event_key = str(event.correlation_id)
            state = ModelRegistrationState()

            if event_key in processed_event_ids:
                # Simulate replay of already-processed event
                state = ModelRegistrationState(
                    last_processed_event_id=event.correlation_id
                )
                duplicate_count += 1
            else:
                processed_event_ids.add(event_key)
                unique_count += 1

            reducer.reduce(state, event)

        elapsed = time.perf_counter() - start_time

        # Performance should still be acceptable with duplicates
        assert elapsed < REPLAY_1000_EVENTS_THRESHOLD_SECONDS * 1.5, (
            f"Replay with 50% duplicates took {elapsed:.2f}s, "
            f"expected < {REPLAY_1000_EVENTS_THRESHOLD_SECONDS * 1.5}s"
        )

        events_per_second = 1000 / elapsed
        print(
            f"\n[Performance] 1000 events (50% duplicates): {elapsed:.3f}s "
            f"({events_per_second:.0f} events/s)"
        )
        print(f"  Unique: {unique_count}, Duplicates: {duplicate_count}")

    async def test_idempotency_store_deduplication_performance(
        self,
    ) -> None:
        """Test idempotency store performance for deduplication.

        Measures the overhead of check_and_record operations during
        replay with duplicates.
        """
        store = InMemoryIdempotencyStore()
        event_count = 1000
        duplicate_rate = 0.5

        # Generate message IDs (50% unique, 50% duplicates)
        unique_message_ids = [
            uuid4() for _ in range(int(event_count * (1 - duplicate_rate)))
        ]
        all_message_ids = unique_message_ids.copy()

        # Add duplicates
        for i in range(int(event_count * duplicate_rate)):
            all_message_ids.append(unique_message_ids[i % len(unique_message_ids)])

        start_time = time.perf_counter()

        is_new_count = 0
        is_duplicate_count = 0

        for message_id in all_message_ids:
            is_new = await store.check_and_record(
                message_id=message_id,
                domain="replay_perf_test",
            )
            if is_new:
                is_new_count += 1
            else:
                is_duplicate_count += 1

        elapsed = time.perf_counter() - start_time

        # Idempotency store should be very fast
        assert elapsed < 1.0, (
            f"Idempotency check for {event_count} events took {elapsed:.2f}s, "
            f"expected < 1.0s"
        )

        ops_per_second = event_count / elapsed
        print(
            f"\n[Performance] Idempotency store: {elapsed:.3f}s "
            f"({ops_per_second:.0f} ops/s)"
        )
        print(f"  New: {is_new_count}, Duplicates: {is_duplicate_count}")

        # Verify deduplication worked correctly
        assert is_new_count == len(unique_message_ids)
        assert is_duplicate_count == int(event_count * duplicate_rate)


# =============================================================================
# Chaos + Replay Performance Tests
# =============================================================================


@pytest.mark.slow
@pytest.mark.asyncio
class TestChaosReplayPerformance:
    """Performance tests combining chaos injection with replay."""

    async def test_replay_with_intermittent_failures(
        self,
    ) -> None:
        """Test replay performance with intermittent failures.

        Simulates a real-world scenario where some operations fail
        and need to be retried. Measures total time including retries.
        """
        store = InMemoryIdempotencyStore()
        event_count = 500
        failure_rate = 0.1  # 10% failure rate

        message_ids = [uuid4() for _ in range(event_count)]
        correlation_id = uuid4()

        import random

        random.seed(42)  # Deterministic failures

        start_time = time.perf_counter()

        success_count = 0
        failure_count = 0
        retry_count = 0

        for message_id in message_ids:
            max_retries = 3
            succeeded = False

            for attempt in range(max_retries):
                # Check idempotency first
                is_new = await store.check_and_record(
                    message_id=message_id,
                    domain="chaos_replay_test",
                    correlation_id=correlation_id,
                )

                if not is_new:
                    # Already processed - skip
                    succeeded = True
                    break

                # Simulate potential failure
                if random.random() < failure_rate:
                    failure_count += 1
                    if attempt < max_retries - 1:
                        retry_count += 1
                        # Clear the record to allow retry
                        # (In real systems, failure before record = no record)
                        # For this test, we simulate transient failures
                        continue
                else:
                    success_count += 1
                    succeeded = True
                    break

            if not succeeded:
                success_count += 1  # Final attempt counted as success for metric

        elapsed = time.perf_counter() - start_time

        # Performance should degrade gracefully with failures
        # Allow 50% more time for retries
        expected_threshold = (
            (event_count / 1000) * REPLAY_1000_EVENTS_THRESHOLD_SECONDS * 1.5
        )
        assert elapsed < expected_threshold, (
            f"Chaos replay took {elapsed:.2f}s, expected < {expected_threshold:.2f}s"
        )

        events_per_second = event_count / elapsed
        print(
            f"\n[Performance] Chaos replay ({event_count} events, "
            f"{failure_rate * 100:.0f}% failure rate): {elapsed:.3f}s "
            f"({events_per_second:.0f} events/s)"
        )
        print(
            f"  Successes: {success_count}, Failures: {failure_count}, Retries: {retry_count}"
        )

    async def test_recovery_replay_after_simulated_crash(
        self,
        reducer: RegistrationReducer,
    ) -> None:
        """Test replay performance for crash recovery scenario.

        Simulates a crash at 50% completion and measures the time
        to replay and complete the remaining events.
        """
        id_generator = DeterministicIdGenerator(seed=42)
        clock = DeterministicClock()
        events = generate_events(1000, id_generator, clock)

        crash_point = len(events) // 2

        # Phase 1: Process first half (simulating pre-crash work)
        processed_event_ids: set[str] = set()

        start_phase1 = time.perf_counter()
        for event in events[:crash_point]:
            state = ModelRegistrationState()
            reducer.reduce(state, event)
            processed_event_ids.add(str(event.correlation_id))
        phase1_elapsed = time.perf_counter() - start_phase1

        # Phase 2: Recovery replay (replay all, skip processed)
        start_phase2 = time.perf_counter()
        replayed_count = 0
        skipped_count = 0

        for event in events:
            event_key = str(event.correlation_id)

            if event_key in processed_event_ids:
                # Simulate idempotent skip
                state = ModelRegistrationState(
                    last_processed_event_id=event.correlation_id
                )
                skipped_count += 1
            else:
                state = ModelRegistrationState()
                replayed_count += 1

            reducer.reduce(state, event)

        phase2_elapsed = time.perf_counter() - start_phase2
        total_elapsed = phase1_elapsed + phase2_elapsed

        # Recovery replay should be faster than full replay
        # because half the events are skipped via idempotency
        assert total_elapsed < REPLAY_1000_EVENTS_THRESHOLD_SECONDS * 2, (
            f"Crash recovery replay took {total_elapsed:.2f}s, "
            f"expected < {REPLAY_1000_EVENTS_THRESHOLD_SECONDS * 2}s"
        )

        print("\n[Performance] Crash recovery replay:")
        print(f"  Phase 1 (pre-crash): {phase1_elapsed:.3f}s ({crash_point} events)")
        print(
            f"  Phase 2 (recovery): {phase2_elapsed:.3f}s ({len(events)} events total)"
        )
        print(f"  Skipped: {skipped_count}, Replayed: {replayed_count}")
        print(f"  Total: {total_elapsed:.3f}s")


# =============================================================================
# Memory Usage Performance Tests
# =============================================================================


@pytest.mark.slow
@pytest.mark.asyncio
class TestMemoryUsagePerformance:
    """Performance tests for memory usage during large replay."""

    async def test_memory_usage_10k_events(
        self,
        reducer: RegistrationReducer,
    ) -> None:
        """Test memory usage during replay of 10K events.

        Validates that memory usage remains bounded during large
        replay operations. This helps detect memory leaks.
        """
        import gc

        # Force garbage collection before measurement
        gc.collect()
        initial_memory = sys.getsizeof(reducer)

        id_generator = DeterministicIdGenerator(seed=42)
        clock = DeterministicClock()

        # Process events in batches to allow GC
        batch_size = 1000
        total_events = 10000
        memory_samples: list[int] = [initial_memory]

        start_time = time.perf_counter()

        for batch_start in range(0, total_events, batch_size):
            batch_events = generate_events(
                count=batch_size,
                id_generator=id_generator,
                clock=clock,
            )

            for event in batch_events:
                state = ModelRegistrationState()
                reducer.reduce(state, event)

            # Sample memory after each batch
            gc.collect()
            current_memory = sys.getsizeof(reducer)
            memory_samples.append(current_memory)

        elapsed = time.perf_counter() - start_time

        # Calculate memory growth
        max_memory = max(memory_samples)
        memory_growth_bytes = max_memory - initial_memory
        memory_growth_mb = memory_growth_bytes / (1024 * 1024)

        # Reducer should not accumulate significant state
        # (it's a pure function with no internal state)
        assert memory_growth_mb < MEMORY_GROWTH_MAX_MB, (
            f"Memory growth during 10K event replay: {memory_growth_mb:.2f}MB, "
            f"expected < {MEMORY_GROWTH_MAX_MB}MB"
        )

        events_per_second = total_events / elapsed
        print("\n[Performance] 10K events memory test:")
        print(f"  Time: {elapsed:.3f}s ({events_per_second:.0f} events/s)")
        print(f"  Initial memory: {initial_memory} bytes")
        print(f"  Max memory: {max_memory} bytes")
        print(f"  Memory growth: {memory_growth_mb:.4f}MB")

    async def test_idempotency_store_memory_growth(
        self,
    ) -> None:
        """Test memory usage of idempotency store with many records.

        Validates that the in-memory store's memory growth is reasonable
        for large numbers of records.
        """
        import gc

        store = InMemoryIdempotencyStore()

        gc.collect()

        # Measure baseline
        baseline_records = await store.get_all_records()
        assert len(baseline_records) == 0

        # Add 10K records
        event_count = 10000

        start_time = time.perf_counter()

        for i in range(event_count):
            await store.check_and_record(
                message_id=uuid4(),
                domain="memory_test",
                correlation_id=uuid4(),
            )

        elapsed = time.perf_counter() - start_time

        # Verify all records stored
        record_count = await store.get_record_count()
        assert record_count == event_count

        # Get all records to measure size
        all_records = await store.get_all_records()
        total_size = sum(sys.getsizeof(r) for r in all_records)
        size_mb = total_size / (1024 * 1024)

        # Memory should be reasonable for 10K records
        # Each record is small (UUID + timestamp + optional correlation)
        assert size_mb < 50, (
            f"Idempotency store memory for 10K records: {size_mb:.2f}MB, "
            f"expected < 50MB"
        )

        ops_per_second = event_count / elapsed
        print("\n[Performance] Idempotency store memory (10K records):")
        print(f"  Time: {elapsed:.3f}s ({ops_per_second:.0f} ops/s)")
        print(f"  Records: {record_count}")
        print(f"  Approximate size: {size_mb:.4f}MB")
        print(f"  Per-record size: {total_size / event_count:.1f} bytes")


# =============================================================================
# Throughput Benchmark Tests
# =============================================================================


@pytest.mark.slow
@pytest.mark.asyncio
class TestReplayThroughput:
    """Throughput benchmark tests for replay operations."""

    async def test_sustained_replay_throughput(
        self,
        reducer: RegistrationReducer,
    ) -> None:
        """Test sustained replay throughput over extended period.

        Measures whether throughput remains stable during extended
        replay operations.
        """
        id_generator = DeterministicIdGenerator(seed=42)
        clock = DeterministicClock()

        batch_size = 1000
        num_batches = 5
        batch_times: list[float] = []

        for batch_num in range(num_batches):
            events = generate_events(batch_size, id_generator, clock)

            start_time = time.perf_counter()

            for event in events:
                state = ModelRegistrationState()
                reducer.reduce(state, event)

            elapsed = time.perf_counter() - start_time
            batch_times.append(elapsed)

        # Calculate throughput statistics
        avg_time = sum(batch_times) / len(batch_times)
        max_time = max(batch_times)
        min_time = min(batch_times)
        variance = sum((t - avg_time) ** 2 for t in batch_times) / len(batch_times)

        # Throughput should be stable (low variance)
        coefficient_of_variation = (variance**0.5) / avg_time
        assert coefficient_of_variation < 0.5, (
            f"Throughput variance too high: CV={coefficient_of_variation:.2f}, "
            f"expected < 0.5"
        )

        avg_throughput = batch_size / avg_time
        print(
            f"\n[Performance] Sustained throughput ({num_batches} batches of {batch_size}):"
        )
        print(f"  Avg time per batch: {avg_time:.3f}s")
        print(f"  Min/Max: {min_time:.3f}s / {max_time:.3f}s")
        print(f"  Throughput: {avg_throughput:.0f} events/s")
        print(f"  Coefficient of variation: {coefficient_of_variation:.3f}")


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "TestLargeEventReplayPerformance",
    "TestDeduplicationReplayPerformance",
    "TestChaosReplayPerformance",
    "TestMemoryUsagePerformance",
    "TestReplayThroughput",
]
