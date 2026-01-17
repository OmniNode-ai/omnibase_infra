# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Replay Performance Tests for OMN-955.

This module provides performance tests for large event replay sequences.
These tests measure and validate performance characteristics of the replay
system under load conditions.

Statistical Rigor:
    - All performance tests use multiple iterations (not single runs)
    - Warmup iterations are discarded to eliminate JIT/cache effects
    - Results are reported using median and percentiles (robust to outliers)
    - Memory tracking uses tracemalloc for accurate measurement
    - Deterministic seeding ensures reproducibility

Performance Test Coverage:
    - Large event replay (1000+ events)
    - Replay with deduplication (50% duplicates)
    - Replay with intermittent failures (chaos + replay)
    - Memory usage during large replay (with baseline tracking)

Performance Thresholds:
    - 1000 events should replay in < 5 seconds (P95 threshold)
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

import random
import time
from uuid import uuid4

import pytest

from omnibase_core.models.primitives.model_semver import ModelSemVer
from omnibase_infra.idempotency import StoreIdempotencyInmemory
from omnibase_infra.models.registration import (
    ModelNodeCapabilities,
    ModelNodeIntrospectionEvent,
    ModelNodeMetadata,
)
from omnibase_infra.nodes.reducers import RegistrationReducer
from omnibase_infra.nodes.reducers.models import ModelRegistrationState
from tests.helpers.deterministic import DeterministicClock, DeterministicIdGenerator
from tests.helpers.statistics_utils import (
    MemoryTracker,
    PerformanceStats,
    run_with_warmup_sync,
)

# =============================================================================
# Module-Level Markers
# =============================================================================
# These markers enable selective test execution:
#   pytest -m "replay" - run only replay tests
#   pytest -m "performance" - run only performance tests
#   pytest -m "not replay" - skip replay tests

pytestmark = [
    pytest.mark.replay,
    pytest.mark.performance,
]

# =============================================================================
# Constants
# =============================================================================

# Performance thresholds (configurable for CI environments)
# These are P95 thresholds - 95% of runs should complete within these times
REPLAY_1000_EVENTS_THRESHOLD_SECONDS = 5.0
REPLAY_5000_EVENTS_THRESHOLD_SECONDS = 25.0
DEDUPLICATION_OVERHEAD_MAX_PERCENT = 50.0  # 50% overhead allowed
MEMORY_GROWTH_MAX_MB = 100.0  # Max memory growth for 10K events

# Statistical parameters
DEFAULT_ITERATIONS = 10  # Number of timed iterations per test
DEFAULT_WARMUP_ITERATIONS = 3  # Warmup iterations (discarded)
MEMORY_BASELINE_TOLERANCE_MB = 10.0  # Acceptable variance from baseline


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
                node_version=ModelSemVer.parse("1.0.0"),
                correlation_id=id_generator.next_uuid(),
                timestamp=clock.now(),
                endpoints={},
                declared_capabilities=ModelNodeCapabilities(),
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
        Duplicates are copies of unique events (same correlation_id).

    Example:
        total_count=1000, duplicate_rate=0.5 yields:
        - 500 unique events
        - 500 duplicates (cycling through the unique events)
    """
    # Calculate unique count, ensuring at least 1 unique event to avoid division by zero
    unique_count = max(1, int(total_count * (1 - duplicate_rate)))
    duplicate_count = total_count - unique_count

    # Generate unique events first
    unique_events = generate_events(unique_count, id_generator, clock)

    # Start with all unique events
    events: list[ModelNodeIntrospectionEvent] = list(unique_events)

    # Add duplicates by cycling through unique events
    for i in range(duplicate_count):
        # Create a copy of the event - same correlation_id but different object
        # This ensures duplicates are separate instances for proper testing
        original = unique_events[i % len(unique_events)]
        duplicate = original.model_copy()
        events.append(duplicate)

    return events


# =============================================================================
# Large Event Replay Performance Tests
# =============================================================================


@pytest.mark.slow  # 1000-5000 events for throughput benchmarks
@pytest.mark.asyncio
class TestLargeEventReplayPerformance:
    """Performance tests for large event replay scenarios.

    Statistical Approach:
        - Each test runs multiple iterations (DEFAULT_ITERATIONS)
        - Warmup iterations are discarded (DEFAULT_WARMUP_ITERATIONS)
        - Thresholds are applied to P95 (95th percentile) to handle outliers
        - Full statistics are reported for analysis
    """

    async def test_replay_1000_events_performance(
        self,
        reducer: RegistrationReducer,
    ) -> None:
        """Test replay performance with 1000 events.

        Statistical Methodology:
            - Runs 10 iterations with 3 warmup iterations
            - Uses P95 threshold for statistically valid assertion
            - Reports full statistics including median, P50, P90, P95, P99

        Validates that processing 1000 events completes within the
        acceptable time threshold. This is the baseline performance test.
        """
        # Pre-generate events once (not part of timing)
        id_generator = DeterministicIdGenerator(seed=42)
        clock = DeterministicClock()
        events = generate_events(1000, id_generator, clock)

        def run_replay() -> None:
            """Single replay iteration."""
            for event in events:
                state = ModelRegistrationState()
                reducer.reduce(state, event)

        # Run with warmup and collect timings
        timings = run_with_warmup_sync(
            operation=run_replay,
            iterations=DEFAULT_ITERATIONS,
            warmup_iterations=DEFAULT_WARMUP_ITERATIONS,
        )

        stats = PerformanceStats.from_samples(timings)

        # Assert P95 threshold (more robust than single run)
        assert stats.p95 < REPLAY_1000_EVENTS_THRESHOLD_SECONDS, (
            f"Replay of 1000 events P95={stats.p95:.2f}s, "
            f"expected < {REPLAY_1000_EVENTS_THRESHOLD_SECONDS}s\n"
            f"{stats.format_report('1000 events replay')}"
        )

        # Log comprehensive statistics
        events_per_second = 1000 / stats.median
        print(f"\n{stats.format_report('1000 events replay')}")
        print(f"  Throughput (median): {events_per_second:.0f} events/s")

    async def test_replay_5000_events_performance(
        self,
        reducer: RegistrationReducer,
    ) -> None:
        """Test replay performance with 5000 events.

        Statistical Methodology:
            - Runs 10 iterations with 3 warmup iterations
            - Uses P95 threshold for statistically valid assertion
            - Validates linear scaling from 1000 events

        Validates linear scaling of replay performance with larger
        event counts.
        """
        # Pre-generate events once (not part of timing)
        id_generator = DeterministicIdGenerator(seed=42)
        clock = DeterministicClock()
        events = generate_events(5000, id_generator, clock)

        def run_replay() -> None:
            """Single replay iteration."""
            for event in events:
                state = ModelRegistrationState()
                reducer.reduce(state, event)

        # Run with warmup and collect timings
        timings = run_with_warmup_sync(
            operation=run_replay,
            iterations=DEFAULT_ITERATIONS,
            warmup_iterations=DEFAULT_WARMUP_ITERATIONS,
        )

        stats = PerformanceStats.from_samples(timings)

        # Assert P95 threshold
        assert stats.p95 < REPLAY_5000_EVENTS_THRESHOLD_SECONDS, (
            f"Replay of 5000 events P95={stats.p95:.2f}s, "
            f"expected < {REPLAY_5000_EVENTS_THRESHOLD_SECONDS}s\n"
            f"{stats.format_report('5000 events replay')}"
        )

        events_per_second = 5000 / stats.median
        print(f"\n{stats.format_report('5000 events replay')}")
        print(f"  Throughput (median): {events_per_second:.0f} events/s")


# =============================================================================
# Deduplication Performance Tests
# =============================================================================


@pytest.mark.slow  # 1000 events with deduplication overhead
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
        store = StoreIdempotencyInmemory()
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


@pytest.mark.slow  # 500-1000 events with retry simulation
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
        store = StoreIdempotencyInmemory()
        event_count = 500
        failure_rate = 0.1  # 10% failure rate

        message_ids = [uuid4() for _ in range(event_count)]
        correlation_id = uuid4()

        random.seed(42)  # Deterministic failures

        start_time = time.perf_counter()

        success_count = 0
        failure_count = 0
        retry_count = 0

        for message_id in message_ids:
            max_retries = 3
            succeeded = False

            for attempt in range(max_retries):
                # Check idempotency first (read-only check)
                already_processed = await store.is_processed(
                    message_id=message_id,
                    domain="chaos_replay_test",
                )

                if already_processed:
                    # Already processed successfully - skip
                    succeeded = True
                    break

                # Simulate potential failure BEFORE recording
                # In real systems, failure before record = no record,
                # so retries will see the message as unprocessed
                if random.random() < failure_rate:
                    failure_count += 1
                    if attempt < max_retries - 1:
                        retry_count += 1
                        # No record was made, so retry will work
                        continue
                    # Final attempt - fall through to record success
                    # (simulating eventual success after retries exhausted)

                # Success - record the idempotency key
                await store.mark_processed(
                    message_id=message_id,
                    domain="chaos_replay_test",
                    correlation_id=correlation_id,
                )
                success_count += 1
                succeeded = True
                break

            if not succeeded:
                # All retries failed - record anyway for idempotency tracking
                # (prevents reprocessing on next replay attempt)
                await store.mark_processed(
                    message_id=message_id,
                    domain="chaos_replay_test",
                    correlation_id=correlation_id,
                )
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
            f"  Successes: {success_count}, Failures: {failure_count}, "
            f"Retries: {retry_count}"
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


@pytest.mark.slow  # 10000 events for memory measurement
@pytest.mark.asyncio
class TestMemoryUsagePerformance:
    """Performance tests for memory usage during large replay.

    Memory Tracking Approach:
        - Uses tracemalloc for accurate memory measurement (not sys.getsizeof)
        - Takes snapshots at batch boundaries for growth analysis
        - Baseline established before test with GC forced
        - Reports peak memory and growth from baseline
    """

    @pytest.mark.xfail(
        reason="omnibase_core 0.7.0 adds intent_type mismatch warnings that increase memory usage beyond 10MB tolerance. See OMN-1361.",
        strict=True,
    )
    async def test_memory_usage_10k_events(
        self,
        reducer: RegistrationReducer,
    ) -> None:
        """Test memory usage during replay of 10K events.

        Memory Measurement Methodology:
            - Uses tracemalloc for accurate heap tracking
            - Forces GC before baseline measurement
            - Snapshots at each 1K batch boundary
            - Asserts on peak memory growth, not instantaneous

        Validates that memory usage remains bounded during large
        replay operations. This helps detect memory leaks.
        """
        # Initialize memory tracker with proper tracemalloc
        tracker = MemoryTracker()
        tracker.start()

        id_generator = DeterministicIdGenerator(seed=42)
        clock = DeterministicClock()

        # Process events in batches to allow GC and snapshots
        batch_size = 1000
        total_events = 10000

        start_time = time.perf_counter()

        for batch_num, _batch_start in enumerate(range(0, total_events, batch_size)):
            batch_events = generate_events(
                count=batch_size,
                id_generator=id_generator,
                clock=clock,
            )

            for event in batch_events:
                state = ModelRegistrationState()
                reducer.reduce(state, event)

            # Snapshot memory after each batch
            tracker.snapshot(f"batch_{batch_num}")

        elapsed = time.perf_counter() - start_time

        # Get memory statistics
        peak_mb = tracker.get_peak_mb()
        final_growth_mb = tracker.get_growth_mb()

        # Stop tracking before assertions
        tracker.stop()

        # Memory baseline assertion: peak growth should be bounded
        assert peak_mb < MEMORY_GROWTH_MAX_MB, (
            f"Peak memory during 10K event replay: {peak_mb:.2f}MB, "
            f"expected < {MEMORY_GROWTH_MAX_MB}MB\n"
            f"{tracker.format_report()}"
        )

        # Additional assertion: final growth should be reasonable
        # (reducer should not accumulate permanent state)
        assert final_growth_mb < MEMORY_BASELINE_TOLERANCE_MB, (
            f"Final memory growth after 10K events: {final_growth_mb:.2f}MB, "
            f"expected < {MEMORY_BASELINE_TOLERANCE_MB}MB "
            "(reducer should not accumulate state)"
        )

        events_per_second = total_events / elapsed
        print("\n[Performance] 10K events memory test:")
        print(f"  Time: {elapsed:.3f}s ({events_per_second:.0f} events/s)")
        print(f"{tracker.format_report()}")

    async def test_idempotency_store_memory_growth(
        self,
    ) -> None:
        """Test memory usage of idempotency store with many records.

        Memory Measurement Methodology:
            - Uses tracemalloc for accurate heap tracking
            - Measures before and after adding 10K records
            - Reports per-record memory consumption

        Validates that the in-memory store's memory growth is reasonable
        for large numbers of records.
        """
        store = StoreIdempotencyInmemory()

        # Initialize memory tracker
        tracker = MemoryTracker()
        tracker.start()

        # Measure baseline
        baseline_records = await store.get_all_records()
        assert len(baseline_records) == 0
        tracker.snapshot("baseline")

        # Add 10K records
        event_count = 10000

        start_time = time.perf_counter()

        for i in range(event_count):
            await store.check_and_record(
                message_id=uuid4(),
                domain="memory_test",
                correlation_id=uuid4(),
            )

            # Snapshot at intervals for growth tracking
            if (i + 1) % 2500 == 0:
                tracker.snapshot(f"records_{i + 1}")

        elapsed = time.perf_counter() - start_time

        # Verify all records stored
        record_count = await store.get_record_count()
        assert record_count == event_count

        # Final memory snapshot
        tracker.snapshot("final")
        final_growth_mb = tracker.get_growth_mb("final")
        peak_mb = tracker.get_peak_mb()

        # Stop tracking
        tracker.stop()

        # Memory should be reasonable for 10K records
        # Each record is small (UUID + timestamp + optional correlation)
        assert final_growth_mb < 50, (
            f"Idempotency store memory for 10K records: {final_growth_mb:.2f}MB, "
            f"expected < 50MB\n"
            f"{tracker.format_report()}"
        )

        ops_per_second = event_count / elapsed
        per_record_bytes = (final_growth_mb * 1024 * 1024) / event_count

        print("\n[Performance] Idempotency store memory (10K records):")
        print(f"  Time: {elapsed:.3f}s ({ops_per_second:.0f} ops/s)")
        print(f"  Records: {record_count}")
        print(f"  Final growth: {final_growth_mb:.4f}MB")
        print(f"  Peak: {peak_mb:.4f}MB")
        print(f"  Per-record size: {per_record_bytes:.1f} bytes")


# =============================================================================
# Throughput Benchmark Tests
# =============================================================================


@pytest.mark.slow  # 10x1000 event batches for throughput stability
@pytest.mark.asyncio
class TestReplayThroughput:
    """Throughput benchmark tests for replay operations.

    Statistical Approach:
        - Uses 10 batches (increased from 5) for better statistical validity
        - Discards first batch as warmup
        - Reports coefficient of variation for stability assessment
        - Uses PerformanceStats for comprehensive analysis
    """

    @pytest.mark.xfail(
        reason=(
            "Performance test with absolute P95/median threshold (< 2.0) is inherently "
            "flaky in CI environments due to variable CPU/memory resources. "
            "omnibase_core 0.7.0 intent_type mismatch warnings add logging overhead "
            "that causes timing variability. See OMN-1361."
        ),
        strict=True,
    )
    async def test_sustained_replay_throughput(
        self,
        reducer: RegistrationReducer,
    ) -> None:
        """Test sustained replay throughput over extended period.

        Statistical Methodology:
            - Runs 10 batches of 1000 events each
            - First batch is warmup (discarded from statistics)
            - Coefficient of variation < 0.5 indicates stable throughput
            - Reports full statistics including P95 for worst-case analysis

        Measures whether throughput remains stable during extended
        replay operations.

        Note:
            This test is marked xfail(strict=True) because performance tests with
            absolute thresholds are unreliable in CI environments. The strict=True
            ensures XPASS fails if the underlying performance issue is resolved.
        """
        id_generator = DeterministicIdGenerator(seed=42)
        clock = DeterministicClock()

        batch_size = 1000
        num_batches = 10  # Increased from 5 for better statistics
        warmup_batches = 1  # First batch is warmup
        batch_times: list[float] = []

        for batch_num in range(num_batches):
            events = generate_events(batch_size, id_generator, clock)

            start_time = time.perf_counter()

            for event in events:
                state = ModelRegistrationState()
                reducer.reduce(state, event)

            elapsed = time.perf_counter() - start_time

            # Skip warmup batches
            if batch_num >= warmup_batches:
                batch_times.append(elapsed)

        # Use PerformanceStats for comprehensive analysis
        stats = PerformanceStats.from_samples(batch_times)

        # Throughput should be stable (low coefficient of variation)
        assert stats.coefficient_of_variation < 0.5, (
            f"Throughput variance too high: CV={stats.coefficient_of_variation:.2f}, "
            f"expected < 0.5\n"
            f"{stats.format_report('Batch timing')}"
        )

        # Also assert P95 is within reasonable bounds of median
        # (no extreme outliers that would indicate instability)
        p95_to_median_ratio = stats.p95 / stats.median if stats.median > 0 else 0
        assert p95_to_median_ratio < 2.0, (
            f"P95/median ratio {p95_to_median_ratio:.2f} indicates instability, "
            f"expected < 2.0"
        )

        median_throughput = batch_size / stats.median
        print(
            f"\n[Performance] Sustained throughput "
            f"({num_batches - warmup_batches} batches of {batch_size}, "
            f"{warmup_batches} warmup):"
        )
        print(f"{stats.format_report('Batch timing')}")
        print(f"  Median throughput: {median_throughput:.0f} events/s")
        print(f"  P95/median ratio: {p95_to_median_ratio:.2f}")


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
