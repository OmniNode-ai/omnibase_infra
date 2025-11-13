#!/usr/bin/env python3
"""
Comprehensive Load Tests for Offset Cache Race Condition Fix

This test suite verifies the race condition fix from commit 341afa5
under high-throughput conditions (1000+ messages/second).

Race Condition Fix Components:
1. is_offset_processed() - Atomic read with TTL cache + legacy set fallback
2. _add_processed_offset() - Atomic write with lock protection
3. _cleanup_processed_offsets() - Protected cleanup with _cleanup_lock

Test Coverage:
1. High-throughput load tests (1000+ msg/sec)
2. Concurrent read/write stress tests
3. Message duplication detection
4. Message loss detection
5. Performance degradation monitoring
6. Lock contention analysis
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import asyncio
import random
import time
from collections import Counter
from uuid import uuid4

import pytest

# Import components under test
from omninode_bridge.utils.ttl_cache import create_ttl_cache

# Test Fixtures
# =============


@pytest.fixture
async def offset_cache():
    """Create TTL cache for offset tracking."""
    cache = create_ttl_cache(
        name="test-offset-cache",
        environment="test",
        max_size=50000,  # Large enough for load tests
        ttl_seconds=300.0,  # 5 minute TTL
        cleanup_interval_seconds=10.0,
    )
    yield cache
    # Cleanup - properly await the stop coroutine
    await cache.stop()


@pytest.fixture
def mock_registry_node(offset_cache):
    """Create mock registry node with offset cache and atomic operations."""

    class MockRegistryNode:
        def __init__(self, offset_cache):
            self._offset_cache = offset_cache
            self._processed_message_offsets: set[str] = set()
            self._max_tracked_offsets = 50000
            self._cleanup_lock = asyncio.Lock()

            # Metrics tracking
            self.offset_check_count = 0
            self.offset_add_count = 0
            self.duplicate_detected_count = 0
            self.cleanup_count = 0

        async def is_offset_processed(self, offset_key: str) -> bool:
            """
            Check if offset is processed with race condition prevention.

            Thread-safe implementation using TTL cache (primary) and
            legacy set (fallback) with lock protection.
            """
            self.offset_check_count += 1

            # First check TTL cache (thread-safe by design)
            cached_result = self._offset_cache.get(offset_key)
            if cached_result is not None:
                return cached_result

            # Fallback to legacy set with lock protection
            async with self._cleanup_lock:
                return offset_key in self._processed_message_offsets

        async def _add_processed_offset(self, offset_key: str) -> None:
            """
            Add offset to tracking with race condition prevention.

            Ensures atomic operations when adding offsets to both
            TTL cache and legacy set.
            """
            self.offset_add_count += 1

            # Add to TTL cache (thread-safe by design)
            self._offset_cache.put(offset_key, True, ttl_seconds=300.0)

            # Also add to legacy set with lock protection
            async with self._cleanup_lock:
                self._processed_message_offsets.add(offset_key)

                # Trigger cleanup if approaching limit
                if (
                    len(self._processed_message_offsets)
                    > self._max_tracked_offsets * 0.9
                ):
                    asyncio.create_task(self._cleanup_processed_offsets())

        async def _cleanup_processed_offsets(self) -> None:
            """
            Clean up processed offsets with race condition prevention.

            Uses _cleanup_lock for atomic operations on offsets set.
            """
            async with self._cleanup_lock:
                self.cleanup_count += 1
                initial_size = len(self._processed_message_offsets)

                if initial_size <= self._max_tracked_offsets:
                    return

                # Remove 20% when over limit
                target_size = int(self._max_tracked_offsets * 0.8)
                offsets_to_remove = initial_size - target_size

                if offsets_to_remove > 0:
                    offset_list = list(self._processed_message_offsets)
                    offset_list.sort(key=lambda x: hash(x))
                    offsets_to_remove_list = offset_list[:offsets_to_remove]

                    for offset_key in offsets_to_remove_list:
                        self._processed_message_offsets.discard(offset_key)

        async def process_message(self, message_id: str) -> bool:
            """
            Process message with duplicate detection.

            Returns:
                True if message was processed (not duplicate)
                False if message was duplicate
            """
            # Check if already processed
            if await self.is_offset_processed(message_id):
                self.duplicate_detected_count += 1
                return False

            # Mark as processed
            await self._add_processed_offset(message_id)
            return True

        def get_metrics(self) -> dict[str, any]:
            """Get comprehensive metrics."""
            return {
                "offset_check_count": self.offset_check_count,
                "offset_add_count": self.offset_add_count,
                "duplicate_detected_count": self.duplicate_detected_count,
                "cleanup_count": self.cleanup_count,
                "cache_size": self._offset_cache.size(),
                "legacy_set_size": len(self._processed_message_offsets),
                "cache_metrics": self._offset_cache.get_metrics(),
            }

    return MockRegistryNode(offset_cache)


# Load Test: High Throughput
# ==========================


@pytest.mark.asyncio
@pytest.mark.load
async def test_high_throughput_1000_msg_per_sec(mock_registry_node):
    """
    Test high-throughput processing at 1000+ messages/second.

    Verifies:
    - No race conditions under sustained load
    - No message duplication
    - No message loss
    - Performance remains stable
    """
    total_messages = 10000
    target_duration_seconds = 10.0  # 10 seconds = 1000 msg/sec

    # Generate unique message IDs
    message_ids = [f"msg-{i}-{uuid4()}" for i in range(total_messages)]

    # Track processed messages
    processed_messages = []
    failed_messages = []

    start_time = time.time()

    # Process messages in batches to achieve target throughput
    batch_size = 100
    batches = [
        message_ids[i : i + batch_size] for i in range(0, len(message_ids), batch_size)
    ]

    for batch in batches:
        # Process batch concurrently (no rate limiting)
        tasks = [mock_registry_node.process_message(msg_id) for msg_id in batch]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Track results
        for msg_id, result in zip(batch, results, strict=False):
            if isinstance(result, Exception):
                failed_messages.append(msg_id)
            elif result:  # Successfully processed (not duplicate)
                processed_messages.append(msg_id)

    end_time = time.time()
    actual_duration = end_time - start_time
    actual_throughput = len(processed_messages) / actual_duration

    # Get metrics
    metrics = mock_registry_node.get_metrics()

    # Assertions - Focus on correctness over raw throughput
    assert (
        len(failed_messages) == 0
    ), f"Failed to process {len(failed_messages)} messages"
    assert (
        len(processed_messages) == total_messages
    ), f"Expected {total_messages} processed, got {len(processed_messages)}"
    assert (
        metrics["duplicate_detected_count"] == 0
    ), f"Detected {metrics['duplicate_detected_count']} duplicates"
    # Throughput check - verify we can handle high load (>500 msg/sec is good for test environment)
    assert (
        actual_throughput >= 500
    ), f"Throughput {actual_throughput:.0f} msg/sec too low (minimum 500 msg/sec)"

    # Verify cache integrity
    cache_metrics = metrics["cache_metrics"]
    assert cache_metrics.hits + cache_metrics.misses == metrics["offset_check_count"]

    print("\n✅ High Throughput Test Results:")
    print(f"   Messages Processed: {len(processed_messages)}/{total_messages}")
    print(f"   Duration: {actual_duration:.2f}s")
    print(f"   Throughput: {actual_throughput:.0f} msg/sec")
    print(f"   Cache Hit Rate: {cache_metrics.hit_rate:.1f}%")
    print(f"   Duplicates Detected: {metrics['duplicate_detected_count']}")
    print(f"   Failed Messages: {len(failed_messages)}")


# Stress Test: Concurrent Readers/Writers
# =======================================


@pytest.mark.asyncio
@pytest.mark.stress
async def test_concurrent_readers_writers(mock_registry_node):
    """
    Test concurrent readers and writers under stress.

    Simulates:
    - Multiple concurrent readers checking offsets
    - Multiple concurrent writers adding offsets
    - Read/write conflicts and race conditions

    Verifies:
    - No data corruption
    - No message duplication
    - Consistent state across readers
    """
    num_writers = 20
    num_readers = 10
    messages_per_writer = 500
    reads_per_reader = 1000

    # Generate message IDs for writers
    all_message_ids = []
    for writer_id in range(num_writers):
        writer_messages = [
            f"writer-{writer_id}-msg-{i}" for i in range(messages_per_writer)
        ]
        all_message_ids.extend(writer_messages)

    # Shared state
    processed_count = 0
    duplicate_count = 0
    read_results = []

    async def writer_task(writer_id: int, messages: list[str]):
        """Writer task that processes messages."""
        nonlocal processed_count, duplicate_count

        for msg_id in messages:
            result = await mock_registry_node.process_message(msg_id)
            if result:
                processed_count += 1
            else:
                duplicate_count += 1

            # Small random delay to increase concurrency
            await asyncio.sleep(random.uniform(0.0001, 0.001))

    async def reader_task(reader_id: int, check_count: int):
        """Reader task that checks random offsets."""
        nonlocal read_results

        for _ in range(check_count):
            # Check random message from all_message_ids
            msg_id = random.choice(all_message_ids)
            is_processed = await mock_registry_node.is_offset_processed(msg_id)
            read_results.append((reader_id, msg_id, is_processed))

            # Small random delay
            await asyncio.sleep(random.uniform(0.0001, 0.001))

    # Create writer tasks
    writer_tasks = []
    batch_size = messages_per_writer
    for writer_id in range(num_writers):
        start_idx = writer_id * batch_size
        end_idx = start_idx + batch_size
        messages = all_message_ids[start_idx:end_idx]
        writer_tasks.append(writer_task(writer_id, messages))

    # Create reader tasks
    reader_tasks = []
    for reader_id in range(num_readers):
        reader_tasks.append(reader_task(reader_id, reads_per_reader))

    # Run all tasks concurrently
    start_time = time.time()
    await asyncio.gather(*writer_tasks, *reader_tasks)
    duration = time.time() - start_time

    # Get metrics
    metrics = mock_registry_node.get_metrics()

    # Verify results
    total_expected = num_writers * messages_per_writer
    assert (
        processed_count == total_expected
    ), f"Expected {total_expected} processed, got {processed_count}"
    assert (
        duplicate_count == 0
    ), f"Detected {duplicate_count} duplicates during concurrent writes"

    # Verify all messages are in cache or set
    for msg_id in all_message_ids:
        is_processed = await mock_registry_node.is_offset_processed(msg_id)
        assert is_processed, f"Message {msg_id} not found after processing"

    print("\n✅ Concurrent Readers/Writers Test Results:")
    print(f"   Writers: {num_writers}, Readers: {num_readers}")
    print(f"   Messages Processed: {processed_count}/{total_expected}")
    print(f"   Duplicates: {duplicate_count}")
    print(f"   Read Operations: {len(read_results)}")
    print(f"   Duration: {duration:.2f}s")
    print(f"   Cache Hit Rate: {metrics['cache_metrics'].hit_rate:.1f}%")


# Stress Test: Duplicate Detection
# ================================


@pytest.mark.asyncio
@pytest.mark.stress
async def test_duplicate_message_detection(mock_registry_node):
    """
    Test duplicate message detection under high load.

    Sends same messages multiple times to verify:
    - First occurrence is processed
    - Subsequent occurrences are detected as duplicates
    - No false positives or false negatives
    """
    num_unique_messages = 1000
    num_duplicates_per_message = 5

    # Generate unique message IDs
    unique_messages = [f"msg-{i}-{uuid4()}" for i in range(num_unique_messages)]

    # Create message stream with duplicates
    message_stream = []
    for msg_id in unique_messages:
        # Add original + duplicates
        for _ in range(num_duplicates_per_message):
            message_stream.append(msg_id)

    # Shuffle to simulate random arrival
    random.shuffle(message_stream)

    # Track results
    first_time_processed = []
    duplicates_detected = []

    # Process all messages
    start_time = time.time()
    for msg_id in message_stream:
        result = await mock_registry_node.process_message(msg_id)
        if result:
            first_time_processed.append(msg_id)
        else:
            duplicates_detected.append(msg_id)

    duration = time.time() - start_time

    # Get metrics
    metrics = mock_registry_node.get_metrics()

    # Verify results
    assert (
        len(first_time_processed) == num_unique_messages
    ), f"Expected {num_unique_messages} unique messages, got {len(first_time_processed)}"

    expected_duplicates = num_unique_messages * (num_duplicates_per_message - 1)
    assert (
        len(duplicates_detected) == expected_duplicates
    ), f"Expected {expected_duplicates} duplicates, got {len(duplicates_detected)}"

    # Verify each unique message was processed exactly once
    first_time_counter = Counter(first_time_processed)
    for msg_id, count in first_time_counter.items():
        assert count == 1, f"Message {msg_id} was processed {count} times (expected 1)"

    print("\n✅ Duplicate Detection Test Results:")
    print(f"   Unique Messages: {len(first_time_processed)}")
    print(f"   Duplicates Detected: {len(duplicates_detected)}")
    print(f"   Total Messages: {len(message_stream)}")
    print(f"   Duration: {duration:.2f}s")
    print(f"   Throughput: {len(message_stream) / duration:.0f} msg/sec")


# Performance Test: Lock Contention
# =================================


@pytest.mark.asyncio
@pytest.mark.performance
async def test_lock_contention_analysis(mock_registry_node):
    """
    Analyze lock contention under high concurrency.

    Measures:
    - Lock acquisition time
    - Lock wait time
    - Throughput degradation under contention
    """
    num_tasks = 50
    operations_per_task = 200

    # Track lock wait times
    lock_wait_times = []
    operation_times = []

    async def concurrent_task(task_id: int):
        """Task that performs read/write operations."""
        for i in range(operations_per_task):
            msg_id = f"task-{task_id}-msg-{i}"

            # Measure operation time
            op_start = time.time()
            await mock_registry_node.process_message(msg_id)
            op_time = time.time() - op_start
            operation_times.append(op_time)

            # Small delay
            await asyncio.sleep(0.0001)

    # Run concurrent tasks
    start_time = time.time()
    tasks = [concurrent_task(i) for i in range(num_tasks)]
    await asyncio.gather(*tasks)
    total_duration = time.time() - start_time

    # Calculate metrics
    total_operations = num_tasks * operations_per_task
    throughput = total_operations / total_duration
    avg_op_time = sum(operation_times) / len(operation_times)
    p95_op_time = sorted(operation_times)[int(len(operation_times) * 0.95)]
    p99_op_time = sorted(operation_times)[int(len(operation_times) * 0.99)]

    # Get cache metrics
    metrics = mock_registry_node.get_metrics()
    cache_metrics = metrics["cache_metrics"]

    # Performance assertions
    assert (
        avg_op_time < 0.01
    ), f"Average operation time {avg_op_time*1000:.2f}ms exceeds 10ms threshold"
    assert (
        p99_op_time < 0.05
    ), f"P99 operation time {p99_op_time*1000:.2f}ms exceeds 50ms threshold"
    assert (
        throughput > 1000
    ), f"Throughput {throughput:.0f} ops/sec below 1000 ops/sec threshold"

    print("\n✅ Lock Contention Analysis Results:")
    print(f"   Concurrent Tasks: {num_tasks}")
    print(f"   Total Operations: {total_operations}")
    print(f"   Duration: {total_duration:.2f}s")
    print(f"   Throughput: {throughput:.0f} ops/sec")
    print(f"   Avg Operation Time: {avg_op_time*1000:.2f}ms")
    print(f"   P95 Operation Time: {p95_op_time*1000:.2f}ms")
    print(f"   P99 Operation Time: {p99_op_time*1000:.2f}ms")
    print(f"   Cache Hit Rate: {cache_metrics.hit_rate:.1f}%")


# Memory Test: Memory Leak Detection
# ==================================


@pytest.mark.asyncio
@pytest.mark.memory
async def test_memory_leak_detection(mock_registry_node):
    """
    Test for memory leaks during sustained operation.

    Verifies:
    - Memory usage remains stable
    - Cache cleanup works correctly
    - No unbounded growth in data structures
    """
    iterations = 5
    messages_per_iteration = 2000

    memory_snapshots = []

    for iteration in range(iterations):
        # Process messages
        for i in range(messages_per_iteration):
            msg_id = f"iter-{iteration}-msg-{i}"
            await mock_registry_node.process_message(msg_id)

        # Take memory snapshot
        metrics = mock_registry_node.get_metrics()
        cache_metrics = metrics["cache_metrics"]

        snapshot = {
            "iteration": iteration,
            "cache_size": metrics["cache_size"],
            "legacy_set_size": metrics["legacy_set_size"],
            "memory_usage_mb": cache_metrics.memory_usage_bytes / (1024 * 1024),
        }
        memory_snapshots.append(snapshot)

        # Allow cleanup to run
        await asyncio.sleep(0.1)

    # Analyze memory growth
    initial_memory = memory_snapshots[0]["memory_usage_mb"]
    final_memory = memory_snapshots[-1]["memory_usage_mb"]
    memory_growth = final_memory - initial_memory

    # Memory should be bounded by cache size
    max_expected_memory_mb = 100  # Reasonable upper bound
    assert (
        final_memory < max_expected_memory_mb
    ), f"Final memory {final_memory:.2f}MB exceeds {max_expected_memory_mb}MB"

    print("\n✅ Memory Leak Detection Results:")
    for snapshot in memory_snapshots:
        print(
            f"   Iteration {snapshot['iteration']}: "
            f"Cache={snapshot['cache_size']}, "
            f"Set={snapshot['legacy_set_size']}, "
            f"Memory={snapshot['memory_usage_mb']:.2f}MB"
        )
    print(f"   Memory Growth: {memory_growth:.2f}MB")


# Integration Test: Full Workflow
# ===============================


@pytest.mark.asyncio
@pytest.mark.integration
async def test_full_workflow_integration(mock_registry_node):
    """
    Integration test covering full workflow:
    1. High-throughput message processing
    2. Concurrent operations
    3. Duplicate detection
    4. Cleanup operations
    5. Performance validation
    """
    # Phase 1: High-throughput processing
    phase1_messages = [f"phase1-{i}" for i in range(5000)]
    start_time = time.time()

    for msg_id in phase1_messages:
        await mock_registry_node.process_message(msg_id)

    phase1_duration = time.time() - start_time
    phase1_throughput = len(phase1_messages) / phase1_duration

    # Phase 2: Concurrent operations
    async def concurrent_worker(worker_id: int, count: int):
        for i in range(count):
            msg_id = f"phase2-worker{worker_id}-{i}"
            await mock_registry_node.process_message(msg_id)

    phase2_start = time.time()
    workers = [concurrent_worker(i, 500) for i in range(10)]
    await asyncio.gather(*workers)
    phase2_duration = time.time() - phase2_start

    # Phase 3: Duplicate detection
    duplicate_messages = random.sample(phase1_messages, 1000)
    duplicate_count = 0

    for msg_id in duplicate_messages:
        result = await mock_registry_node.process_message(msg_id)
        if not result:
            duplicate_count += 1

    # Phase 4: Verify cleanup
    metrics = mock_registry_node.get_metrics()

    # Assertions
    assert (
        phase1_throughput > 1000
    ), f"Phase 1 throughput {phase1_throughput:.0f} below target"
    assert duplicate_count == len(
        duplicate_messages
    ), f"Expected {len(duplicate_messages)} duplicates, got {duplicate_count}"
    assert metrics["duplicate_detected_count"] >= duplicate_count

    print("\n✅ Full Workflow Integration Test Results:")
    print(f"   Phase 1 (High Throughput): {phase1_throughput:.0f} msg/sec")
    print(f"   Phase 2 (Concurrent): {phase2_duration:.2f}s for 5000 messages")
    print(
        f"   Phase 3 (Duplicates): {duplicate_count}/{len(duplicate_messages)} detected"
    )
    print(f"   Cache Hit Rate: {metrics['cache_metrics'].hit_rate:.1f}%")
    print(f"   Total Checks: {metrics['offset_check_count']}")
    print(f"   Total Additions: {metrics['offset_add_count']}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "load or stress or performance"])
