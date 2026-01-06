# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Performance tests for Registry Effect Node.

This test suite validates the effect node's behavior under high load
and measures latency characteristics. Tests use mock clients for
consistent timing measurements.

Test Categories:
    1. High Volume Sequential: 1000+ sequential requests
    2. Concurrent Load: 100+ concurrent requests via asyncio.gather
    3. Idempotency Cache Stress: Fill cache to max, verify LRU eviction
    4. Memory Bounds: Verify memory stays bounded under sustained load
    5. Latency Distribution: Measure p50, p95, p99 latencies

Performance Thresholds:
    These thresholds are intentionally lenient for CI environments
    where resources may be constrained. Adjust for dedicated perf testing.

    - Sequential 1000 ops: < 2 seconds (with mocks)
    - Concurrent 100 ops: < 1 second (with mocks)
    - LRU eviction: O(1) per operation
    - Memory: Bounded by max_cache_size

Usage:
    Run all performance tests:
        poetry run pytest tests/performance/registration/effect/ -v

    Skip in normal CI (use marker):
        poetry run pytest -m "not performance" tests/

Related:
    - OMN-954: Registry Effect Node testing requirements
    - InMemoryEffectIdempotencyStore: Primary store under test
    - NodeRegistryEffect: Effect node implementation (when available)
"""

from __future__ import annotations

import asyncio
import sys
import time
from statistics import mean, quantiles, stdev
from typing import TYPE_CHECKING
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from omnibase_infra.idempotency import InMemoryIdempotencyStore
from omnibase_infra.nodes.effects.models.model_effect_idempotency_config import (
    ModelEffectIdempotencyConfig,
)
from omnibase_infra.nodes.effects.store_effect_idempotency_inmemory import (
    InMemoryEffectIdempotencyStore,
)

if TYPE_CHECKING:
    from uuid import UUID


# -----------------------------------------------------------------------------
# Simulated Effect Executor for Load Testing
# -----------------------------------------------------------------------------


class SimulatedEffectExecutor:
    """Simulates Effect node behavior for performance testing.

    Provides a realistic simulation of effect execution patterns
    with idempotency checking and backend operations. Uses mock
    clients for consistent timing measurements.

    Attributes:
        idempotency_store: Store for tracking processed intents.
        consul_client: Mock Consul client.
        postgres_client: Mock PostgreSQL client.
        execution_count: Counter for actual backend calls.
        latencies: List of operation latencies in seconds.
    """

    def __init__(
        self,
        idempotency_store: InMemoryIdempotencyStore,
        consul_client: MagicMock,
        postgres_client: MagicMock,
    ) -> None:
        """Initialize the simulated effect executor.

        Args:
            idempotency_store: Store for idempotency checking.
            consul_client: Mock Consul client.
            postgres_client: Mock PostgreSQL client.
        """
        self.idempotency_store = idempotency_store
        self.consul_client = consul_client
        self.postgres_client = postgres_client
        self.execution_count = 0
        self.latencies: list[float] = []

    async def execute_consul_register(
        self,
        intent_id: UUID,
        service_id: str,
        service_name: str,
        correlation_id: UUID | None = None,
    ) -> bool:
        """Execute Consul service registration with timing.

        Args:
            intent_id: Unique identifier for this intent.
            service_id: Consul service ID.
            service_name: Consul service name.
            correlation_id: Optional correlation ID.

        Returns:
            True if registration succeeded or was already done.
        """
        start = time.perf_counter()
        try:
            is_new = await self.idempotency_store.check_and_record(
                message_id=intent_id,
                domain="consul",
                correlation_id=correlation_id,
            )

            if not is_new:
                return True

            self.execution_count += 1
            await self.consul_client.agent.service.register(
                service_id=service_id,
                name=service_name,
            )
            return True
        finally:
            self.latencies.append(time.perf_counter() - start)


# -----------------------------------------------------------------------------
# High Volume Sequential Tests
# -----------------------------------------------------------------------------


class TestHighVolumeSequential:
    """Test high volume sequential request processing."""

    @pytest.mark.asyncio
    async def test_1000_sequential_registrations(
        self,
        inmemory_idempotency_store: InMemoryIdempotencyStore,
        fast_mock_consul_client: MagicMock,
        fast_mock_postgres_client: MagicMock,
    ) -> None:
        """Test 1000 sequential registrations complete within threshold.

        Validates that sequential processing of 1000 unique registrations
        completes within acceptable time bounds using mock backends.

        Performance Target:
            < 2 seconds for 1000 operations (lenient for CI)
        """
        executor = SimulatedEffectExecutor(
            idempotency_store=inmemory_idempotency_store,
            consul_client=fast_mock_consul_client,
            postgres_client=fast_mock_postgres_client,
        )

        start = time.perf_counter()
        for i in range(1000):
            result = await executor.execute_consul_register(
                intent_id=uuid4(),
                service_id=f"node-effect-{i}",
                service_name="onex-effect",
                correlation_id=uuid4(),
            )
            assert result is True

        elapsed = time.perf_counter() - start

        # All should execute (unique intent IDs)
        assert executor.execution_count == 1000

        # Performance assertion (lenient for CI)
        assert elapsed < 2.0, f"Sequential 1000 ops took {elapsed:.2f}s, expected < 2s"

    @pytest.mark.asyncio
    async def test_sequential_with_duplicates(
        self,
        inmemory_idempotency_store: InMemoryIdempotencyStore,
        fast_mock_consul_client: MagicMock,
        fast_mock_postgres_client: MagicMock,
    ) -> None:
        """Test sequential processing with 50% duplicate rate.

        Validates that idempotency deduplication works correctly
        and improves throughput for duplicate requests.
        """
        executor = SimulatedEffectExecutor(
            idempotency_store=inmemory_idempotency_store,
            consul_client=fast_mock_consul_client,
            postgres_client=fast_mock_postgres_client,
        )

        # Create 500 unique intent IDs
        unique_intents = [uuid4() for _ in range(500)]

        start = time.perf_counter()
        # Process each intent twice (1000 total, 500 unique)
        for intent_id in unique_intents:
            await executor.execute_consul_register(
                intent_id=intent_id,
                service_id="node-effect-dup",
                service_name="onex-effect",
            )
            await executor.execute_consul_register(
                intent_id=intent_id,
                service_id="node-effect-dup",
                service_name="onex-effect",
            )

        elapsed = time.perf_counter() - start

        # Only 500 should actually execute
        assert executor.execution_count == 500

        # Should be faster than 1000 unique due to dedup
        assert elapsed < 2.0, f"Sequential with dups took {elapsed:.2f}s, expected < 2s"


# -----------------------------------------------------------------------------
# Concurrent Load Tests
# -----------------------------------------------------------------------------


class TestConcurrentLoad:
    """Test concurrent request processing."""

    @pytest.mark.asyncio
    async def test_100_concurrent_registrations(
        self,
        inmemory_idempotency_store: InMemoryIdempotencyStore,
        fast_mock_consul_client: MagicMock,
        fast_mock_postgres_client: MagicMock,
    ) -> None:
        """Test 100 concurrent registrations via asyncio.gather.

        Validates that concurrent processing of registrations
        works correctly with proper locking and no race conditions.

        Performance Target:
            < 1 second for 100 concurrent operations
        """
        executor = SimulatedEffectExecutor(
            idempotency_store=inmemory_idempotency_store,
            consul_client=fast_mock_consul_client,
            postgres_client=fast_mock_postgres_client,
        )

        async def register_one(index: int) -> bool:
            return await executor.execute_consul_register(
                intent_id=uuid4(),
                service_id=f"node-effect-{index}",
                service_name="onex-effect",
                correlation_id=uuid4(),
            )

        start = time.perf_counter()
        results = await asyncio.gather(*[register_one(i) for i in range(100)])
        elapsed = time.perf_counter() - start

        # All should succeed
        assert len(results) == 100
        assert all(r is True for r in results)
        assert executor.execution_count == 100

        # Performance assertion
        assert elapsed < 1.0, f"Concurrent 100 ops took {elapsed:.2f}s, expected < 1s"

    @pytest.mark.asyncio
    async def test_500_concurrent_registrations(
        self,
        inmemory_idempotency_store: InMemoryIdempotencyStore,
        fast_mock_consul_client: MagicMock,
        fast_mock_postgres_client: MagicMock,
    ) -> None:
        """Test 500 concurrent registrations (stress test).

        Higher concurrency stress test to validate lock contention
        and memory safety under heavy load.

        Performance Target:
            < 3 seconds for 500 concurrent operations
        """
        executor = SimulatedEffectExecutor(
            idempotency_store=inmemory_idempotency_store,
            consul_client=fast_mock_consul_client,
            postgres_client=fast_mock_postgres_client,
        )

        async def register_one(index: int) -> bool:
            return await executor.execute_consul_register(
                intent_id=uuid4(),
                service_id=f"node-effect-{index}",
                service_name="onex-effect",
            )

        start = time.perf_counter()
        results = await asyncio.gather(*[register_one(i) for i in range(500)])
        elapsed = time.perf_counter() - start

        assert len(results) == 500
        assert all(r is True for r in results)
        assert executor.execution_count == 500

        assert elapsed < 3.0, f"Concurrent 500 ops took {elapsed:.2f}s, expected < 3s"

    @pytest.mark.asyncio
    async def test_concurrent_with_duplicate_intents(
        self,
        inmemory_idempotency_store: InMemoryIdempotencyStore,
        fast_mock_consul_client: MagicMock,
        fast_mock_postgres_client: MagicMock,
    ) -> None:
        """Test concurrent processing of duplicate intents.

        Validates that concurrent duplicate detection works correctly
        and only one execution occurs per unique intent.
        """
        executor = SimulatedEffectExecutor(
            idempotency_store=inmemory_idempotency_store,
            consul_client=fast_mock_consul_client,
            postgres_client=fast_mock_postgres_client,
        )

        # Single intent ID, processed concurrently 50 times
        shared_intent_id = uuid4()

        async def register_same() -> bool:
            return await executor.execute_consul_register(
                intent_id=shared_intent_id,
                service_id="node-effect-shared",
                service_name="onex-effect",
            )

        results = await asyncio.gather(*[register_same() for _ in range(50)])

        # All should report success
        assert len(results) == 50
        assert all(r is True for r in results)

        # Only ONE should actually execute (first one wins race)
        assert executor.execution_count == 1


# -----------------------------------------------------------------------------
# Idempotency Cache Stress Tests
# -----------------------------------------------------------------------------


class TestIdempotencyCacheStress:
    """Test idempotency store under stress conditions."""

    @pytest.mark.asyncio
    async def test_lru_eviction_under_load(self) -> None:
        """Test LRU eviction works correctly under sustained load.

        Fills cache to max and verifies that:
        1. Cache size stays bounded at max_cache_size
        2. LRU eviction removes oldest entries
        3. Most recent entries are preserved
        """
        config = ModelEffectIdempotencyConfig(
            max_cache_size=100,
            cache_ttl_seconds=3600.0,
        )
        store = InMemoryEffectIdempotencyStore(config=config)

        # Track correlation IDs in order
        correlation_ids: list[UUID] = []

        # Add 200 entries (2x cache size)
        for _ in range(200):
            cid = uuid4()
            correlation_ids.append(cid)
            await store.mark_completed(cid, "consul")

        # Cache should be at max size
        cache_size = await store.get_cache_size()
        assert cache_size == 100, f"Cache size {cache_size}, expected 100"

        # First 100 should be evicted (LRU)
        for cid in correlation_ids[:100]:
            is_completed = await store.is_completed(cid, "consul")
            assert not is_completed, f"Entry {cid} should have been evicted"

        # Last 100 should be present
        for cid in correlation_ids[100:]:
            is_completed = await store.is_completed(cid, "consul")
            assert is_completed, f"Entry {cid} should be present"

    @pytest.mark.asyncio
    async def test_cache_stress_10k_entries(self) -> None:
        """Test cache performance with 10,000 entries.

        Validates that the default cache size (10k) works correctly
        and operations remain fast.
        """
        store = InMemoryEffectIdempotencyStore()  # Default 10k max

        correlation_ids: list[UUID] = []

        start = time.perf_counter()
        # Fill to capacity
        for _ in range(10000):
            cid = uuid4()
            correlation_ids.append(cid)
            await store.mark_completed(cid, "consul")

        fill_time = time.perf_counter() - start

        # All 10k should be present
        cache_size = await store.get_cache_size()
        assert cache_size == 10000

        # Read performance check
        start = time.perf_counter()
        for cid in correlation_ids[:1000]:  # Sample 1000 reads
            await store.is_completed(cid, "consul")
        read_time = time.perf_counter() - start

        # Performance assertions (lenient for CI)
        assert fill_time < 5.0, f"Fill 10k took {fill_time:.2f}s, expected < 5s"
        assert read_time < 1.0, f"Read 1k took {read_time:.2f}s, expected < 1s"

    @pytest.mark.asyncio
    async def test_concurrent_lru_eviction(self) -> None:
        """Test LRU eviction under concurrent write load.

        Validates that concurrent writes with eviction don't cause
        corruption or deadlocks.
        """
        config = ModelEffectIdempotencyConfig(
            max_cache_size=50,
            cache_ttl_seconds=3600.0,
        )
        store = InMemoryEffectIdempotencyStore(config=config)

        async def add_entries(batch_id: int) -> None:
            for i in range(100):
                await store.mark_completed(uuid4(), f"backend_{batch_id}_{i}")

        # Run 10 concurrent batches (1000 total entries, 50 max cache)
        start = time.perf_counter()
        await asyncio.gather(*[add_entries(i) for i in range(10)])
        elapsed = time.perf_counter() - start

        # Cache should be at max
        cache_size = await store.get_cache_size()
        assert cache_size == 50

        # Should complete without deadlock
        assert elapsed < 3.0, f"Concurrent eviction took {elapsed:.2f}s"


# -----------------------------------------------------------------------------
# Memory Bounds Tests
# -----------------------------------------------------------------------------


class TestMemoryBounds:
    """Test that memory usage stays bounded."""

    @pytest.mark.asyncio
    async def test_memory_bounded_under_sustained_load(self) -> None:
        """Test memory stays bounded under sustained write load.

        Continuously writes to cache and verifies memory doesn't grow
        beyond expected bounds.
        """
        config = ModelEffectIdempotencyConfig(
            max_cache_size=1000,
            cache_ttl_seconds=3600.0,
        )
        store = InMemoryEffectIdempotencyStore(config=config)

        # Process 10,000 entries through 1000-entry cache
        for _ in range(10000):
            await store.mark_completed(uuid4(), "consul")

        # Cache should be bounded
        cache_size = await store.get_cache_size()
        assert cache_size <= 1000, f"Cache grew to {cache_size}, expected <= 1000"

        # Memory check via sys.getsizeof (rough estimate)
        # Note: This is a rough check; actual memory usage may vary
        # due to Python object overhead
        cache_bytes = sys.getsizeof(store._cache)
        # With 1000 entries at ~100 bytes each, expect ~100KB max
        # But getsizeof only measures shallow size, so we check < 1MB
        assert cache_bytes < 1_000_000, f"Cache size {cache_bytes} bytes"

    @pytest.mark.asyncio
    async def test_multi_backend_entries_bounded(self) -> None:
        """Test that entries with many backends stay bounded.

        Each entry can track multiple backends. Verify that adding
        many backends to same entry doesn't cause unbounded growth.
        """
        store = InMemoryEffectIdempotencyStore()

        cid = uuid4()

        # Add 1000 different backends to same correlation ID
        for i in range(1000):
            await store.mark_completed(cid, f"backend_{i}")

        # Should still be single entry in cache
        cache_size = await store.get_cache_size()
        assert cache_size == 1

        # Entry should have all 1000 backends
        completed = await store.get_completed_backends(cid)
        assert len(completed) == 1000


# -----------------------------------------------------------------------------
# Latency Distribution Tests
# -----------------------------------------------------------------------------


class TestLatencyDistribution:
    """Test operation latency characteristics."""

    @pytest.mark.asyncio
    async def test_latency_distribution_1000_ops(
        self,
        inmemory_idempotency_store: InMemoryIdempotencyStore,
        fast_mock_consul_client: MagicMock,
        fast_mock_postgres_client: MagicMock,
    ) -> None:
        """Measure p50, p95, p99 latencies for 1000 operations.

        Collects timing data for 1000 operations and calculates
        latency percentiles to understand performance distribution.
        """
        executor = SimulatedEffectExecutor(
            idempotency_store=inmemory_idempotency_store,
            consul_client=fast_mock_consul_client,
            postgres_client=fast_mock_postgres_client,
        )

        # Execute 1000 operations
        for _ in range(1000):
            await executor.execute_consul_register(
                intent_id=uuid4(),
                service_id="node-effect-latency",
                service_name="onex-effect",
            )

        # Calculate statistics
        latencies = executor.latencies
        assert len(latencies) == 1000

        avg_latency = mean(latencies)
        std_latency = stdev(latencies)

        # Calculate percentiles (quantiles returns n-1 cut points for n parts)
        # For p50, p95, p99 we use 100 quantiles
        percentiles = quantiles(latencies, n=100)
        p50 = percentiles[49]  # 50th percentile
        p95 = percentiles[94]  # 95th percentile
        p99 = percentiles[98]  # 99th percentile

        # Performance assertions (microseconds with mocks)
        # These are very lenient to pass in any CI environment
        assert p50 < 0.01, f"p50 latency {p50 * 1000:.2f}ms, expected < 10ms"
        assert p95 < 0.05, f"p95 latency {p95 * 1000:.2f}ms, expected < 50ms"
        assert p99 < 0.10, f"p99 latency {p99 * 1000:.2f}ms, expected < 100ms"

        # Log statistics for debugging (visible with -v flag)
        print("\nLatency Statistics (1000 ops):")
        print(f"  Mean: {avg_latency * 1000:.3f}ms")
        print(f"  Std:  {std_latency * 1000:.3f}ms")
        print(f"  p50:  {p50 * 1000:.3f}ms")
        print(f"  p95:  {p95 * 1000:.3f}ms")
        print(f"  p99:  {p99 * 1000:.3f}ms")

    @pytest.mark.asyncio
    async def test_read_vs_write_latency(self) -> None:
        """Compare read (is_completed) vs write (mark_completed) latency.

        Validates that read operations are not significantly slower
        than write operations.
        """
        store = InMemoryEffectIdempotencyStore()

        # Pre-populate with some entries
        correlation_ids = [uuid4() for _ in range(100)]
        for cid in correlation_ids:
            await store.mark_completed(cid, "consul")

        # Measure write latency (1000 new entries)
        write_latencies: list[float] = []
        for _ in range(1000):
            start = time.perf_counter()
            await store.mark_completed(uuid4(), "consul")
            write_latencies.append(time.perf_counter() - start)

        # Measure read latency (1000 reads of existing entries)
        read_latencies: list[float] = []
        for _ in range(1000):
            cid = correlation_ids[_ % 100]  # Cycle through existing
            start = time.perf_counter()
            await store.is_completed(cid, "consul")
            read_latencies.append(time.perf_counter() - start)

        avg_write = mean(write_latencies)
        avg_read = mean(read_latencies)

        # Read should not be significantly slower than write
        # (Both should be O(1) operations)
        ratio = avg_read / avg_write if avg_write > 0 else 1.0
        assert ratio < 5.0, f"Read/write ratio {ratio:.2f}, expected < 5"

        print("\nRead vs Write Latency:")
        print(f"  Avg Write: {avg_write * 1000:.4f}ms")
        print(f"  Avg Read:  {avg_read * 1000:.4f}ms")
        print(f"  Ratio:     {ratio:.2f}x")


# -----------------------------------------------------------------------------
# Throughput Tests
# -----------------------------------------------------------------------------


class TestThroughput:
    """Test operation throughput characteristics."""

    @pytest.mark.asyncio
    async def test_sustained_throughput(self) -> None:
        """Measure sustained operations per second.

        Runs continuous operations for a fixed duration and
        calculates throughput.
        """
        store = InMemoryEffectIdempotencyStore()

        target_duration = 1.0  # Run for 1 second
        operations = 0
        start = time.perf_counter()

        while time.perf_counter() - start < target_duration:
            await store.mark_completed(uuid4(), "consul")
            operations += 1

        actual_duration = time.perf_counter() - start
        ops_per_second = operations / actual_duration

        # Should achieve reasonable throughput (at least 10k ops/sec with async)
        assert ops_per_second > 5000, (
            f"Throughput {ops_per_second:.0f} ops/s, expected > 5000"
        )

        print("\nSustained Throughput:")
        print(f"  Operations: {operations}")
        print(f"  Duration:   {actual_duration:.2f}s")
        print(f"  Throughput: {ops_per_second:.0f} ops/sec")

    @pytest.mark.asyncio
    async def test_concurrent_throughput(self) -> None:
        """Measure throughput with concurrent workers.

        Uses multiple concurrent tasks to measure peak throughput.
        """
        store = InMemoryEffectIdempotencyStore()

        num_workers = 10
        ops_per_worker = 1000

        async def worker() -> int:
            count = 0
            for _ in range(ops_per_worker):
                await store.mark_completed(uuid4(), "consul")
                count += 1
            return count

        start = time.perf_counter()
        results = await asyncio.gather(*[worker() for _ in range(num_workers)])
        elapsed = time.perf_counter() - start

        total_ops = sum(results)
        ops_per_second = total_ops / elapsed

        assert total_ops == num_workers * ops_per_worker
        # Concurrent should be faster than sequential
        assert ops_per_second > 10000, (
            f"Concurrent throughput {ops_per_second:.0f} ops/s"
        )

        print(f"\nConcurrent Throughput ({num_workers} workers):")
        print(f"  Total Operations: {total_ops}")
        print(f"  Duration:         {elapsed:.2f}s")
        print(f"  Throughput:       {ops_per_second:.0f} ops/sec")
