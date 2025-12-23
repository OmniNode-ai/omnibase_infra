# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Performance tests for InMemoryEffectIdempotencyStore.

This test suite focuses on idempotency store-specific performance
characteristics including LRU eviction efficiency, TTL cleanup,
and concurrent access patterns.

Test Categories:
    1. LRU Eviction Efficiency: O(1) operations under load
    2. TTL Cleanup Performance: Batch cleanup efficiency
    3. Concurrent Access Scaling: Performance with N workers
    4. Cache Warmup: Cold vs warm cache performance
    5. Mixed Workload: Combined read/write patterns

Usage:
    poetry run pytest tests/performance/registration/effect/test_idempotency_store_performance.py -v

Related:
    - OMN-954: Effect node testing requirements
    - InMemoryEffectIdempotencyStore: Store implementation
"""

from __future__ import annotations

import asyncio
import time
from statistics import mean
from unittest.mock import patch
from uuid import UUID, uuid4

import pytest

from omnibase_infra.nodes.effects.models.model_effect_idempotency_config import (
    ModelEffectIdempotencyConfig,
)
from omnibase_infra.nodes.effects.store_effect_idempotency_inmemory import (
    InMemoryEffectIdempotencyStore,
)

# -----------------------------------------------------------------------------
# LRU Eviction Efficiency Tests
# -----------------------------------------------------------------------------


@pytest.mark.performance
@pytest.mark.unit
class TestLRUEvictionEfficiency:
    """Test LRU eviction maintains O(1) performance."""

    @pytest.mark.asyncio
    async def test_eviction_latency_constant(self) -> None:
        """Verify eviction latency doesn't grow with cache size.

        LRU eviction should be O(1) regardless of cache size.
        This test compares eviction latency at different fill levels.
        """
        config = ModelEffectIdempotencyConfig(
            max_cache_size=1000,
            cache_ttl_seconds=3600.0,
        )
        store = InMemoryEffectIdempotencyStore(config=config)

        # Measure latency at different fill levels
        latencies_by_level: dict[int, list[float]] = {
            100: [],
            500: [],
            1000: [],
            1100: [],  # Triggers eviction
        }

        for level, latency_list in latencies_by_level.items():
            # Reset store
            await store.clear_all()

            # Fill to target level
            for _ in range(level):
                start = time.perf_counter()
                await store.mark_completed(uuid4(), "consul")
                latency_list.append(time.perf_counter() - start)

        # Calculate average latency at each level
        avg_latencies = {
            level: mean(latencies) for level, latencies in latencies_by_level.items()
        }

        # Eviction latency (at 1100) should not be > 10x non-eviction
        eviction_ratio = avg_latencies[1100] / avg_latencies[500]
        assert eviction_ratio < 10.0, (
            f"Eviction slowdown {eviction_ratio:.1f}x, expected < 10x"
        )

        print("\nLRU Eviction Latency by Fill Level:")
        for level, avg in avg_latencies.items():
            print(f"  {level:5d} entries: {avg * 1000:.4f}ms avg")
        print(f"  Eviction ratio: {eviction_ratio:.2f}x")

    @pytest.mark.asyncio
    async def test_bulk_eviction_performance(self) -> None:
        """Test performance when bulk evictions occur.

        When adding many entries that trigger evictions,
        overall throughput should remain acceptable.
        """
        config = ModelEffectIdempotencyConfig(
            max_cache_size=100,
            cache_ttl_seconds=3600.0,
        )
        store = InMemoryEffectIdempotencyStore(config=config)

        # Add 10,000 entries to 100-entry cache (99% evicted)
        start = time.perf_counter()
        for _ in range(10000):
            await store.mark_completed(uuid4(), "consul")
        elapsed = time.perf_counter() - start

        # Should complete quickly despite constant eviction
        assert elapsed < 5.0, f"Bulk eviction took {elapsed:.2f}s, expected < 5s"

        # Cache should be at max
        cache_size = await store.get_cache_size()
        assert cache_size == 100

        ops_per_sec = 10000 / elapsed
        print("\nBulk Eviction Performance:")
        print("  Total ops:  10,000")
        print(f"  Duration:   {elapsed:.2f}s")
        print(f"  Throughput: {ops_per_sec:.0f} ops/sec")


# -----------------------------------------------------------------------------
# TTL Cleanup Performance Tests
# -----------------------------------------------------------------------------


@pytest.mark.performance
@pytest.mark.unit
class TestTTLCleanupPerformance:
    """Test TTL cleanup performance characteristics."""

    @pytest.mark.asyncio
    async def test_cleanup_expired_performance(self) -> None:
        """Test cleanup_expired performance with many expired entries.

        When many entries expire, cleanup should be efficient.
        """
        config = ModelEffectIdempotencyConfig(
            max_cache_size=10000,
            cache_ttl_seconds=1.0,  # 1 second TTL
        )
        store = InMemoryEffectIdempotencyStore(config=config)

        # Add 5000 entries
        start_time = time.monotonic()
        with patch(
            "omnibase_infra.nodes.effects.store_effect_idempotency_inmemory.time.monotonic"
        ) as mock_monotonic:
            mock_monotonic.return_value = start_time

            for _ in range(5000):
                await store.mark_completed(uuid4(), "consul")

            assert await store.get_cache_size() == 5000

            # Expire all entries
            mock_monotonic.return_value = start_time + 2.0  # Past TTL

            # Measure cleanup time
            cleanup_start = time.perf_counter()
            removed = await store.cleanup_expired()
            cleanup_elapsed = time.perf_counter() - cleanup_start

        assert removed == 5000
        assert await store.get_cache_size() == 0

        # Cleanup should be fast (< 1 second for 5000 entries)
        assert cleanup_elapsed < 1.0, f"Cleanup took {cleanup_elapsed:.2f}s"

        print("\nTTL Cleanup Performance:")
        print(f"  Entries cleaned: {removed}")
        print(f"  Duration:        {cleanup_elapsed:.4f}s")
        print(f"  Rate:            {removed / cleanup_elapsed:.0f} entries/sec")

    @pytest.mark.asyncio
    async def test_mixed_expired_fresh_cleanup(self) -> None:
        """Test cleanup with mix of expired and fresh entries.

        Cleanup should only remove expired entries efficiently.
        """
        config = ModelEffectIdempotencyConfig(
            max_cache_size=10000,
            cache_ttl_seconds=10.0,  # 10 second TTL
        )
        store = InMemoryEffectIdempotencyStore(config=config)

        start_time = time.monotonic()
        expired_ids: list[UUID] = []
        fresh_ids: list[UUID] = []

        with patch(
            "omnibase_infra.nodes.effects.store_effect_idempotency_inmemory.time.monotonic"
        ) as mock_monotonic:
            mock_monotonic.return_value = start_time

            # Add old entries (will expire)
            for _ in range(2500):
                cid = uuid4()
                expired_ids.append(cid)
                await store.mark_completed(cid, "consul")

            # Advance time by 8 seconds
            mock_monotonic.return_value = start_time + 8.0

            # Add fresh entries
            for _ in range(2500):
                cid = uuid4()
                fresh_ids.append(cid)
                await store.mark_completed(cid, "consul")

            assert await store.get_cache_size() == 5000

            # Advance time to expire old entries (12 seconds total)
            mock_monotonic.return_value = start_time + 12.0

            # Cleanup
            cleanup_start = time.perf_counter()
            removed = await store.cleanup_expired()
            cleanup_elapsed = time.perf_counter() - cleanup_start

        # Should remove exactly old entries
        assert removed == 2500
        assert await store.get_cache_size() == 2500

        # Fresh entries should remain
        for cid in fresh_ids[:100]:  # Sample check
            assert await store.is_completed(cid, "consul")

        print("\nMixed Cleanup Performance:")
        print(f"  Expired removed: {removed}")
        print(f"  Fresh remaining: {await store.get_cache_size()}")
        print(f"  Duration:        {cleanup_elapsed:.4f}s")


# -----------------------------------------------------------------------------
# Concurrent Access Scaling Tests
# -----------------------------------------------------------------------------


@pytest.mark.performance
@pytest.mark.unit
class TestConcurrentAccessScaling:
    """Test performance scaling with concurrent access."""

    @pytest.mark.asyncio
    async def test_scaling_with_worker_count(self) -> None:
        """Test throughput scaling with increasing worker counts.

        Measure how throughput changes as concurrent workers increase.

        Note: Due to asyncio.Lock serialization, concurrent workers may not
        achieve higher throughput than sequential workers. The test validates
        that the store remains stable and performs reasonably under concurrent
        load, not that it scales linearly.
        """
        store = InMemoryEffectIdempotencyStore()

        worker_counts = [1, 2, 4, 8, 16]
        ops_per_worker = 500
        results: dict[int, float] = {}

        for num_workers in worker_counts:
            # Reset store
            await store.clear_all()

            async def worker() -> int:
                for _ in range(ops_per_worker):
                    await store.mark_completed(uuid4(), "consul")
                return ops_per_worker

            start = time.perf_counter()
            await asyncio.gather(*[worker() for _ in range(num_workers)])
            elapsed = time.perf_counter() - start

            total_ops = num_workers * ops_per_worker
            throughput = total_ops / elapsed
            results[num_workers] = throughput

        # Print scaling results
        print(f"\nConcurrency Scaling ({ops_per_worker} ops/worker):")
        print("  Workers  Throughput      Efficiency")
        base_throughput = results[1]
        for workers, throughput in results.items():
            efficiency = (throughput / workers) / base_throughput * 100
            print(f"  {workers:7d}  {throughput:10.0f}/s  {efficiency:6.1f}%")

        # With asyncio.Lock, concurrent workers may not achieve higher throughput
        # due to serialization. The key invariant is that throughput doesn't
        # collapse completely under concurrent load.
        #
        # NOTE: CI environments have highly variable performance due to shared
        # resources, container overhead, and CPU throttling. A 10% threshold
        # ensures the test validates basic functionality without being flaky.
        # Local testing typically shows 50%+ efficiency.
        min_acceptable_throughput = results[1] * 0.10  # At least 10% of single-worker
        assert results[16] > min_acceptable_throughput, (
            f"16-worker throughput {results[16]:.0f} too low, "
            f"expected > {min_acceptable_throughput:.0f} (10% of single-worker)"
        )

    @pytest.mark.asyncio
    async def test_read_heavy_vs_write_heavy(self) -> None:
        """Compare performance of read-heavy vs write-heavy workloads.

        Tests different read/write ratios to understand contention.
        """
        store = InMemoryEffectIdempotencyStore()

        # Pre-populate
        correlation_ids = [uuid4() for _ in range(1000)]
        for cid in correlation_ids:
            await store.mark_completed(cid, "consul")

        async def read_worker(ops: int) -> int:
            for i in range(ops):
                await store.is_completed(correlation_ids[i % 1000], "consul")
            return ops

        async def write_worker(ops: int) -> int:
            for _ in range(ops):
                await store.mark_completed(uuid4(), "consul")
            return ops

        # Test different ratios
        workloads = {
            "100% writes": ([write_worker(500)] * 4, []),
            "100% reads": ([], [read_worker(500)] * 4),
            "50/50": ([write_worker(500)] * 2, [read_worker(500)] * 2),
            "90% reads": ([write_worker(100)] * 1, [read_worker(500)] * 4),
        }

        results: dict[str, float] = {}
        for name, (writers, readers) in workloads.items():
            await store.clear_all()
            for cid in correlation_ids:
                await store.mark_completed(cid, "consul")

            tasks = writers + readers
            if tasks:
                start = time.perf_counter()
                task_results = await asyncio.gather(*tasks)
                elapsed = time.perf_counter() - start
                total_ops = sum(task_results)
                results[name] = total_ops / elapsed

        print("\nWorkload Mix Performance:")
        for name, throughput in results.items():
            print(f"  {name:15s}: {throughput:10.0f} ops/sec")


# -----------------------------------------------------------------------------
# Cache Warmup Tests
# -----------------------------------------------------------------------------


@pytest.mark.performance
@pytest.mark.unit
class TestCacheWarmup:
    """Test cold vs warm cache performance."""

    @pytest.mark.asyncio
    async def test_cold_vs_warm_cache(self) -> None:
        """Compare performance with cold (empty) vs warm (populated) cache.

        Write operations to an empty cache may differ from writes
        to a near-full cache.
        """
        config = ModelEffectIdempotencyConfig(
            max_cache_size=1000,
            cache_ttl_seconds=3600.0,
        )

        # Cold cache test
        cold_store = InMemoryEffectIdempotencyStore(config=config)
        cold_latencies: list[float] = []
        for _ in range(500):
            start = time.perf_counter()
            await cold_store.mark_completed(uuid4(), "consul")
            cold_latencies.append(time.perf_counter() - start)

        # Warm cache test (90% full)
        warm_store = InMemoryEffectIdempotencyStore(config=config)
        for _ in range(900):
            await warm_store.mark_completed(uuid4(), "consul")

        warm_latencies: list[float] = []
        for _ in range(500):
            start = time.perf_counter()
            await warm_store.mark_completed(uuid4(), "consul")
            warm_latencies.append(time.perf_counter() - start)

        cold_avg = mean(cold_latencies)
        warm_avg = mean(warm_latencies)

        # Warm cache may be slightly slower due to more entries to check
        # but should not be dramatically different
        ratio = warm_avg / cold_avg if cold_avg > 0 else 1.0
        assert ratio < 5.0, f"Warm/cold ratio {ratio:.2f}x, expected < 5x"

        print("\nCold vs Warm Cache:")
        print(f"  Cold cache avg: {cold_avg * 1000:.4f}ms")
        print(f"  Warm cache avg: {warm_avg * 1000:.4f}ms")
        print(f"  Ratio:          {ratio:.2f}x")


# -----------------------------------------------------------------------------
# Mixed Workload Tests
# -----------------------------------------------------------------------------


@pytest.mark.performance
@pytest.mark.unit
class TestMixedWorkload:
    """Test realistic mixed workload patterns."""

    @pytest.mark.asyncio
    async def test_realistic_workload(self) -> None:
        """Simulate realistic workload with mixed operations.

        Realistic workload:
        - Initial registration (mark_completed)
        - Check before second operation (is_completed)
        - Second operation if not completed
        - Occasional cleanup
        """
        store = InMemoryEffectIdempotencyStore()

        operations = 0
        duplicates = 0
        total_latency = 0.0

        # Simulate 1000 registration workflows
        for _ in range(1000):
            cid = uuid4()

            # Step 1: Mark consul completed
            start = time.perf_counter()
            await store.mark_completed(cid, "consul")
            total_latency += time.perf_counter() - start
            operations += 1

            # Step 2: Check if postgres completed (simulating partial failure retry)
            start = time.perf_counter()
            is_done = await store.is_completed(cid, "postgres")
            total_latency += time.perf_counter() - start
            operations += 1

            if not is_done:
                # Step 3: Mark postgres completed
                start = time.perf_counter()
                await store.mark_completed(cid, "postgres")
                total_latency += time.perf_counter() - start
                operations += 1
            else:
                duplicates += 1

        avg_latency = total_latency / operations

        print("\nRealistic Workload Simulation:")
        print(f"  Total operations: {operations}")
        print(f"  Duplicates found: {duplicates}")
        print(f"  Avg latency:      {avg_latency * 1000:.4f}ms")
        print(f"  Total time:       {total_latency:.2f}s")

        # Should complete quickly
        assert total_latency < 5.0, f"Workload took {total_latency:.2f}s"

    @pytest.mark.asyncio
    async def test_burst_pattern(self) -> None:
        """Test burst traffic pattern with pauses.

        Simulates realistic traffic with bursts of activity
        followed by quiet periods.
        """
        store = InMemoryEffectIdempotencyStore()

        burst_size = 100
        num_bursts = 10
        burst_latencies: list[float] = []

        for _ in range(num_bursts):
            # Burst of activity
            start = time.perf_counter()
            for _ in range(burst_size):
                await store.mark_completed(uuid4(), "consul")
            burst_latencies.append(time.perf_counter() - start)

            # Brief pause between bursts
            await asyncio.sleep(0.01)

        avg_burst = mean(burst_latencies)
        total_ops = burst_size * num_bursts

        print("\nBurst Pattern Performance:")
        print(f"  Bursts:           {num_bursts}")
        print(f"  Ops per burst:    {burst_size}")
        print(f"  Avg burst time:   {avg_burst * 1000:.2f}ms")
        print(f"  Burst throughput: {burst_size / avg_burst:.0f} ops/sec")

        # Each burst should complete quickly
        assert avg_burst < 0.5, f"Burst took {avg_burst:.2f}s, expected < 0.5s"


# -----------------------------------------------------------------------------
# Edge Case Performance Tests
# -----------------------------------------------------------------------------


@pytest.mark.performance
@pytest.mark.unit
class TestEdgeCasePerformance:
    """Test performance in edge cases."""

    @pytest.mark.asyncio
    async def test_same_correlation_id_many_backends(self) -> None:
        """Test performance when one correlation ID has many backends.

        Validates that per-entry backend set doesn't degrade performance.
        """
        store = InMemoryEffectIdempotencyStore()

        cid = uuid4()

        # Add 1000 backends to same correlation ID
        start = time.perf_counter()
        for i in range(1000):
            await store.mark_completed(cid, f"backend_{i}")
        write_time = time.perf_counter() - start

        # Check all backends
        start = time.perf_counter()
        for i in range(1000):
            assert await store.is_completed(cid, f"backend_{i}")
        read_time = time.perf_counter() - start

        # Get all backends
        start = time.perf_counter()
        backends = await store.get_completed_backends(cid)
        get_all_time = time.perf_counter() - start

        assert len(backends) == 1000

        print("\nMany Backends Performance:")
        print(f"  Write 1000 backends: {write_time * 1000:.2f}ms")
        print(f"  Read 1000 backends:  {read_time * 1000:.2f}ms")
        print(f"  Get all backends:    {get_all_time * 1000:.4f}ms")

        # All operations should be fast
        assert write_time < 1.0, f"Write took {write_time:.2f}s"
        assert read_time < 1.0, f"Read took {read_time:.2f}s"

    @pytest.mark.asyncio
    async def test_rapid_add_remove_cycle(self) -> None:
        """Test rapid add/remove cycles.

        Simulates workload where entries are added and cleared quickly.
        """
        store = InMemoryEffectIdempotencyStore()

        cycles = 100
        entries_per_cycle = 100

        start = time.perf_counter()
        for _ in range(cycles):
            # Add entries
            cids = []
            for _ in range(entries_per_cycle):
                cid = uuid4()
                cids.append(cid)
                await store.mark_completed(cid, "consul")

            # Clear entries
            for cid in cids:
                await store.clear(cid)

        elapsed = time.perf_counter() - start
        total_ops = cycles * entries_per_cycle * 2  # add + clear

        print("\nRapid Add/Remove Cycle:")
        print(f"  Cycles:     {cycles}")
        print(f"  Total ops:  {total_ops}")
        print(f"  Duration:   {elapsed:.2f}s")
        print(f"  Throughput: {total_ops / elapsed:.0f} ops/sec")

        # Should complete quickly
        assert elapsed < 5.0, f"Cycles took {elapsed:.2f}s"

        # Store should be empty
        assert await store.get_cache_size() == 0
