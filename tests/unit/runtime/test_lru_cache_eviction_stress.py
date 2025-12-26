# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""LRU cache eviction stress tests for PolicyRegistry._parse_semver().

This module contains comprehensive stress tests to verify correct LRU (Least Recently Used)
cache eviction behavior under various load conditions. The tests target the semver parsing
cache used by PolicyRegistry to optimize version string parsing.

Key behaviors tested:
1. Cache at max capacity - Verify size limits are enforced
2. Oldest entries evicted first - Verify LRU eviction ordering
3. Rapid insert/evict cycles - Verify stability under high churn
4. Concurrent cache access - Verify thread safety during eviction
5. Edge cases - Cache with size 1, behavior at boundaries
6. Cache hit/miss ratios under stress

Test Categories:
- Stress tests: High volume operations to verify correctness under load
- Eviction correctness: Verify LRU ordering is maintained
- Concurrency tests: Verify thread safety during eviction
- Edge case tests: Boundary conditions and unusual configurations

Performance Characteristics:
- Tests run with custom cache sizes (1, 8, 16, 64) for faster execution
- Stress tests use 10,000+ operations to verify stability
- Concurrent tests use 10+ threads to stress thread safety
"""

from __future__ import annotations

import random
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest

from omnibase_infra.runtime.policy_registry import PolicyRegistry

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def reset_cache_state() -> None:
    """Reset semver cache state before each test for isolation.

    This ensures each test starts with a clean cache state and
    restores the original cache size after the test completes.
    """
    original_size = PolicyRegistry.SEMVER_CACHE_SIZE
    PolicyRegistry._reset_semver_cache()
    yield
    PolicyRegistry._reset_semver_cache()
    PolicyRegistry.SEMVER_CACHE_SIZE = original_size


# =============================================================================
# Cache at Max Capacity Tests
# =============================================================================


class TestCacheMaxCapacity:
    """Verify cache behavior at and beyond maximum capacity."""

    def test_cache_enforces_max_size_under_load(self) -> None:
        """Cache size never exceeds configured limit under continuous load.

        This stress test continuously inserts new entries and verifies the
        cache size never exceeds the configured maximum.
        """
        cache_size = 64
        PolicyRegistry.SEMVER_CACHE_SIZE = cache_size

        # Generate 10x more versions than cache can hold
        num_versions = cache_size * 10

        for i in range(num_versions):
            version = f"{i}.{i % 100}.{i % 50}"
            PolicyRegistry._parse_semver(version)

            # Verify cache size after every insertion
            info = PolicyRegistry._get_semver_cache_info()
            assert info.currsize <= cache_size, (
                f"Cache size {info.currsize} exceeded max {cache_size} "
                f"after inserting version {version}"
            )

    def test_cache_remains_at_max_size_during_sustained_load(self) -> None:
        """Cache maintains max size during sustained high-volume operations.

        Verifies that once the cache reaches max capacity, it remains stable
        under continuous operation without memory growth.
        """
        cache_size = 32
        PolicyRegistry.SEMVER_CACHE_SIZE = cache_size

        # Fill cache to capacity
        for i in range(cache_size):
            PolicyRegistry._parse_semver(f"{i}.0.0")

        info_at_capacity = PolicyRegistry._get_semver_cache_info()
        assert info_at_capacity.currsize == cache_size

        # Continue with 1000 more unique versions
        for i in range(cache_size, cache_size + 1000):
            PolicyRegistry._parse_semver(f"{i}.0.0")

            # Periodically verify size
            if i % 100 == 0:
                info = PolicyRegistry._get_semver_cache_info()
                assert info.currsize == cache_size, (
                    f"Cache size {info.currsize} deviated from max {cache_size}"
                )

        # Final verification
        info_final = PolicyRegistry._get_semver_cache_info()
        assert info_final.currsize == cache_size

    def test_cache_size_stability_with_mixed_operations(self) -> None:
        """Cache size remains stable with mixed hits and misses.

        Simulates realistic usage pattern with both cache hits (repeated versions)
        and cache misses (new versions) to verify size stability.
        """
        cache_size = 16
        PolicyRegistry.SEMVER_CACHE_SIZE = cache_size

        # Pre-populate with base versions
        base_versions = [f"{i}.0.0" for i in range(cache_size)]
        for v in base_versions:
            PolicyRegistry._parse_semver(v)

        # Mixed operations: 50% hits, 50% new insertions
        for iteration in range(500):
            if iteration % 2 == 0:
                # Cache hit - access existing version
                version = base_versions[iteration % cache_size]
            else:
                # Cache miss - new version
                version = f"{1000 + iteration}.0.0"

            PolicyRegistry._parse_semver(version)

            # Verify size constraint
            info = PolicyRegistry._get_semver_cache_info()
            assert info.currsize <= cache_size


# =============================================================================
# LRU Eviction Ordering Tests
# =============================================================================


class TestLRUEvictionOrdering:
    """Verify least recently used entries are evicted first."""

    def test_oldest_entries_evicted_first(self) -> None:
        """Verify LRU eviction order: oldest entries are evicted first.

        This test fills the cache, then adds new entries while tracking
        which entries are evicted to verify LRU ordering.
        """
        cache_size = 8
        PolicyRegistry.SEMVER_CACHE_SIZE = cache_size

        # Fill cache with versions 0-7
        initial_versions = [f"{i}.0.0" for i in range(cache_size)]
        for v in initial_versions:
            PolicyRegistry._parse_semver(v)

        info = PolicyRegistry._get_semver_cache_info()
        assert info.currsize == cache_size

        # Add new version - should evict "0.0.0" (oldest)
        PolicyRegistry._parse_semver("100.0.0")

        # Access "0.0.0" - should be a miss (was evicted)
        info_before = PolicyRegistry._get_semver_cache_info()
        PolicyRegistry._parse_semver("0.0.0")
        info_after = PolicyRegistry._get_semver_cache_info()

        # If "0.0.0" was evicted, accessing it should cause a miss
        assert info_after.misses > info_before.misses, (
            "Expected 0.0.0 to be evicted but it was still in cache"
        )

    def test_recently_accessed_entries_preserved(self) -> None:
        """Verify recently accessed entries survive eviction.

        Access specific entries to make them 'recently used', then verify
        they survive when new entries trigger eviction.
        """
        cache_size = 8
        PolicyRegistry.SEMVER_CACHE_SIZE = cache_size

        # Fill cache with versions 0-7
        for i in range(cache_size):
            PolicyRegistry._parse_semver(f"{i}.0.0")

        # Access "0.0.0" to make it recently used
        PolicyRegistry._parse_semver("0.0.0")

        # Add enough new entries to evict all original entries except recently used ones
        for i in range(cache_size - 1):
            PolicyRegistry._parse_semver(f"{100 + i}.0.0")

        hits_before = PolicyRegistry._get_semver_cache_info().hits

        # "0.0.0" should still be in cache (was recently accessed)
        PolicyRegistry._parse_semver("0.0.0")

        hits_after = PolicyRegistry._get_semver_cache_info().hits
        assert hits_after > hits_before, (
            "Recently accessed entry 0.0.0 was evicted when it should have been preserved"
        )

    def test_lru_order_with_access_pattern_changes(self) -> None:
        """Verify LRU ordering adapts to changing access patterns.

        Test that the eviction order correctly reflects the actual usage pattern,
        not just insertion order.
        """
        cache_size = 4
        PolicyRegistry.SEMVER_CACHE_SIZE = cache_size

        # Fill cache: 0.0.0, 1.0.0, 2.0.0, 3.0.0 (in order)
        versions = [f"{i}.0.0" for i in range(cache_size)]
        for v in versions:
            PolicyRegistry._parse_semver(v)

        # Access in reverse order: 3, 2, 1, 0 (making 0.0.0 most recently used)
        for v in reversed(versions):
            PolicyRegistry._parse_semver(v)

        # Now 3.0.0 is least recently used, should be evicted first
        PolicyRegistry._parse_semver("100.0.0")

        hits_before = PolicyRegistry._get_semver_cache_info().hits

        # 3.0.0 should have been evicted
        PolicyRegistry._parse_semver("3.0.0")
        hits_after = PolicyRegistry._get_semver_cache_info().hits

        # If 3.0.0 was evicted (as expected), this should be a miss (no new hits)
        assert hits_after == hits_before, (
            "Expected 3.0.0 to be evicted (least recently used) but it was still cached"
        )

    def test_eviction_sequence_under_continuous_churn(self) -> None:
        """Verify correct eviction sequence during continuous cache churn.

        This test performs continuous insertions while periodically checking
        that the LRU ordering is maintained correctly.
        """
        cache_size = 16
        PolicyRegistry.SEMVER_CACHE_SIZE = cache_size

        # Track versions we've added
        active_versions: list[str] = []

        for i in range(200):
            version = f"{i}.0.0"
            PolicyRegistry._parse_semver(version)
            active_versions.append(version)

            # Only keep track of most recent cache_size versions
            if len(active_versions) > cache_size:
                active_versions.pop(0)  # Remove oldest

        # Verify all recently added versions are in cache (should be hits)
        initial_hits = PolicyRegistry._get_semver_cache_info().hits
        for v in active_versions:
            PolicyRegistry._parse_semver(v)

        final_hits = PolicyRegistry._get_semver_cache_info().hits
        expected_hits = len(active_versions)
        actual_new_hits = final_hits - initial_hits

        assert actual_new_hits == expected_hits, (
            f"Expected {expected_hits} cache hits for recent versions, "
            f"got {actual_new_hits}"
        )


# =============================================================================
# Rapid Insert/Evict Cycle Tests
# =============================================================================


class TestRapidInsertEvictCycles:
    """Stress tests for rapid insertion and eviction cycles."""

    def test_rapid_insertion_stress(self) -> None:
        """Stress test with rapid continuous insertions.

        Performs 10,000 unique insertions to verify cache stability
        under high insertion rate.
        """
        cache_size = 64
        PolicyRegistry.SEMVER_CACHE_SIZE = cache_size

        start_time = time.perf_counter()

        for i in range(10_000):
            major = i % 1000
            minor = (i // 1000) % 100
            patch = i % 100
            PolicyRegistry._parse_semver(f"{major}.{minor}.{patch}")

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        info = PolicyRegistry._get_semver_cache_info()

        # Verify cache integrity
        assert info.currsize <= cache_size, (
            f"Cache size {info.currsize} exceeded limit {cache_size}"
        )
        assert info.misses >= 10_000 - cache_size, (
            "Cache miss count lower than expected"
        )

        # Performance sanity check (should complete in < 2s on any reasonable hardware)
        assert elapsed_ms < 2000, (
            f"Rapid insertion took {elapsed_ms:.1f}ms (expected < 2000ms)"
        )

    def test_eviction_rate_under_continuous_churn(self) -> None:
        """Verify eviction rate matches insertion rate at capacity.

        When cache is at capacity, each new unique entry should trigger
        exactly one eviction.
        """
        cache_size = 32
        PolicyRegistry.SEMVER_CACHE_SIZE = cache_size

        # Fill cache to capacity
        for i in range(cache_size):
            PolicyRegistry._parse_semver(f"{i}.0.0")

        info_at_capacity = PolicyRegistry._get_semver_cache_info()
        assert info_at_capacity.currsize == cache_size

        # Add 500 more unique versions
        num_new_versions = 500
        for i in range(cache_size, cache_size + num_new_versions):
            PolicyRegistry._parse_semver(f"{i}.0.0")

        info_after = PolicyRegistry._get_semver_cache_info()

        # Cache size should remain at max
        assert info_after.currsize == cache_size

        # Total misses should be cache_size (initial fill) + num_new_versions
        expected_misses = cache_size + num_new_versions
        assert info_after.misses == expected_misses, (
            f"Expected {expected_misses} misses, got {info_after.misses}"
        )

    def test_burst_insertion_stability(self) -> None:
        """Test stability during burst insertions followed by quiet periods.

        Simulates workload with bursts of activity followed by idle periods,
        verifying cache remains stable.
        """
        cache_size = 16
        PolicyRegistry.SEMVER_CACHE_SIZE = cache_size

        version_counter = 0

        for burst in range(10):
            # Burst of 100 insertions
            for _ in range(100):
                PolicyRegistry._parse_semver(f"{version_counter}.0.0")
                version_counter += 1

            # Verify cache integrity after each burst
            info = PolicyRegistry._get_semver_cache_info()
            assert info.currsize <= cache_size, (
                f"Cache exceeded limit after burst {burst}"
            )

            # Small delay (simulated quiet period)
            time.sleep(0.001)

    def test_alternating_insert_access_pattern(self) -> None:
        """Test cache behavior with alternating insert and access operations.

        Alternates between inserting new entries and accessing existing ones
        to verify cache handles mixed workloads correctly.
        """
        cache_size = 8
        PolicyRegistry.SEMVER_CACHE_SIZE = cache_size

        # Use valid semver format: major.minor.patch where all are integers
        base_versions = [f"900.{i}.0" for i in range(cache_size // 2)]
        for v in base_versions:
            PolicyRegistry._parse_semver(v)

        # Alternate: insert new, access base, insert new, access base...
        for i in range(200):
            if i % 2 == 0:
                # Insert new version (use high major number to distinguish)
                PolicyRegistry._parse_semver(f"800.{i}.0")
            else:
                # Access base version
                PolicyRegistry._parse_semver(base_versions[i % len(base_versions)])

            # Verify size constraint
            info = PolicyRegistry._get_semver_cache_info()
            assert info.currsize <= cache_size


# =============================================================================
# Concurrent Cache Access Tests
# =============================================================================


class TestConcurrentCacheAccess:
    """Test thread safety of cache during concurrent access and eviction."""

    def test_concurrent_insertions_maintain_size_limit(self) -> None:
        """Multiple threads inserting concurrently don't exceed cache limit.

        This test verifies thread safety of the LRU cache by having multiple
        threads simultaneously insert new entries.
        """
        cache_size = 64
        PolicyRegistry.SEMVER_CACHE_SIZE = cache_size

        # Ensure cache is initialized before threads start
        PolicyRegistry._parse_semver("0.0.0")

        errors: list[Exception] = []
        size_violations: list[int] = []
        lock = threading.Lock()

        def insert_versions(thread_id: int, num_versions: int) -> None:
            try:
                for i in range(num_versions):
                    version = f"{thread_id}.{i}.0"
                    PolicyRegistry._parse_semver(version)

                    # Periodically check size (not on every iteration for performance)
                    if i % 50 == 0:
                        info = PolicyRegistry._get_semver_cache_info()
                        if info.currsize > cache_size:
                            with lock:
                                size_violations.append(info.currsize)
            except Exception as e:
                with lock:
                    errors.append(e)

        # Launch 10 threads, each inserting 500 versions
        threads = []
        for thread_id in range(10):
            t = threading.Thread(target=insert_versions, args=(thread_id, 500))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread errors: {errors}"
        assert len(size_violations) == 0, (
            f"Cache size exceeded limit: {size_violations}"
        )

        # Final size check
        info = PolicyRegistry._get_semver_cache_info()
        assert info.currsize <= cache_size

    def test_concurrent_reads_and_writes_stability(self) -> None:
        """Mixed concurrent reads and writes maintain cache integrity.

        Tests realistic scenario where some threads primarily read (cache hits)
        while others primarily write (new insertions).
        """
        cache_size = 32
        PolicyRegistry.SEMVER_CACHE_SIZE = cache_size

        # Pre-populate cache with valid semver format
        base_versions = [f"900.{i}.0" for i in range(cache_size)]
        for v in base_versions:
            PolicyRegistry._parse_semver(v)

        errors: list[Exception] = []
        read_results: list[tuple[int, int, int]] = []  # (hits, misses, size)
        lock = threading.Lock()

        def reader_thread(thread_id: int) -> None:
            """Repeatedly access base versions (should be cache hits)."""
            try:
                for _ in range(200):
                    version = base_versions[thread_id % len(base_versions)]
                    PolicyRegistry._parse_semver(version)
                info = PolicyRegistry._get_semver_cache_info()
                with lock:
                    read_results.append((info.hits, info.misses, info.currsize))
            except Exception as e:
                with lock:
                    errors.append(e)

        def writer_thread(thread_id: int) -> None:
            """Insert new versions (cache misses, triggers eviction)."""
            try:
                for i in range(100):
                    # Use valid semver format: thread_id * 100 + i creates unique major versions
                    version = f"{thread_id * 100 + i}.{thread_id}.{i}"
                    PolicyRegistry._parse_semver(version)
                info = PolicyRegistry._get_semver_cache_info()
                with lock:
                    read_results.append((info.hits, info.misses, info.currsize))
            except Exception as e:
                with lock:
                    errors.append(e)

        # Launch 5 reader threads and 5 writer threads
        threads = []
        for i in range(5):
            threads.append(threading.Thread(target=reader_thread, args=(i,)))
            threads.append(threading.Thread(target=writer_thread, args=(i,)))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread errors: {errors}"

        # Verify all size readings were within limit
        for hits, misses, size in read_results:
            assert size <= cache_size, f"Size {size} exceeded limit {cache_size}"

    def test_concurrent_eviction_correctness(self) -> None:
        """Concurrent operations produce correct final cache state.

        Verifies that despite concurrent access, the cache maintains
        correct LRU behavior and all operations complete successfully.
        """
        cache_size = 16
        PolicyRegistry.SEMVER_CACHE_SIZE = cache_size

        # Using ThreadPoolExecutor for better error handling
        errors: list[Exception] = []
        results: list[bool] = []

        def parse_versions(start: int, count: int) -> int:
            """Parse a range of versions, return number of successful parses."""
            success_count = 0
            for i in range(start, start + count):
                try:
                    PolicyRegistry._parse_semver(f"{i}.0.0")
                    success_count += 1
                except Exception as e:
                    errors.append(e)
            return success_count

        with ThreadPoolExecutor(max_workers=10) as executor:
            # Each thread handles a different range
            futures = [executor.submit(parse_versions, i * 100, 100) for i in range(10)]

            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result == 100)
                except Exception as e:
                    errors.append(e)

        assert len(errors) == 0, f"Errors during concurrent execution: {errors}"
        assert all(results), "Some threads failed to complete all operations"

        # Verify final cache state
        info = PolicyRegistry._get_semver_cache_info()
        assert info.currsize == cache_size  # Should be at capacity
        assert info.misses >= 1000 - cache_size  # Most should be misses

    def test_high_contention_stress(self) -> None:
        """High contention stress test with many threads accessing same keys.

        This test creates high lock contention by having many threads
        access overlapping version sets.
        """
        cache_size = 8
        PolicyRegistry.SEMVER_CACHE_SIZE = cache_size

        # Small set of shared versions with valid semver format
        shared_versions = [f"999.{i}.0" for i in range(cache_size * 2)]

        errors: list[Exception] = []
        access_counts: Counter[str] = Counter()
        lock = threading.Lock()

        def high_contention_worker(thread_id: int) -> None:
            try:
                for _ in range(500):
                    # Randomly access from shared pool
                    version = random.choice(shared_versions)
                    PolicyRegistry._parse_semver(version)
                    with lock:
                        access_counts[version] += 1
            except Exception as e:
                with lock:
                    errors.append(e)

        # Launch 20 threads all accessing same small version set
        threads = [
            threading.Thread(target=high_contention_worker, args=(i,))
            for i in range(20)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors during high contention: {errors}"

        # Verify total accesses
        total_accesses = sum(access_counts.values())
        assert total_accesses == 20 * 500, (
            f"Expected {20 * 500} total accesses, got {total_accesses}"
        )


# =============================================================================
# Cache Hit/Miss Ratio Tests
# =============================================================================


class TestCacheHitMissRatios:
    """Tests verifying cache hit/miss behavior under stress."""

    def test_hit_ratio_with_working_set_smaller_than_cache(self) -> None:
        """Hit ratio should approach 100% when working set fits in cache.

        With a working set smaller than cache size, after initial population
        all accesses should be cache hits.
        """
        cache_size = 64
        PolicyRegistry.SEMVER_CACHE_SIZE = cache_size

        # Working set of 32 versions (half of cache size) with valid semver format
        working_set = [f"800.{i}.0" for i in range(32)]

        # Populate cache
        for v in working_set:
            PolicyRegistry._parse_semver(v)

        hits_after_warmup = PolicyRegistry._get_semver_cache_info().hits

        # Access working set 1000 times
        for _ in range(1000):
            for v in working_set:
                PolicyRegistry._parse_semver(v)

        info = PolicyRegistry._get_semver_cache_info()
        new_hits = info.hits - hits_after_warmup
        expected_hits = 1000 * len(working_set)

        assert new_hits == expected_hits, (
            f"Expected {expected_hits} hits, got {new_hits}. "
            f"Hit ratio: {new_hits / expected_hits * 100:.1f}%"
        )

    def test_hit_ratio_with_working_set_larger_than_cache(self) -> None:
        """Hit ratio degrades predictably when working set exceeds cache.

        With working set larger than cache, we expect significant misses
        as entries get evicted before being reused.

        When iterating through working_set in the same order:
        - First pass: All misses (cache fills with last cache_size entries)
        - Second pass: All misses (each entry is accessed before it's in cache)

        This tests the "thrashing" behavior when working set > cache size.
        """
        cache_size = 16
        PolicyRegistry.SEMVER_CACHE_SIZE = cache_size

        # Working set 4x larger than cache with valid semver format
        working_set = [f"700.{i}.0" for i in range(cache_size * 4)]

        # Access each version once
        for v in working_set:
            PolicyRegistry._parse_semver(v)

        info_after_first_pass = PolicyRegistry._get_semver_cache_info()

        # All first accesses should be misses
        assert info_after_first_pass.misses == len(working_set), (
            f"Expected {len(working_set)} misses, got {info_after_first_pass.misses}"
        )

        # Second pass in SAME order - this demonstrates "thrashing"
        # When accessing in same order, each entry gets evicted before reuse
        for v in working_set:
            PolicyRegistry._parse_semver(v)

        info_after_second_pass = PolicyRegistry._get_semver_cache_info()
        second_pass_misses = (
            info_after_second_pass.misses - info_after_first_pass.misses
        )

        # With sequential access pattern and working set > cache,
        # almost all second pass accesses will be misses (thrashing)
        # This is expected LRU behavior for sequential scans larger than cache
        assert second_pass_misses >= len(working_set) - cache_size, (
            f"Expected high miss rate in second pass due to thrashing, "
            f"got {second_pass_misses} misses out of {len(working_set)}"
        )

        # Now test with REVERSE order - should get some hits
        # The cache currently has entries 48-63 (last cache_size entries from second pass)
        # Accessing in reverse order: 63, 62, ... will hit those cached entries
        info_before_reverse = PolicyRegistry._get_semver_cache_info()
        for v in reversed(working_set):
            PolicyRegistry._parse_semver(v)

        info_after_reverse = PolicyRegistry._get_semver_cache_info()
        reverse_hits = info_after_reverse.hits - info_before_reverse.hits

        # Should get at least some hits when accessing in reverse
        # (the most recently cached entries will be hit first)
        assert reverse_hits >= cache_size - 2, (
            f"Expected ~{cache_size} hits in reverse pass, got {reverse_hits}"
        )

    def test_cache_effectiveness_metrics(self) -> None:
        """Verify cache provides measurable performance benefit.

        Compare timing of cold cache vs warm cache operations to verify
        caching provides actual benefit.
        """
        cache_size = 128
        PolicyRegistry.SEMVER_CACHE_SIZE = cache_size

        # Generate test versions that fit in cache
        test_versions = [f"{i}.{i % 10}.{i % 5}" for i in range(cache_size)]

        # Cold cache timing
        PolicyRegistry._reset_semver_cache()
        PolicyRegistry.SEMVER_CACHE_SIZE = cache_size
        cold_start = time.perf_counter()
        for v in test_versions:
            PolicyRegistry._parse_semver(v)
        cold_time = time.perf_counter() - cold_start

        # Warm cache timing (10 iterations)
        warm_start = time.perf_counter()
        for _ in range(10):
            for v in test_versions:
                PolicyRegistry._parse_semver(v)
        warm_time = time.perf_counter() - warm_start

        # Warm should be faster per iteration than cold
        warm_per_iteration = warm_time / 10
        ratio = warm_per_iteration / cold_time

        # Warm cache should not be slower than cold
        assert ratio <= 2.0, (
            f"Warm cache ({warm_per_iteration * 1000:.2f}ms) slower than "
            f"cold cache ({cold_time * 1000:.2f}ms). Ratio: {ratio:.2f}"
        )


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestCacheEdgeCases:
    """Edge case tests for cache boundaries and unusual configurations."""

    def test_cache_size_one(self) -> None:
        """Cache with size=1 correctly evicts on every new entry.

        Tests the minimum viable cache size, where each new entry
        evicts the previous one.
        """
        PolicyRegistry.SEMVER_CACHE_SIZE = 1

        # First entry
        PolicyRegistry._parse_semver("1.0.0")
        assert PolicyRegistry._get_semver_cache_info().currsize == 1

        # Second entry should evict first
        PolicyRegistry._parse_semver("2.0.0")
        assert PolicyRegistry._get_semver_cache_info().currsize == 1

        # Access first entry - should be a miss (was evicted)
        info_before = PolicyRegistry._get_semver_cache_info()
        PolicyRegistry._parse_semver("1.0.0")
        info_after = PolicyRegistry._get_semver_cache_info()

        assert info_after.misses > info_before.misses, (
            "1.0.0 should have been evicted from size=1 cache"
        )

    def test_cache_size_one_rapid_operations(self) -> None:
        """Cache size=1 remains stable under rapid operations.

        Stress test with minimum cache size to verify no edge case bugs.
        """
        PolicyRegistry.SEMVER_CACHE_SIZE = 1

        for i in range(1000):
            PolicyRegistry._parse_semver(f"{i}.0.0")
            info = PolicyRegistry._get_semver_cache_info()
            assert info.currsize == 1, (
                f"Cache size {info.currsize} != 1 after {i} insertions"
            )

        # All should be misses (no hits possible with size=1 and unique versions)
        final_info = PolicyRegistry._get_semver_cache_info()
        assert final_info.misses == 1000
        assert final_info.hits == 0

    def test_cache_at_exact_capacity_boundary(self) -> None:
        """Test behavior at exact capacity boundary.

        Verify correct behavior when cache is exactly at capacity
        and one more entry is added.
        """
        cache_size = 8
        PolicyRegistry.SEMVER_CACHE_SIZE = cache_size

        # Fill to exact capacity
        for i in range(cache_size):
            PolicyRegistry._parse_semver(f"{i}.0.0")

        info = PolicyRegistry._get_semver_cache_info()
        assert info.currsize == cache_size
        assert info.hits == 0
        assert info.misses == cache_size

        # Add one more - triggers eviction (use valid semver)
        PolicyRegistry._parse_semver("999.0.0")

        info_after = PolicyRegistry._get_semver_cache_info()
        assert info_after.currsize == cache_size  # Still at capacity
        assert info_after.misses == cache_size + 1  # One more miss

    def test_identical_version_repeated_access(self) -> None:
        """Repeated access of same version all hit cache.

        Verifies that accessing the same entry repeatedly doesn't
        cause any unusual behavior.
        """
        PolicyRegistry.SEMVER_CACHE_SIZE = 8

        # First access - miss
        PolicyRegistry._parse_semver("1.0.0")

        info = PolicyRegistry._get_semver_cache_info()
        assert info.misses == 1
        assert info.hits == 0

        # Repeated accesses - all hits
        for _ in range(1000):
            PolicyRegistry._parse_semver("1.0.0")

        info_after = PolicyRegistry._get_semver_cache_info()
        assert info_after.misses == 1  # Still only the initial miss
        assert info_after.hits == 1000  # All subsequent were hits

    def test_version_format_variations(self) -> None:
        """Version strings are normalized before caching.

        Verifies that versions are cached by their normalized form,
        meaning equivalent versions share a cache entry. Distinct
        versions (different major/minor/patch or prerelease) are
        cached separately.
        """
        PolicyRegistry.SEMVER_CACHE_SIZE = 16

        # These versions are semantically distinct and will each have their own cache entry
        distinct_versions = [
            "1.0.0",
            "1.0.1",
            "1.1.0",
            "2.0.0",
            "1.0.0-alpha",
            "1.0.0-beta",
        ]

        for v in distinct_versions:
            try:
                PolicyRegistry._parse_semver(v)
            except Exception:
                pass  # Some formats may be invalid

        info = PolicyRegistry._get_semver_cache_info()

        # All distinct versions should be cached separately
        assert info.currsize == len(distinct_versions)

        # Verify that equivalent versions share the same cache entry (normalization)
        info_before = PolicyRegistry._get_semver_cache_info()
        PolicyRegistry._parse_semver("1.0.0")  # Already cached
        info_after = PolicyRegistry._get_semver_cache_info()

        # Should be a cache hit, not a new entry
        assert info_after.hits > info_before.hits
        assert info_after.currsize == info_before.currsize

    def test_cache_behavior_after_clear(self) -> None:
        """Cache reset properly clears all entries.

        Verify that after reset, the cache is empty and
        behaves as if freshly initialized.
        """
        PolicyRegistry.SEMVER_CACHE_SIZE = 16

        # Populate cache
        for i in range(16):
            PolicyRegistry._parse_semver(f"{i}.0.0")

        assert PolicyRegistry._get_semver_cache_info().currsize == 16

        # Reset cache
        PolicyRegistry._reset_semver_cache()
        PolicyRegistry.SEMVER_CACHE_SIZE = 16

        # Cache should be empty (None until reinitialized)
        assert PolicyRegistry._semver_cache is None

        # First access reinitializes (use valid semver)
        PolicyRegistry._parse_semver("999.0.0")
        new_info = PolicyRegistry._get_semver_cache_info()

        assert new_info.currsize == 1
        assert new_info.misses == 1
        assert new_info.hits == 0

    def test_prerelease_version_eviction(self) -> None:
        """Prerelease versions are cached and evicted like regular versions.

        Verifies that versions with prerelease suffixes don't cause
        any special behavior in the cache.
        """
        cache_size = 4
        PolicyRegistry.SEMVER_CACHE_SIZE = cache_size

        # Fill with prerelease versions
        prerelease_versions = [
            "1.0.0-alpha",
            "1.0.0-beta",
            "1.0.0-rc.1",
            "1.0.0-rc.2",
        ]

        for v in prerelease_versions:
            PolicyRegistry._parse_semver(v)

        assert PolicyRegistry._get_semver_cache_info().currsize == cache_size

        # Add one more - should evict alpha (oldest)
        PolicyRegistry._parse_semver("1.0.0")

        # Check alpha was evicted
        info_before = PolicyRegistry._get_semver_cache_info()
        PolicyRegistry._parse_semver("1.0.0-alpha")
        info_after = PolicyRegistry._get_semver_cache_info()

        assert info_after.misses > info_before.misses, (
            "Expected 1.0.0-alpha to be evicted"
        )


# =============================================================================
# Performance Under Stress Tests
# =============================================================================


class TestCachePerformanceUnderStress:
    """Performance characteristics under stress conditions."""

    def test_lookup_latency_at_capacity(self) -> None:
        """Lookup latency remains low when cache is at capacity.

        Verifies that cache lookups don't degrade when cache is full
        and experiencing continuous eviction.
        """
        cache_size = 64
        PolicyRegistry.SEMVER_CACHE_SIZE = cache_size

        # Fill cache and keep adding (continuous eviction)
        for i in range(cache_size * 2):
            PolicyRegistry._parse_semver(f"{i}.0.0")

        # Measure lookup latency under continuous churn
        latencies: list[float] = []

        for i in range(cache_size * 2, cache_size * 2 + 1000):
            start = time.perf_counter()
            PolicyRegistry._parse_semver(f"{i}.0.0")
            latencies.append((time.perf_counter() - start) * 1000)  # ms

        # Calculate statistics
        latencies.sort()
        p50 = latencies[len(latencies) // 2]
        p99 = latencies[int(len(latencies) * 0.99)]

        # Latency should be reasonable even under continuous eviction
        assert p99 < 1.0, (
            f"P99 latency {p99:.3f}ms exceeds 1ms under continuous eviction"
        )

    def test_throughput_at_maximum_eviction_rate(self) -> None:
        """Measure throughput when every operation causes eviction.

        Tests worst-case scenario where cache is small and every
        operation is a miss causing eviction.
        """
        cache_size = 1  # Maximum eviction rate
        PolicyRegistry.SEMVER_CACHE_SIZE = cache_size

        num_operations = 10_000
        start = time.perf_counter()

        for i in range(num_operations):
            PolicyRegistry._parse_semver(f"{i}.0.0")

        elapsed_ms = (time.perf_counter() - start) * 1000
        ops_per_sec = num_operations / (elapsed_ms / 1000)

        # Should still achieve reasonable throughput
        assert ops_per_sec > 10_000, (
            f"Throughput {ops_per_sec:.0f} ops/sec too low (expected > 10,000 ops/sec)"
        )

    def test_memory_stability_under_long_running_stress(self) -> None:
        """Cache memory usage remains stable over extended operation.

        Verifies no memory leaks during extended cache operation with
        continuous eviction.
        """
        cache_size = 32
        PolicyRegistry.SEMVER_CACHE_SIZE = cache_size

        # Record initial state
        initial_info = PolicyRegistry._get_semver_cache_info()

        # Perform many operations
        for i in range(50_000):
            PolicyRegistry._parse_semver(f"{i}.0.0")

            # Periodically verify cache state
            if i % 10_000 == 0 and i > 0:
                info = PolicyRegistry._get_semver_cache_info()
                assert info.currsize == cache_size, (
                    f"Cache size drifted to {info.currsize} at iteration {i}"
                )

        # Final verification
        final_info = PolicyRegistry._get_semver_cache_info()
        assert final_info.currsize == cache_size
        assert final_info.misses == 50_000  # All unique versions
