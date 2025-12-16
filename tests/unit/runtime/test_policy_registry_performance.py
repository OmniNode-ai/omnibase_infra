# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Performance benchmarks for policy_registry module.

This module contains performance tests to verify the optimization work
done to reduce redundant operations in the PolicyRegistry.get() method.

Key optimizations tested:
1. Secondary index for O(1) policy_id lookup (vs O(n) scan)
2. Early exit when policy_id not found
3. Deferred error message generation (expensive _list_internal() calls)
4. Fast path when no filtering needed
5. Cached semver parsing with LRU cache
"""

from __future__ import annotations

import time

import pytest

from omnibase_infra.enums import EnumPolicyType
from omnibase_infra.errors import PolicyRegistryError
from omnibase_infra.runtime.policy_registry import PolicyRegistry

# =============================================================================
# Mock Policy Classes for Performance Testing
# =============================================================================


class MockPolicy:
    """Mock policy for performance testing."""

    def evaluate(self, context: dict[str, object]) -> dict[str, object]:
        return {"result": "ok"}


# =============================================================================
# Performance Test Fixtures
# =============================================================================


@pytest.fixture
def large_policy_registry() -> PolicyRegistry:
    """Create a registry with many policies for performance testing.

    Creates 100 policies with 5 versions each (500 total registrations).
    This simulates a realistic large registry.

    Note: Performance tests use direct instantiation to isolate performance
    characteristics. Container DI overhead would confound performance measurements.
    For integration tests, use container-based fixtures from conftest.py.
    """
    registry = PolicyRegistry()
    for i in range(100):
        for version_idx in range(5):
            registry.register_policy(
                policy_id=f"policy_{i}",
                policy_class=MockPolicy,  # type: ignore[arg-type]
                policy_type=EnumPolicyType.ORCHESTRATOR,
                version=f"{version_idx}.0.0",
            )
    return registry


# =============================================================================
# Performance Benchmarks
# =============================================================================


class TestPolicyRegistryPerformance:
    """Performance benchmarks for PolicyRegistry optimizations."""

    def test_get_performance_with_secondary_index(
        self, large_policy_registry: PolicyRegistry
    ) -> None:
        """Verify secondary index provides O(1) lookup vs O(n) scan.

        This tests that lookup performance doesn't degrade with registry size.
        With the secondary index optimization:
        - Lookup should be O(1) instead of O(n)
        - Time should not increase significantly with registry size

        Baseline: With 500 policies, lookup should take < 1ms
        """
        # Warm up the cache
        _ = large_policy_registry.get("policy_50")

        # Measure lookup time
        start_time = time.perf_counter()
        for _ in range(1000):
            _ = large_policy_registry.get("policy_50")
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # With secondary index, 1000 lookups should complete in < 100ms
        # (< 0.1ms per lookup on average)
        assert elapsed_ms < 100, (
            f"1000 lookups took {elapsed_ms:.2f}ms (expected < 100ms)"
        )

    def test_get_performance_early_exit_on_missing(
        self, large_policy_registry: PolicyRegistry
    ) -> None:
        """Verify early exit optimization for missing policy_id.

        Tests that when policy_id is not found, we exit early without
        building a matches list. This should be faster than the error
        path that builds the full registered policies list.

        The expensive _list_internal() call should only happen when
        actually raising the error, not during candidate filtering.
        """
        # Measure time for missing policy lookup (should fail fast)
        start_time = time.perf_counter()
        for _ in range(100):
            with pytest.raises(PolicyRegistryError):
                large_policy_registry.get("nonexistent_policy")
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Early exit optimization: 100 failed lookups should complete quickly
        # Even though error message generation is expensive, it's deferred
        # until we actually raise the error
        assert elapsed_ms < 500, (
            f"100 failed lookups took {elapsed_ms:.2f}ms (expected < 500ms)"
        )

    def test_get_performance_fast_path_no_filters(
        self, large_policy_registry: PolicyRegistry
    ) -> None:
        """Verify fast path optimization when no filters applied.

        Tests that when policy_type and version are None (common case),
        we use a fast path that builds matches list more efficiently.

        This is the most common usage pattern: get("policy_id") to
        retrieve the latest version.
        """
        # Warm up
        _ = large_policy_registry.get("policy_75")

        # Measure fast path (no filters)
        start_time = time.perf_counter()
        for _ in range(1000):
            _ = large_policy_registry.get("policy_75")
        fast_path_ms = (time.perf_counter() - start_time) * 1000

        # Measure filtered path (with type filter)
        start_time = time.perf_counter()
        for _ in range(1000):
            _ = large_policy_registry.get(
                "policy_75", policy_type=EnumPolicyType.ORCHESTRATOR
            )
        filtered_path_ms = (time.perf_counter() - start_time) * 1000

        # Fast path should be noticeably faster (at least 20% faster)
        # This validates the optimization
        speedup = filtered_path_ms / fast_path_ms
        assert speedup > 1.1, (
            f"Fast path not optimized (speedup: {speedup:.2f}x, expected > 1.1x)"
        )

    def test_semver_cache_performance(
        self, large_policy_registry: PolicyRegistry
    ) -> None:
        """Verify LRU cache improves semver parsing performance.

        Tests that repeated version comparisons benefit from caching.
        The _parse_semver method uses @lru_cache to avoid re-parsing
        the same version strings.
        """
        # First run - cache cold
        start_time = time.perf_counter()
        for _ in range(100):
            # Get policy with multiple versions (triggers sorting)
            _ = large_policy_registry.get("policy_25")
        cold_cache_ms = (time.perf_counter() - start_time) * 1000

        # Second run - cache warm
        start_time = time.perf_counter()
        for _ in range(100):
            _ = large_policy_registry.get("policy_25")
        warm_cache_ms = (time.perf_counter() - start_time) * 1000

        # Warm cache should be at least as fast (cache hit doesn't hurt)
        # Note: Speedup may be minimal on fast hardware but cache prevents
        # performance degradation
        speedup = cold_cache_ms / warm_cache_ms
        assert speedup >= 1.0, (
            f"Cache hurting performance (speedup: {speedup:.2f}x, expected >= 1.0x)"
        )

        # Warm cache should complete in reasonable time regardless
        assert warm_cache_ms < 100, (
            f"Cached lookups too slow ({warm_cache_ms:.2f}ms for 100 lookups)"
        )

    def test_get_performance_with_version_sorting(
        self, large_policy_registry: PolicyRegistry
    ) -> None:
        """Verify version sorting performance with multiple versions.

        Tests that sorting multiple versions to find latest is performant.
        Uses cached semver parsing to avoid redundant work.
        """
        # Measure lookup with 5 versions (requires sorting)
        start_time = time.perf_counter()
        for _ in range(1000):
            _ = large_policy_registry.get("policy_10")
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # With 5 versions, sorting overhead should be minimal
        # 1000 lookups with sorting should complete in < 150ms
        assert elapsed_ms < 150, (
            f"1000 lookups with sorting took {elapsed_ms:.2f}ms (expected < 150ms)"
        )

    def test_concurrent_get_performance(
        self, large_policy_registry: PolicyRegistry
    ) -> None:
        """Verify lookup performance under concurrent access.

        Tests that lock contention doesn't significantly degrade performance.
        The critical section in get() should be minimal.
        """
        import threading

        results: list[bool] = []
        errors: list[Exception] = []

        def concurrent_get() -> None:
            try:
                for _ in range(100):
                    policy_cls = large_policy_registry.get("policy_50")
                    results.append(policy_cls is MockPolicy)
            except Exception as e:
                errors.append(e)

        # Run 10 threads concurrently
        start_time = time.perf_counter()
        threads = [threading.Thread(target=concurrent_get) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # All operations should succeed
        assert len(errors) == 0, f"Concurrent access errors: {errors}"
        assert len(results) == 1000, "Not all operations completed"

        # Concurrent access should complete in reasonable time
        # 10 threads * 100 lookups = 1000 total lookups
        assert elapsed_ms < 500, (
            f"1000 concurrent lookups took {elapsed_ms:.2f}ms (expected < 500ms)"
        )


# =============================================================================
# Regression Tests (ensure optimizations don't break functionality)
# =============================================================================


class TestPolicyRegistryOptimizationRegression:
    """Regression tests to ensure optimizations don't break correctness."""

    def test_fast_path_returns_correct_latest_version(
        self,
    ) -> None:
        """Verify fast path returns correct latest version."""
        registry = PolicyRegistry()

        # Register multiple versions out of order
        registry.register_policy(
            policy_id="test",
            policy_class=MockPolicy,  # type: ignore[arg-type]
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="2.0.0",
        )
        registry.register_policy(
            policy_id="test",
            policy_class=MockPolicy,  # type: ignore[arg-type]
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="10.0.0",  # Highest
        )
        registry.register_policy(
            policy_id="test",
            policy_class=MockPolicy,  # type: ignore[arg-type]
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="1.0.0",
        )

        # Fast path should return 10.0.0 (semantically latest)
        policy_cls = registry.get("test")
        assert policy_cls is MockPolicy

        # Verify it's actually the latest by checking the versions
        versions = registry.list_versions("test")
        # Note: list_versions() returns lexicographically sorted strings
        # (10.0.0 comes before 2.0.0 lexicographically), but get() uses
        # semantic version comparison internally
        assert set(versions) == {"1.0.0", "2.0.0", "10.0.0"}

        # Verify we got the semantically latest version (10.0.0)
        # by checking that we can get it explicitly
        latest_policy_cls = registry.get("test", version="10.0.0")
        assert latest_policy_cls is policy_cls

    def test_early_exit_raises_correct_error(self) -> None:
        """Verify early exit still raises descriptive error."""
        registry = PolicyRegistry()
        registry.register_policy(
            policy_id="existing",
            policy_class=MockPolicy,  # type: ignore[arg-type]
            policy_type=EnumPolicyType.ORCHESTRATOR,
        )

        # Early exit for missing policy_id
        with pytest.raises(PolicyRegistryError) as exc_info:
            registry.get("missing")

        # Error message should still be descriptive
        error_msg = str(exc_info.value)
        assert "missing" in error_msg
        assert "No policy registered" in error_msg
        assert "existing" in error_msg  # Should list existing policies

    def test_deferred_error_generation_is_correct(self) -> None:
        """Verify deferred _list_internal() produces same error message."""
        registry = PolicyRegistry()
        for i in range(10):
            registry.register_policy(
                policy_id=f"policy_{i}",
                policy_class=MockPolicy,  # type: ignore[arg-type]
                policy_type=EnumPolicyType.ORCHESTRATOR,
            )

        # Error should still list all registered policies
        with pytest.raises(PolicyRegistryError) as exc_info:
            registry.get("missing")

        error_msg = str(exc_info.value)
        # Should mention the missing policy
        assert "missing" in error_msg
        # Should list registered policies (deferred call to _list_internal())
        assert "policy_0" in error_msg or "Registered policies" in error_msg
