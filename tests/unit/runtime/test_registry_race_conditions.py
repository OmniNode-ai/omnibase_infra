# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Race condition and concurrent access tests for registry components.

This module provides comprehensive race condition tests for:
- RegistryPolicy: Thread-safe policy registration and versioning
- RegistryProtocolBinding: Thread-safe handler registration
- RegistryEventBusBinding: Thread-safe event bus registration
- Singleton factory functions: Thread-safe lazy initialization

Test Categories:
1. Concurrent Registration: Multiple threads registering simultaneously
2. Concurrent Read/Write: Readers and writers operating together
3. State Consistency: Verifying shared state under concurrent modifications
4. Boundary Conditions: Testing at threshold boundaries (e.g., circuit breaker)
5. Stress Tests: High-volume concurrent operations
6. Secondary Index Consistency: RegistryPolicy index integrity under load

All tests are designed to be deterministic and not flaky.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING

import pytest

pytestmark = pytest.mark.unit

from omnibase_infra.enums import EnumPolicyType
from omnibase_infra.errors import ComputeRegistryError, PolicyRegistryError
from omnibase_infra.runtime import handler_registry as registry_module
from omnibase_infra.runtime.handler_registry import (
    HANDLER_TYPE_HTTP,
    RegistryEventBusBinding,
    RegistryProtocolBinding,
    get_event_bus_registry,
    get_handler_registry,
)
from omnibase_infra.runtime.registry_compute import RegistryCompute
from omnibase_infra.runtime.registry_policy import RegistryPolicy

if TYPE_CHECKING:
    from omnibase_infra.runtime.protocol_policy import ProtocolPolicy


# =============================================================================
# Mock Classes for Testing
# =============================================================================


class MockSyncPolicy:
    """Mock synchronous policy for testing."""

    @property
    def policy_id(self) -> str:
        """Return the policy identifier."""
        return "mock-sync"

    @property
    def policy_type(self) -> str:
        """Return the policy type."""
        return "orchestrator"

    def evaluate(self, context: dict[str, object]) -> dict[str, object]:
        return {"result": "sync"}

    def decide(self, context: dict[str, object]) -> dict[str, object]:
        return self.evaluate(context)


class MockSyncPolicyV2:
    """Second mock policy for version testing."""

    @property
    def policy_id(self) -> str:
        """Return the policy identifier."""
        return "mock-sync-v2"

    @property
    def policy_type(self) -> str:
        """Return the policy type."""
        return "orchestrator"

    def evaluate(self, context: dict[str, object]) -> dict[str, object]:
        return {"result": "v2"}


class MockHandler:
    """Generic mock handler for testing."""

    def execute(self) -> None:
        """Mock execute method for ProtocolHandler compliance."""


class MockEventBus:
    """Generic mock event bus for testing."""

    async def publish_envelope(
        self, envelope: object, topic: str, *, key: bytes | None = None
    ) -> None:
        """Mock publish_envelope method for ProtocolEventBus compliance."""


class MockComputePlugin:
    """Mock synchronous compute plugin for testing."""

    def execute(
        self, input_data: dict[str, object], context: dict[str, object]
    ) -> dict[str, object]:
        """Execute synchronous computation."""
        return {"result": "computed"}


class MockComputePluginV2:
    """Second mock compute plugin for version testing."""

    def execute(
        self, input_data: dict[str, object], context: dict[str, object]
    ) -> dict[str, object]:
        """Execute synchronous computation v2."""
        return {"result": "computed_v2"}


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def policy_registry() -> RegistryPolicy:
    """Provide a fresh RegistryPolicy instance."""
    # Reset the semver cache to ensure test isolation
    RegistryPolicy._reset_semver_cache()
    return RegistryPolicy()


@pytest.fixture
def handler_registry() -> RegistryProtocolBinding:
    """Provide a fresh RegistryProtocolBinding instance."""
    return RegistryProtocolBinding()


@pytest.fixture
def event_bus_registry() -> RegistryEventBusBinding:
    """Provide a fresh RegistryEventBusBinding instance."""
    return RegistryEventBusBinding()


@pytest.fixture
def compute_registry() -> RegistryCompute:
    """Provide a fresh RegistryCompute instance."""
    # Reset the semver cache to ensure test isolation
    RegistryCompute._reset_semver_cache()
    return RegistryCompute()


@pytest.fixture(autouse=True)
def reset_singletons() -> Iterator[None]:
    """Reset singleton instances before each test."""
    with registry_module._singleton_lock:
        registry_module._handler_registry = None
        registry_module._event_bus_registry = None
    yield
    with registry_module._singleton_lock:
        registry_module._handler_registry = None
        registry_module._event_bus_registry = None


# =============================================================================
# RegistryPolicy Race Condition Tests
# =============================================================================


class TestPolicyRegistryConcurrentRegistration:
    """Tests for concurrent policy registration scenarios."""

    def test_concurrent_registration_different_policies(
        self, policy_registry: RegistryPolicy
    ) -> None:
        """Test concurrent registration of different policies is thread-safe."""
        num_threads = 50
        errors: list[Exception] = []

        def register_policy(index: int) -> None:
            try:
                policy_registry.register_policy(
                    policy_id=f"policy-{index}",
                    policy_class=MockSyncPolicy,  # type: ignore[arg-type]
                    policy_type=EnumPolicyType.ORCHESTRATOR,
                    version="1.0.0",
                )
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=register_policy, args=(i,))
            for i in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors during registration: {errors}"
        assert len(policy_registry) == num_threads

    def test_concurrent_registration_same_policy_different_versions(
        self, policy_registry: RegistryPolicy
    ) -> None:
        """Test concurrent registration of same policy with different versions."""
        num_versions = 20
        errors: list[Exception] = []

        def register_version(version_num: int) -> None:
            try:
                policy_registry.register_policy(
                    policy_id="versioned-policy",
                    policy_class=MockSyncPolicy,  # type: ignore[arg-type]
                    policy_type=EnumPolicyType.ORCHESTRATOR,
                    version=f"{version_num}.0.0",
                )
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=register_version, args=(i,))
            for i in range(1, num_versions + 1)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors during registration: {errors}"
        # All versions should be registered
        versions = policy_registry.list_versions("versioned-policy")
        assert len(versions) == num_versions

    def test_concurrent_get_during_registration(
        self, policy_registry: RegistryPolicy
    ) -> None:
        """Test concurrent get operations during registration."""
        # Pre-register a policy to read
        policy_registry.register_policy(
            policy_id="existing-policy",
            policy_class=MockSyncPolicy,  # type: ignore[arg-type]
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="1.0.0",
        )

        read_errors: list[Exception] = []
        write_errors: list[Exception] = []
        read_results: list[type[ProtocolPolicy]] = []

        def read_policy() -> None:
            try:
                for _ in range(100):
                    result = policy_registry.get("existing-policy")
                    read_results.append(result)
            except Exception as e:
                read_errors.append(e)

        def write_policy(index: int) -> None:
            try:
                policy_registry.register_policy(
                    policy_id=f"new-policy-{index}",
                    policy_class=MockSyncPolicy,  # type: ignore[arg-type]
                    policy_type=EnumPolicyType.ORCHESTRATOR,
                    version="1.0.0",
                )
            except Exception as e:
                write_errors.append(e)

        readers = [threading.Thread(target=read_policy) for _ in range(5)]
        writers = [threading.Thread(target=write_policy, args=(i,)) for i in range(20)]

        for t in readers + writers:
            t.start()
        for t in readers + writers:
            t.join()

        assert len(read_errors) == 0, f"Read errors: {read_errors}"
        assert len(write_errors) == 0, f"Write errors: {write_errors}"
        # All reads should return the same class
        assert all(r is MockSyncPolicy for r in read_results)


class TestPolicyRegistrySecondaryIndexRaceConditions:
    """Tests for secondary index (_policy_id_index) integrity under concurrent access."""

    def test_secondary_index_consistency_under_concurrent_registration(
        self, policy_registry: RegistryPolicy
    ) -> None:
        """Test that _policy_id_index remains consistent under concurrent registration."""
        num_threads = 30
        errors: list[Exception] = []

        def register_with_versions(policy_index: int) -> None:
            """Register multiple versions for a policy."""
            try:
                for version in range(1, 6):
                    policy_registry.register_policy(
                        policy_id=f"policy-{policy_index}",
                        policy_class=MockSyncPolicy,  # type: ignore[arg-type]
                        policy_type=EnumPolicyType.ORCHESTRATOR,
                        version=f"{version}.0.0",
                    )
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=register_with_versions, args=(i,))
            for i in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors during registration: {errors}"

        # Verify secondary index consistency
        for i in range(num_threads):
            policy_id = f"policy-{i}"
            versions = policy_registry.list_versions(policy_id)
            assert len(versions) == 5, (
                f"Policy {policy_id} has {len(versions)} versions, expected 5"
            )

    def test_secondary_index_consistency_during_unregister(
        self, policy_registry: RegistryPolicy
    ) -> None:
        """Test secondary index remains consistent during concurrent unregister operations."""
        # Pre-register policies
        num_policies = 20
        for i in range(num_policies):
            for v in range(1, 4):
                policy_registry.register_policy(
                    policy_id=f"policy-{i}",
                    policy_class=MockSyncPolicy,  # type: ignore[arg-type]
                    policy_type=EnumPolicyType.ORCHESTRATOR,
                    version=f"{v}.0.0",
                )

        errors: list[Exception] = []
        unregister_counts: list[int] = []

        def unregister_policy(policy_index: int) -> None:
            """Unregister a specific version."""
            try:
                count = policy_registry.unregister(
                    f"policy-{policy_index}", version="1.0.0"
                )
                unregister_counts.append(count)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=unregister_policy, args=(i,))
            for i in range(num_policies)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors during unregister: {errors}"
        # Each unregister should have removed exactly 1 entry
        assert all(c == 1 for c in unregister_counts)

        # Verify remaining versions
        for i in range(num_policies):
            versions = policy_registry.list_versions(f"policy-{i}")
            assert len(versions) == 2, (
                f"Policy policy-{i} should have 2 versions remaining"
            )


class TestPolicyRegistrySemverCacheRaceConditions:
    """Tests for semver cache thread safety."""

    def test_semver_cache_concurrent_initialization(self) -> None:
        """Test semver cache is initialized safely under concurrent access."""
        RegistryPolicy._reset_semver_cache()

        results: list[tuple[int, int, int, str]] = []
        errors: list[Exception] = []

        def parse_version() -> None:
            try:
                for i in range(50):
                    result = RegistryPolicy._parse_semver(f"{i % 10}.{i % 5}.{i % 3}")
                    results.append(result)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=parse_version) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors during parsing: {errors}"
        # All threads should have parsed successfully
        assert len(results) == 500  # 10 threads * 50 parses each

    def test_semver_cache_returns_consistent_results_under_load(self) -> None:
        """Test that semver cache returns consistent results under concurrent load."""
        RegistryPolicy._reset_semver_cache()

        results: dict[str, list[tuple[int, int, int, str]]] = {}
        lock = threading.Lock()
        errors: list[Exception] = []

        def parse_and_collect(version: str) -> None:
            try:
                result = RegistryPolicy._parse_semver(version)
                with lock:
                    if version not in results:
                        results[version] = []
                    results[version].append(result)
            except Exception as e:
                errors.append(e)

        # Parse the same versions from many threads
        versions = ["1.0.0", "2.0.0", "1.10.0", "1.9.0", "10.0.0"]
        threads = [
            threading.Thread(target=parse_and_collect, args=(v,))
            for v in versions * 20  # Each version parsed 20 times concurrently
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors during parsing: {errors}"
        # Each version should have consistent results
        for version, version_results in results.items():
            first = version_results[0]
            assert all(r == first for r in version_results), (
                f"Inconsistent results for {version}: {set(version_results)}"
            )

    def test_semver_cache_reset_during_concurrent_parsing(self) -> None:
        """Test that cache reset during concurrent parsing is thread-safe.

        This test specifically verifies the TOCTOU (time-of-check-time-of-use)
        fix in _get_semver_parser(). Without the fix, a thread could:
        1. Check if cls._semver_cache is not None (True)
        2. Another thread calls _reset_semver_cache() setting it to None
        3. First thread returns cls._semver_cache which is now None
        4. Caller gets TypeError: 'NoneType' object is not callable

        The fix stores the cache reference in a local variable before the check,
        ensuring the returned reference is always valid.
        """
        RegistryPolicy._reset_semver_cache()

        # Initialize the cache first
        RegistryPolicy._parse_semver("1.0.0")

        errors: list[Exception] = []
        parse_count = 0
        reset_count = 0
        lock = threading.Lock()

        def parse_continuously() -> None:
            """Parse versions continuously while resets are happening."""
            nonlocal parse_count
            local_count = 0
            try:
                for i in range(200):
                    # This should NEVER raise TypeError even during resets
                    result = RegistryPolicy._parse_semver(f"{i % 10}.{i % 5}.{i % 3}")
                    assert result is not None
                    local_count += 1
            except Exception as e:
                errors.append(e)
            finally:
                with lock:
                    parse_count += local_count

        def reset_repeatedly() -> None:
            """Reset the cache repeatedly to trigger race conditions."""
            nonlocal reset_count
            local_count = 0
            try:
                for _ in range(50):
                    RegistryPolicy._reset_semver_cache()
                    local_count += 1
                    # Small sleep to allow interleaving
                    time.sleep(0.0001)
            except Exception as e:
                errors.append(e)
            finally:
                with lock:
                    reset_count += local_count

        # Create multiple parsing threads and reset threads
        parsers = [threading.Thread(target=parse_continuously) for _ in range(5)]
        resetters = [threading.Thread(target=reset_repeatedly) for _ in range(2)]

        for t in parsers + resetters:
            t.start()
        for t in parsers + resetters:
            t.join()

        # No errors should occur - especially not TypeError
        assert len(errors) == 0, (
            f"Errors during concurrent reset/parse: {errors}. "
            f"TypeError indicates TOCTOU race condition in _get_semver_parser()"
        )

        # All parses should have completed successfully
        assert parse_count == 1000, f"Expected 1000 parses, got {parse_count}"
        # All resets should have completed
        assert reset_count == 100, f"Expected 100 resets, got {reset_count}"


class TestComputeRegistrySemverCacheResetDuringParsing:
    """Tests for RegistryCompute semver cache reset thread safety."""

    def test_semver_cache_reset_during_concurrent_parsing(
        self, compute_registry: RegistryCompute
    ) -> None:
        """Test that cache reset during concurrent parsing is thread-safe.

        This test specifically verifies the TOCTOU (time-of-check-time-of-use)
        fix in _get_semver_parser(). Without the fix, a thread could return
        None from _get_semver_parser() causing TypeError when calling the parser.
        """
        RegistryCompute._reset_semver_cache()

        # Initialize the cache first
        RegistryCompute._parse_semver("1.0.0")

        errors: list[Exception] = []
        parse_count = 0
        reset_count = 0
        lock = threading.Lock()

        def parse_continuously() -> None:
            """Parse versions continuously while resets are happening."""
            nonlocal parse_count
            local_count = 0
            try:
                for i in range(200):
                    # This should NEVER raise TypeError even during resets
                    result = RegistryCompute._parse_semver(f"{i % 10}.{i % 5}.{i % 3}")
                    assert result is not None
                    local_count += 1
            except Exception as e:
                errors.append(e)
            finally:
                with lock:
                    parse_count += local_count

        def reset_repeatedly() -> None:
            """Reset the cache repeatedly to trigger race conditions."""
            nonlocal reset_count
            local_count = 0
            try:
                for _ in range(50):
                    RegistryCompute._reset_semver_cache()
                    local_count += 1
                    time.sleep(0.0001)
            except Exception as e:
                errors.append(e)
            finally:
                with lock:
                    reset_count += local_count

        parsers = [threading.Thread(target=parse_continuously) for _ in range(5)]
        resetters = [threading.Thread(target=reset_repeatedly) for _ in range(2)]

        for t in parsers + resetters:
            t.start()
        for t in parsers + resetters:
            t.join()

        # No errors should occur
        assert len(errors) == 0, (
            f"Errors during concurrent reset/parse: {errors}. "
            f"TypeError indicates TOCTOU race condition in _get_semver_parser()"
        )
        assert parse_count == 1000
        assert reset_count == 100


class TestSemverCacheClearOnReset:
    """Tests for verifying cache_clear() is called during reset.

    These tests verify that the LRU cache's internal entries are properly
    cleared during reset to ensure prompt memory reclamation.
    """

    def test_policy_registry_cache_clear_on_reset(self) -> None:
        """Test that RegistryPolicy clears LRU cache entries on reset."""
        # Reset to ensure clean state
        RegistryPolicy._reset_semver_cache()

        # Populate the cache with some entries
        for i in range(50):
            RegistryPolicy._parse_semver(f"{i}.0.0")

        # Get cache info before reset using the proper accessor method
        # Note: RegistryPolicy uses a two-level cache structure:
        # - _semver_cache: outer wrapper function (normalizes input)
        # - _semver_cache_inner: inner LRU-cached function (has cache_info())
        # The _get_semver_cache_info() method accesses the inner function's cache_info()
        cache_info_before = RegistryPolicy._get_semver_cache_info()
        assert cache_info_before is not None, "Cache should be initialized"
        assert cache_info_before.currsize > 0, "Cache should have entries"

        # Reset the cache
        RegistryPolicy._reset_semver_cache()

        # Verify cache reference is None
        assert RegistryPolicy._semver_cache is None

        # Create new cache and verify it's empty
        # Trigger cache initialization by parsing a version
        RegistryPolicy._get_semver_parser()
        cache_info_after = RegistryPolicy._get_semver_cache_info()
        assert cache_info_after is not None, "Cache should be re-initialized"
        assert cache_info_after.currsize == 0, "New cache should be empty after reset"

    def test_compute_registry_cache_clear_on_reset(self) -> None:
        """Test that RegistryCompute clears LRU cache entries on reset."""
        # Reset to ensure clean state
        RegistryCompute._reset_semver_cache()

        # Populate the cache with some entries
        for i in range(50):
            RegistryCompute._parse_semver(f"{i}.0.0")

        # Get cache info before reset
        parser = RegistryCompute._get_semver_parser()
        cache_info_before = parser.cache_info()
        assert cache_info_before.currsize > 0, "Cache should have entries"

        # Reset the cache
        RegistryCompute._reset_semver_cache()

        # Verify cache reference is None
        assert RegistryCompute._semver_cache is None

        # Create new cache and verify it's empty
        new_parser = RegistryCompute._get_semver_parser()
        cache_info_after = new_parser.cache_info()
        assert cache_info_after.currsize == 0, "New cache should be empty after reset"

    def test_concurrent_cache_clear_no_errors(self) -> None:
        """Test that concurrent cache_clear() calls during reset don't cause errors.

        This test verifies that calling cache_clear() on an LRU cache while other
        threads are using it doesn't cause data corruption or exceptions.
        """
        RegistryCompute._reset_semver_cache()

        # Populate initial cache
        for i in range(100):
            RegistryCompute._parse_semver(f"{i % 20}.{i % 10}.{i % 5}")

        errors: list[Exception] = []
        parse_count = 0
        reset_count = 0
        lock = threading.Lock()

        def heavy_parse() -> None:
            """Perform many parse operations to stress the cache."""
            nonlocal parse_count
            local_count = 0
            try:
                for i in range(500):
                    # Parse versions that may or may not be cached
                    RegistryCompute._parse_semver(f"{i % 30}.{i % 15}.{i % 10}")
                    local_count += 1
            except Exception as e:
                errors.append(e)
            finally:
                with lock:
                    parse_count += local_count

        def rapid_reset() -> None:
            """Rapidly reset the cache to stress cache_clear()."""
            nonlocal reset_count
            local_count = 0
            try:
                for _ in range(25):
                    RegistryCompute._reset_semver_cache()
                    local_count += 1
                    # Very short sleep to maximize interleaving
                    time.sleep(0.0001)
            except Exception as e:
                errors.append(e)
            finally:
                with lock:
                    reset_count += local_count

        # Create threads - more parsers than resetters
        parsers = [threading.Thread(target=heavy_parse) for _ in range(8)]
        resetters = [threading.Thread(target=rapid_reset) for _ in range(3)]

        # Start all threads
        for t in parsers + resetters:
            t.start()
        for t in parsers + resetters:
            t.join()

        # Verify no errors occurred
        assert len(errors) == 0, (
            f"Errors during concurrent cache_clear stress test: {errors}"
        )

        # Verify operations completed
        assert parse_count == 4000, f"Expected 4000 parses, got {parse_count}"
        assert reset_count == 75, f"Expected 75 resets, got {reset_count}"

        # Final cache should be in valid state
        final_parser = RegistryCompute._get_semver_parser()
        assert final_parser is not None
        # Should be able to parse after all the stress
        result = RegistryCompute._parse_semver("1.2.3")
        assert result == (1, 2, 3, chr(127))  # chr(127) is the release sentinel


class TestPolicyRegistryStressTest:
    """Stress tests for RegistryPolicy under high concurrent load."""

    def test_high_volume_concurrent_operations(
        self, policy_registry: RegistryPolicy
    ) -> None:
        """Stress test with high volume of concurrent operations."""
        num_operations = 1000
        errors: list[Exception] = []
        results: list[str] = []
        lock = threading.Lock()

        def mixed_operations(thread_id: int) -> None:
            """Perform mixed read/write operations."""
            try:
                for i in range(100):
                    op_type = (thread_id + i) % 4
                    if op_type == 0:
                        # Register
                        policy_registry.register_policy(
                            policy_id=f"stress-{thread_id}-{i}",
                            policy_class=MockSyncPolicy,  # type: ignore[arg-type]
                            policy_type=EnumPolicyType.ORCHESTRATOR,
                            version="1.0.0",
                        )
                        with lock:
                            results.append("register")
                    elif op_type == 1:
                        # List
                        _ = policy_registry.list_keys()
                        with lock:
                            results.append("list")
                    elif op_type == 2:
                        # Check registered
                        _ = policy_registry.is_registered(f"stress-{thread_id}-{i}")
                        with lock:
                            results.append("check")
                    else:
                        # List versions
                        _ = policy_registry.list_versions(f"stress-{thread_id}-{i}")
                        with lock:
                            results.append("versions")
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=mixed_operations, args=(i,)) for i in range(10)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors during stress test: {errors}"
        # All operations should have completed
        assert len(results) == num_operations


# =============================================================================
# RegistryProtocolBinding Race Condition Tests
# =============================================================================


class TestHandlerRegistryConcurrentOperations:
    """Tests for concurrent operations on RegistryProtocolBinding."""

    def test_concurrent_registration_multiple_handlers(
        self, handler_registry: RegistryProtocolBinding
    ) -> None:
        """Test concurrent registration of multiple handlers is thread-safe."""
        errors: list[Exception] = []

        def register_handler(protocol: str, cls: type) -> None:
            try:
                handler_registry.register(protocol, cls)
            except Exception as e:
                errors.append(e)

        # Use unique protocol types to avoid overwrite conflicts
        def handle(self) -> None:
            pass

        handlers_with_unique_keys = [
            (f"custom-{i}", type(f"MockHandler{i}", (), {"handle": handle}))
            for i in range(50)
        ]

        threads = [
            threading.Thread(target=register_handler, args=(proto, cls))
            for proto, cls in handlers_with_unique_keys
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors during registration: {errors}"
        assert len(handler_registry) == 50

    def test_concurrent_read_write_handler_registry(
        self, handler_registry: RegistryProtocolBinding
    ) -> None:
        """Test concurrent reads and writes don't cause data corruption."""
        handler_registry.register(HANDLER_TYPE_HTTP, MockHandler)  # type: ignore[arg-type]

        errors: list[Exception] = []
        read_count = 0
        write_count = 0
        lock = threading.Lock()

        def read_operations() -> None:
            nonlocal read_count
            try:
                for _ in range(200):
                    _ = handler_registry.get(HANDLER_TYPE_HTTP)
                    _ = handler_registry.is_registered(HANDLER_TYPE_HTTP)
                    _ = handler_registry.list_protocols()
                    with lock:
                        read_count += 1
            except Exception as e:
                errors.append(e)

        def write_operations(thread_id: int) -> None:
            nonlocal write_count
            try:
                for i in range(50):
                    handler_registry.register(f"custom-{thread_id}-{i}", MockHandler)  # type: ignore[arg-type]
                    with lock:
                        write_count += 1
            except Exception as e:
                errors.append(e)

        readers = [threading.Thread(target=read_operations) for _ in range(5)]
        writers = [
            threading.Thread(target=write_operations, args=(i,)) for i in range(3)
        ]

        for t in readers + writers:
            t.start()
        for t in readers + writers:
            t.join()

        assert len(errors) == 0, f"Errors: {errors}"
        assert read_count == 1000  # 5 threads * 200 iterations
        assert write_count == 150  # 3 threads * 50 writes

    def test_concurrent_unregister_operations(
        self, handler_registry: RegistryProtocolBinding
    ) -> None:
        """Test concurrent unregister operations are thread-safe."""
        # Pre-register handlers
        for i in range(100):
            handler_registry.register(f"handler-{i}", MockHandler)  # type: ignore[arg-type]

        errors: list[Exception] = []
        unregister_results: list[bool] = []
        lock = threading.Lock()

        def unregister_handler(index: int) -> None:
            try:
                result = handler_registry.unregister(f"handler-{index}")
                with lock:
                    unregister_results.append(result)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=unregister_handler, args=(i,)) for i in range(100)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors: {errors}"
        # All unregisters should succeed (return True)
        assert all(unregister_results)
        assert len(handler_registry) == 0


# =============================================================================
# RegistryEventBusBinding Race Condition Tests
# =============================================================================


class TestEventBusRegistryConcurrentOperations:
    """Tests for concurrent operations on RegistryEventBusBinding."""

    def test_concurrent_registration_unique_kinds(
        self, event_bus_registry: RegistryEventBusBinding
    ) -> None:
        """Test concurrent registration of unique bus kinds is thread-safe."""
        num_buses = 50
        errors: list[Exception] = []

        def register_bus(index: int) -> None:
            try:
                event_bus_registry.register(f"bus-{index}", MockEventBus)  # type: ignore[arg-type]
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=register_bus, args=(i,)) for i in range(num_buses)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors: {errors}"
        assert len(event_bus_registry.list_bus_kinds()) == num_buses

    def test_concurrent_duplicate_registration_races(
        self, event_bus_registry: RegistryEventBusBinding
    ) -> None:
        """Test that concurrent duplicate registrations are properly handled."""
        num_threads = 10
        errors: list[Exception] = []
        success_count = 0
        lock = threading.Lock()

        def try_register() -> None:
            nonlocal success_count
            try:
                event_bus_registry.register("shared-bus", MockEventBus)  # type: ignore[arg-type]
                with lock:
                    success_count += 1
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=try_register) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Exactly one registration should succeed
        assert success_count == 1, f"Expected 1 success, got {success_count}"
        # Remaining should have raised EventBusRegistryError
        assert len(errors) == num_threads - 1

    def test_concurrent_is_registered_during_registration(
        self, event_bus_registry: RegistryEventBusBinding
    ) -> None:
        """Test is_registered during concurrent registration."""
        check_results: list[bool] = []
        errors: list[Exception] = []
        lock = threading.Lock()

        def check_registered() -> None:
            try:
                for _ in range(100):
                    result = event_bus_registry.is_registered("test-bus")
                    with lock:
                        check_results.append(result)
            except Exception as e:
                errors.append(e)

        def register_bus() -> None:
            try:
                # Small delay to allow some checks to run first
                time.sleep(0.001)
                event_bus_registry.register("test-bus", MockEventBus)  # type: ignore[arg-type]
            except Exception as e:
                errors.append(e)

        checkers = [threading.Thread(target=check_registered) for _ in range(5)]
        register_thread = threading.Thread(target=register_bus)

        for t in checkers:
            t.start()
        register_thread.start()

        for t in checkers:
            t.join()
        register_thread.join()

        assert len(errors) == 0, f"Errors: {errors}"
        # Results should be a mix of True and False, with later checks being True
        # The exact distribution depends on timing


# =============================================================================
# Singleton Factory Race Condition Tests
# =============================================================================


class TestSingletonFactoryRaceConditions:
    """Tests for singleton factory function thread safety."""

    def test_get_handler_registry_concurrent_initialization(self) -> None:
        """Test get_handler_registry is thread-safe during lazy initialization."""
        registries: list[RegistryProtocolBinding] = []
        errors: list[Exception] = []
        lock = threading.Lock()

        def get_registry() -> None:
            try:
                registry = get_handler_registry()
                with lock:
                    registries.append(registry)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=get_registry) for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors: {errors}"
        # All registries should be the same instance
        assert len(registries) == 50
        first = registries[0]
        assert all(r is first for r in registries)

    def test_get_event_bus_registry_concurrent_initialization(self) -> None:
        """Test get_event_bus_registry is thread-safe during lazy initialization."""
        registries: list[RegistryEventBusBinding] = []
        errors: list[Exception] = []
        lock = threading.Lock()

        def get_registry() -> None:
            try:
                registry = get_event_bus_registry()
                with lock:
                    registries.append(registry)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=get_registry) for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors: {errors}"
        # All registries should be the same instance
        assert len(registries) == 50
        first = registries[0]
        assert all(r is first for r in registries)

    def test_both_singletons_concurrent_initialization(self) -> None:
        """Test both singletons can be initialized concurrently without issues."""
        handler_registries: list[RegistryProtocolBinding] = []
        event_bus_registries: list[RegistryEventBusBinding] = []
        errors: list[Exception] = []
        lock = threading.Lock()

        def get_handler() -> None:
            try:
                registry = get_handler_registry()
                with lock:
                    handler_registries.append(registry)
            except Exception as e:
                errors.append(e)

        def get_event_bus() -> None:
            try:
                registry = get_event_bus_registry()
                with lock:
                    event_bus_registries.append(registry)
            except Exception as e:
                errors.append(e)

        handler_threads = [threading.Thread(target=get_handler) for _ in range(25)]
        event_bus_threads = [threading.Thread(target=get_event_bus) for _ in range(25)]

        for t in handler_threads + event_bus_threads:
            t.start()
        for t in handler_threads + event_bus_threads:
            t.join()

        assert len(errors) == 0, f"Errors: {errors}"
        assert len(handler_registries) == 25
        assert len(event_bus_registries) == 25
        # All handler registries should be same instance
        assert all(r is handler_registries[0] for r in handler_registries)
        # All event bus registries should be same instance
        assert all(r is event_bus_registries[0] for r in event_bus_registries)


# =============================================================================
# High-Concurrency Stress Tests with ThreadPoolExecutor
# =============================================================================


class TestThreadPoolExecutorStress:
    """Stress tests using ThreadPoolExecutor for controlled concurrency."""

    def test_policy_registry_high_concurrency_executor(
        self, policy_registry: RegistryPolicy
    ) -> None:
        """Stress test RegistryPolicy with ThreadPoolExecutor."""
        num_workers = 20
        num_operations = 200
        errors: list[Exception] = []

        def operation(index: int) -> str:
            try:
                policy_registry.register_policy(
                    policy_id=f"executor-policy-{index}",
                    policy_class=MockSyncPolicy,  # type: ignore[arg-type]
                    policy_type=EnumPolicyType.ORCHESTRATOR,
                    version="1.0.0",
                )
                return f"success-{index}"
            except Exception as e:
                errors.append(e)
                return f"error-{index}"

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(operation, i) for i in range(num_operations)]
            results = [f.result() for f in as_completed(futures)]

        assert len(errors) == 0, f"Errors: {errors}"
        assert len(results) == num_operations
        assert len(policy_registry) == num_operations

    def test_handler_registry_high_concurrency_executor(
        self, handler_registry: RegistryProtocolBinding
    ) -> None:
        """Stress test RegistryProtocolBinding with ThreadPoolExecutor."""
        num_workers = 20
        num_operations = 200
        errors: list[Exception] = []

        def operation(index: int) -> str:
            try:
                handler_registry.register(f"executor-handler-{index}", MockHandler)  # type: ignore[arg-type]
                return f"success-{index}"
            except Exception as e:
                errors.append(e)
                return f"error-{index}"

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(operation, i) for i in range(num_operations)]
            results = [f.result() for f in as_completed(futures)]

        assert len(errors) == 0, f"Errors: {errors}"
        assert len(results) == num_operations
        assert len(handler_registry) == num_operations

    def test_mixed_registry_operations_executor(
        self,
        policy_registry: RegistryPolicy,
        handler_registry: RegistryProtocolBinding,
    ) -> None:
        """Test mixed operations across multiple registries with ThreadPoolExecutor."""
        num_workers = 15
        errors: list[Exception] = []
        results: list[str] = []
        lock = threading.Lock()

        def policy_operation(index: int) -> None:
            try:
                policy_registry.register_policy(
                    policy_id=f"mixed-policy-{index}",
                    policy_class=MockSyncPolicy,  # type: ignore[arg-type]
                    policy_type=EnumPolicyType.ORCHESTRATOR,
                    version="1.0.0",
                )
                with lock:
                    results.append(f"policy-{index}")
            except Exception as e:
                errors.append(e)

        def handler_operation(index: int) -> None:
            try:
                handler_registry.register(f"mixed-handler-{index}", MockHandler)  # type: ignore[arg-type]
                with lock:
                    results.append(f"handler-{index}")
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for i in range(50):
                futures.append(executor.submit(policy_operation, i))
                futures.append(executor.submit(handler_operation, i))

            for f in as_completed(futures):
                try:
                    f.result()
                except Exception as e:
                    errors.append(e)

        assert len(errors) == 0, f"Errors: {errors}"
        assert len(results) == 100
        assert len(policy_registry) == 50
        assert len(handler_registry) == 50


# =============================================================================
# Version Selection Race Conditions
# =============================================================================


class TestVersionSelectionRaceConditions:
    """Tests for race conditions during version selection (get latest)."""

    def test_get_latest_during_version_registration(
        self, policy_registry: RegistryPolicy
    ) -> None:
        """Test get() returns valid version during concurrent version registration."""
        policy_id = "versioned-race"

        # Pre-register version 1.0.0
        policy_registry.register_policy(
            policy_id=policy_id,
            policy_class=MockSyncPolicy,  # type: ignore[arg-type]
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="1.0.0",
        )

        errors: list[Exception] = []
        get_results: list[type[ProtocolPolicy]] = []
        lock = threading.Lock()

        def register_versions() -> None:
            try:
                for i in range(2, 21):
                    policy_registry.register_policy(
                        policy_id=policy_id,
                        policy_class=MockSyncPolicy,  # type: ignore[arg-type]
                        policy_type=EnumPolicyType.ORCHESTRATOR,
                        version=f"{i}.0.0",
                    )
            except Exception as e:
                errors.append(e)

        def get_policy() -> None:
            try:
                for _ in range(100):
                    result = policy_registry.get(policy_id)
                    with lock:
                        get_results.append(result)
            except Exception as e:
                errors.append(e)

        register_thread = threading.Thread(target=register_versions)
        get_threads = [threading.Thread(target=get_policy) for _ in range(5)]

        register_thread.start()
        for t in get_threads:
            t.start()

        register_thread.join()
        for t in get_threads:
            t.join()

        assert len(errors) == 0, f"Errors: {errors}"
        # All gets should have returned a valid policy class
        assert all(r is MockSyncPolicy for r in get_results)


# =============================================================================
# Clear Operation Race Conditions
# =============================================================================


class TestClearOperationRaceConditions:
    """Tests for race conditions during clear operations."""

    def test_clear_during_concurrent_operations(
        self, policy_registry: RegistryPolicy
    ) -> None:
        """Test clear() during concurrent read/write operations."""
        # Pre-register some policies
        for i in range(10):
            policy_registry.register_policy(
                policy_id=f"clear-test-{i}",
                policy_class=MockSyncPolicy,  # type: ignore[arg-type]
                policy_type=EnumPolicyType.ORCHESTRATOR,
                version="1.0.0",
            )

        errors: list[Exception] = []

        def read_operations() -> None:
            try:
                for _ in range(100):
                    try:
                        # May fail after clear
                        _ = policy_registry.get("clear-test-0")
                    except PolicyRegistryError:
                        pass  # Expected after clear
                    _ = policy_registry.list_keys()
            except Exception as e:
                errors.append(e)

        def write_operations() -> None:
            try:
                for i in range(50):
                    policy_registry.register_policy(
                        policy_id=f"new-policy-{i}",
                        policy_class=MockSyncPolicy,  # type: ignore[arg-type]
                        policy_type=EnumPolicyType.ORCHESTRATOR,
                        version="1.0.0",
                    )
            except Exception as e:
                errors.append(e)

        def clear_operation() -> None:
            try:
                time.sleep(0.01)  # Let some operations start
                policy_registry.clear()
            except Exception as e:
                errors.append(e)

        readers = [threading.Thread(target=read_operations) for _ in range(3)]
        writers = [threading.Thread(target=write_operations) for _ in range(2)]
        clear_thread = threading.Thread(target=clear_operation)

        for t in readers + writers + [clear_thread]:
            t.start()
        for t in readers + writers + [clear_thread]:
            t.join()

        # Should not have any unexpected errors (only PolicyRegistryError is ok)
        unexpected_errors = [
            e for e in errors if not isinstance(e, PolicyRegistryError)
        ]
        assert len(unexpected_errors) == 0, f"Unexpected errors: {unexpected_errors}"

    def test_handler_registry_clear_during_concurrent_operations(
        self, handler_registry: RegistryProtocolBinding
    ) -> None:
        """Test handler registry clear() during concurrent operations."""
        # Pre-register handlers
        for i in range(20):
            handler_registry.register(f"handler-{i}", MockHandler)  # type: ignore[arg-type]

        errors: list[Exception] = []

        def read_and_write() -> None:
            try:
                for i in range(50):
                    handler_registry.register(
                        f"new-handler-{threading.current_thread().name}-{i}",
                        MockHandler,
                    )  # type: ignore[arg-type]
                    _ = handler_registry.list_protocols()
            except Exception as e:
                errors.append(e)

        def clear_operation() -> None:
            try:
                time.sleep(0.005)
                handler_registry.clear()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=read_and_write) for _ in range(5)]
        clear_thread = threading.Thread(target=clear_operation)

        for t in threads + [clear_thread]:
            t.start()
        for t in threads + [clear_thread]:
            t.join()

        # No errors should occur
        assert len(errors) == 0, f"Errors: {errors}"


# =============================================================================
# RegistryCompute Concurrent Write-Read Tests
# =============================================================================


class TestComputeRegistryConcurrentWriteRead:
    """Tests for concurrent write-read scenarios in RegistryCompute.

    These tests verify thread safety when writers are registering plugins
    while readers are simultaneously accessing the registry.
    """

    def test_concurrent_writer_registers_while_readers_call_get(
        self, compute_registry: RegistryCompute
    ) -> None:
        """Test writer registering plugins while readers call get().

        This verifies that get() operations are not corrupted by concurrent
        register() operations and return consistent results.
        """
        # Pre-register a plugin that readers will access
        compute_registry.register_plugin(
            plugin_id="existing-plugin",
            plugin_class=MockComputePlugin,  # type: ignore[arg-type]
            version="1.0.0",
        )

        read_errors: list[Exception] = []
        write_errors: list[Exception] = []
        read_results: list[type] = []
        lock = threading.Lock()

        def reader_task() -> None:
            """Continuously read the existing plugin."""
            try:
                for _ in range(100):
                    result = compute_registry.get("existing-plugin")
                    with lock:
                        read_results.append(result)
            except Exception as e:
                read_errors.append(e)

        def writer_task(thread_id: int) -> None:
            """Register new plugins concurrently."""
            try:
                for i in range(20):
                    compute_registry.register_plugin(
                        plugin_id=f"new-plugin-{thread_id}-{i}",
                        plugin_class=MockComputePlugin,  # type: ignore[arg-type]
                        version="1.0.0",
                    )
            except Exception as e:
                write_errors.append(e)

        # Create threads
        readers = [threading.Thread(target=reader_task) for _ in range(5)]
        writers = [threading.Thread(target=writer_task, args=(i,)) for i in range(3)]

        # Start all threads
        for t in readers + writers:
            t.start()
        for t in readers + writers:
            t.join()

        # Verify no errors occurred
        assert len(read_errors) == 0, f"Read errors: {read_errors}"
        assert len(write_errors) == 0, f"Write errors: {write_errors}"

        # All reads should return the correct class
        assert len(read_results) == 500  # 5 readers * 100 iterations
        assert all(r is MockComputePlugin for r in read_results)

        # Verify all writes completed
        assert len(compute_registry) == 1 + (3 * 20)  # existing + 3 writers * 20 each

    def test_concurrent_multiple_writers_different_plugins(
        self, compute_registry: RegistryCompute
    ) -> None:
        """Test multiple writers registering different plugins simultaneously.

        This verifies that concurrent registration of different plugins
        does not cause data corruption or lost registrations.
        """
        num_writers = 10
        plugins_per_writer = 20
        errors: list[Exception] = []

        def writer_task(thread_id: int) -> None:
            """Register plugins for this thread."""
            try:
                for i in range(plugins_per_writer):
                    compute_registry.register_plugin(
                        plugin_id=f"plugin-{thread_id}-{i}",
                        plugin_class=MockComputePlugin,  # type: ignore[arg-type]
                        version="1.0.0",
                    )
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer_task, args=(i,)) for i in range(num_writers)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors during registration: {errors}"

        # Verify all plugins were registered
        expected_count = num_writers * plugins_per_writer
        assert len(compute_registry) == expected_count, (
            f"Expected {expected_count} plugins, got {len(compute_registry)}"
        )

        # Verify each plugin can be retrieved
        for thread_id in range(num_writers):
            for i in range(plugins_per_writer):
                assert compute_registry.is_registered(f"plugin-{thread_id}-{i}")

    def test_concurrent_readers_list_keys_while_writers_register(
        self, compute_registry: RegistryCompute
    ) -> None:
        """Test readers calling list_keys() while writers register.

        This verifies that list_keys() returns consistent results
        (a valid snapshot) while registration is occurring.
        """
        errors: list[Exception] = []
        list_keys_results: list[list[tuple[str, str]]] = []
        lock = threading.Lock()

        def reader_task() -> None:
            """Continuously call list_keys()."""
            try:
                for _ in range(50):
                    keys = compute_registry.list_keys()
                    with lock:
                        list_keys_results.append(keys)
            except Exception as e:
                errors.append(e)

        def writer_task(thread_id: int) -> None:
            """Register new plugins."""
            try:
                for i in range(30):
                    compute_registry.register_plugin(
                        plugin_id=f"writer-{thread_id}-plugin-{i}",
                        plugin_class=MockComputePlugin,  # type: ignore[arg-type]
                        version="1.0.0",
                    )
            except Exception as e:
                errors.append(e)

        readers = [threading.Thread(target=reader_task) for _ in range(5)]
        writers = [threading.Thread(target=writer_task, args=(i,)) for i in range(3)]

        for t in readers + writers:
            t.start()
        for t in readers + writers:
            t.join()

        assert len(errors) == 0, f"Errors: {errors}"

        # Verify all list_keys() calls completed
        assert len(list_keys_results) == 250  # 5 readers * 50 iterations

        # Each result should be a valid list (may have varying lengths due to timing)
        for keys in list_keys_results:
            assert isinstance(keys, list)
            # Keys should be sorted (plugin_id, version) tuples
            for key in keys:
                assert isinstance(key, tuple)
                assert len(key) == 2

        # Final state should have all registered plugins
        final_count = len(compute_registry)
        assert final_count == 90  # 3 writers * 30 plugins each


class TestComputeRegistryConcurrentVersioning:
    """Tests for concurrent versioning scenarios in RegistryCompute."""

    def test_concurrent_readers_list_versions_while_writer_adds_versions(
        self, compute_registry: RegistryCompute
    ) -> None:
        """Test readers calling list_versions() while writer adds versions.

        This verifies that list_versions() returns consistent results
        while new versions are being registered.
        """
        plugin_id = "versioned-plugin"

        # Pre-register initial version
        compute_registry.register_plugin(
            plugin_id=plugin_id,
            plugin_class=MockComputePlugin,  # type: ignore[arg-type]
            version="1.0.0",
        )

        errors: list[Exception] = []
        version_results: list[list[str]] = []
        lock = threading.Lock()

        def reader_task() -> None:
            """Continuously call list_versions()."""
            try:
                for _ in range(100):
                    versions = compute_registry.list_versions(plugin_id)
                    with lock:
                        version_results.append(versions)
            except Exception as e:
                errors.append(e)

        def writer_task() -> None:
            """Add new versions."""
            try:
                for i in range(2, 21):
                    compute_registry.register_plugin(
                        plugin_id=plugin_id,
                        plugin_class=MockComputePluginV2,  # type: ignore[arg-type]
                        version=f"{i}.0.0",
                    )
            except Exception as e:
                errors.append(e)

        readers = [threading.Thread(target=reader_task) for _ in range(5)]
        writer = threading.Thread(target=writer_task)

        for t in readers + [writer]:
            t.start()
        for t in readers + [writer]:
            t.join()

        assert len(errors) == 0, f"Errors: {errors}"

        # All reads should have returned valid version lists
        assert len(version_results) == 500  # 5 readers * 100 iterations

        for versions in version_results:
            assert isinstance(versions, list)
            # Should always include at least the initial version
            assert len(versions) >= 1
            assert "1.0.0" in versions

        # Final state should have all 20 versions
        final_versions = compute_registry.list_versions(plugin_id)
        assert len(final_versions) == 20

    def test_concurrent_get_latest_during_version_registration(
        self, compute_registry: RegistryCompute
    ) -> None:
        """Test get() (latest version) during concurrent version registration.

        This verifies that get() always returns a valid plugin class
        while new versions are being added.
        """
        plugin_id = "version-race-plugin"

        # Pre-register version 1.0.0
        compute_registry.register_plugin(
            plugin_id=plugin_id,
            plugin_class=MockComputePlugin,  # type: ignore[arg-type]
            version="1.0.0",
        )

        errors: list[Exception] = []
        get_results: list[type] = []
        lock = threading.Lock()

        def reader_task() -> None:
            """Continuously call get() to get latest version."""
            try:
                for _ in range(100):
                    result = compute_registry.get(plugin_id)
                    with lock:
                        get_results.append(result)
            except Exception as e:
                errors.append(e)

        def writer_task() -> None:
            """Add new versions with increasing version numbers."""
            try:
                for i in range(2, 21):
                    compute_registry.register_plugin(
                        plugin_id=plugin_id,
                        plugin_class=MockComputePluginV2,  # type: ignore[arg-type]
                        version=f"{i}.0.0",
                    )
            except Exception as e:
                errors.append(e)

        readers = [threading.Thread(target=reader_task) for _ in range(5)]
        writer = threading.Thread(target=writer_task)

        for t in readers + [writer]:
            t.start()
        for t in readers + [writer]:
            t.join()

        assert len(errors) == 0, f"Errors: {errors}"

        # All gets should have returned a valid plugin class
        assert len(get_results) == 500  # 5 readers * 100 iterations
        assert all(r in (MockComputePlugin, MockComputePluginV2) for r in get_results)


class TestComputeRegistryConcurrentUnregister:
    """Tests for concurrent unregister scenarios in RegistryCompute."""

    def test_concurrent_unregister_while_readers_reading(
        self, compute_registry: RegistryCompute
    ) -> None:
        """Test unregister while readers are reading.

        This verifies that unregister() doesn't cause crashes or
        inconsistent state when readers are concurrently accessing.
        """
        # Pre-register plugins
        num_plugins = 50
        for i in range(num_plugins):
            compute_registry.register_plugin(
                plugin_id=f"plugin-{i}",
                plugin_class=MockComputePlugin,  # type: ignore[arg-type]
                version="1.0.0",
            )

        errors: list[Exception] = []
        read_count = 0
        unregister_count = 0
        lock = threading.Lock()

        def reader_task() -> None:
            """Continuously check registration and list keys."""
            nonlocal read_count
            try:
                for _ in range(100):
                    # These operations should not crash even if plugins are unregistered
                    _ = compute_registry.list_keys()
                    for i in range(num_plugins):
                        _ = compute_registry.is_registered(f"plugin-{i}")
                    with lock:
                        read_count += 1
            except Exception as e:
                errors.append(e)

        def unregister_task() -> None:
            """Unregister plugins one by one."""
            nonlocal unregister_count
            try:
                for i in range(num_plugins):
                    count = compute_registry.unregister(f"plugin-{i}")
                    with lock:
                        unregister_count += count
            except Exception as e:
                errors.append(e)

        readers = [threading.Thread(target=reader_task) for _ in range(3)]
        unregisterer = threading.Thread(target=unregister_task)

        for t in readers + [unregisterer]:
            t.start()
        for t in readers + [unregisterer]:
            t.join()

        # Should have no crashes or unexpected exceptions
        assert len(errors) == 0, f"Errors: {errors}"

        # All reads should have completed
        assert read_count == 300  # 3 readers * 100 iterations

        # All plugins should be unregistered
        assert unregister_count == num_plugins
        assert len(compute_registry) == 0

    def test_concurrent_unregister_specific_version_while_readers_read(
        self, compute_registry: RegistryCompute
    ) -> None:
        """Test unregistering specific versions while readers access.

        This verifies consistent state when specific versions are removed.
        """
        plugin_id = "multi-version-plugin"

        # Register multiple versions
        for v in range(1, 11):
            compute_registry.register_plugin(
                plugin_id=plugin_id,
                plugin_class=MockComputePlugin,  # type: ignore[arg-type]
                version=f"{v}.0.0",
            )

        errors: list[Exception] = []
        version_counts: list[int] = []
        lock = threading.Lock()

        def reader_task() -> None:
            """Continuously list versions."""
            try:
                for _ in range(100):
                    versions = compute_registry.list_versions(plugin_id)
                    with lock:
                        version_counts.append(len(versions))
            except Exception as e:
                errors.append(e)

        def unregister_task() -> None:
            """Unregister versions one by one."""
            try:
                for v in range(1, 6):  # Unregister first 5 versions
                    compute_registry.unregister(plugin_id, version=f"{v}.0.0")
                    time.sleep(0.001)  # Small delay to increase interleaving
            except Exception as e:
                errors.append(e)

        readers = [threading.Thread(target=reader_task) for _ in range(3)]
        unregisterer = threading.Thread(target=unregister_task)

        for t in readers + [unregisterer]:
            t.start()
        for t in readers + [unregisterer]:
            t.join()

        assert len(errors) == 0, f"Errors: {errors}"

        # Version counts should always be valid (between 5 and 10)
        for count in version_counts:
            assert 5 <= count <= 10, f"Invalid version count: {count}"

        # Final state should have 5 versions remaining
        final_versions = compute_registry.list_versions(plugin_id)
        assert len(final_versions) == 5


class TestComputeRegistryHighContention:
    """High contention stress tests for RegistryCompute."""

    def test_high_contention_10_writers_10_readers_timed(
        self, compute_registry: RegistryCompute
    ) -> None:
        """High contention test: 10 writers + 10 readers for ~1 second.

        This stress test verifies that the registry remains consistent
        under high concurrent load.
        """
        import random

        errors: list[Exception] = []
        write_count = 0
        read_count = 0
        lock = threading.Lock()
        stop_event = threading.Event()

        def writer_task(thread_id: int) -> None:
            """Continuously register plugins until stopped."""
            nonlocal write_count
            local_count = 0
            try:
                while not stop_event.is_set():
                    compute_registry.register_plugin(
                        plugin_id=f"stress-plugin-{thread_id}-{local_count}",
                        plugin_class=MockComputePlugin,  # type: ignore[arg-type]
                        version="1.0.0",
                    )
                    local_count += 1
            except Exception as e:
                errors.append(e)
            finally:
                with lock:
                    write_count += local_count

        def reader_task() -> None:
            """Continuously read registry until stopped."""
            nonlocal read_count
            local_count = 0
            try:
                while not stop_event.is_set():
                    # Mix of different read operations
                    op = local_count % 4
                    if op == 0:
                        _ = compute_registry.list_keys()
                    elif op == 1:
                        _ = len(compute_registry)
                    elif op == 2:
                        # Random check for registered plugin
                        random_id = f"stress-plugin-{random.randint(0, 9)}-{random.randint(0, 100)}"
                        _ = compute_registry.is_registered(random_id)
                    else:
                        # Try to get a plugin that may or may not exist
                        try:
                            random_id = f"stress-plugin-{random.randint(0, 9)}-{random.randint(0, 50)}"
                            _ = compute_registry.get(random_id)
                        except ComputeRegistryError:
                            pass  # Expected - plugin may not exist yet
                    local_count += 1
            except Exception as e:
                if not isinstance(e, ComputeRegistryError):
                    errors.append(e)
            finally:
                with lock:
                    read_count += local_count

        # Create 10 writers and 10 readers
        writers = [threading.Thread(target=writer_task, args=(i,)) for i in range(10)]
        readers = [threading.Thread(target=reader_task) for _ in range(10)]

        # Start all threads
        for t in writers + readers:
            t.start()

        # Run for ~1 second
        time.sleep(1.0)

        # Signal threads to stop
        stop_event.set()

        # Wait for threads to complete
        for t in writers + readers:
            t.join(timeout=5.0)

        # Verify no unexpected errors
        unexpected_errors = [
            e for e in errors if not isinstance(e, ComputeRegistryError)
        ]
        assert len(unexpected_errors) == 0, f"Unexpected errors: {unexpected_errors}"

        # Verify significant operations occurred (at least 100 each)
        assert write_count >= 100, f"Too few writes: {write_count}"
        assert read_count >= 100, f"Too few reads: {read_count}"

        # Verify registry is in consistent state
        final_keys = compute_registry.list_keys()
        assert len(final_keys) == len(compute_registry)

    def test_concurrent_stress_random_interleaving(
        self, compute_registry: RegistryCompute
    ) -> None:
        """Stress test with random operation interleaving.

        This test creates high contention by having threads perform
        random combinations of read, write, and unregister operations.
        """
        import random

        num_threads = 20
        operations_per_thread = 100
        errors: list[Exception] = []
        operation_counts: dict[str, int] = {
            "register": 0,
            "get": 0,
            "list_keys": 0,
            "list_versions": 0,
            "is_registered": 0,
            "unregister": 0,
        }
        lock = threading.Lock()

        # Pre-register some plugins for read operations to access
        for i in range(10):
            compute_registry.register_plugin(
                plugin_id=f"base-plugin-{i}",
                plugin_class=MockComputePlugin,  # type: ignore[arg-type]
                version="1.0.0",
            )

        def random_operations_task(thread_id: int) -> None:
            """Perform random operations."""
            local_counts: dict[str, int] = dict.fromkeys(operation_counts, 0)
            try:
                for i in range(operations_per_thread):
                    op = random.choice(
                        [
                            "register",
                            "get",
                            "list_keys",
                            "list_versions",
                            "is_registered",
                            "unregister",
                        ]
                    )

                    if op == "register":
                        compute_registry.register_plugin(
                            plugin_id=f"random-{thread_id}-{i}",
                            plugin_class=MockComputePlugin,  # type: ignore[arg-type]
                            version="1.0.0",
                        )
                        local_counts["register"] += 1

                    elif op == "get":
                        try:
                            plugin_id = f"base-plugin-{random.randint(0, 9)}"
                            _ = compute_registry.get(plugin_id)
                        except ComputeRegistryError:
                            pass  # May have been unregistered
                        local_counts["get"] += 1

                    elif op == "list_keys":
                        _ = compute_registry.list_keys()
                        local_counts["list_keys"] += 1

                    elif op == "list_versions":
                        plugin_id = f"base-plugin-{random.randint(0, 9)}"
                        _ = compute_registry.list_versions(plugin_id)
                        local_counts["list_versions"] += 1

                    elif op == "is_registered":
                        plugin_id = f"base-plugin-{random.randint(0, 9)}"
                        _ = compute_registry.is_registered(plugin_id)
                        local_counts["is_registered"] += 1

                    elif op == "unregister":
                        # Only unregister random-* plugins, not base plugins
                        plugin_id = f"random-{random.randint(0, num_threads - 1)}-{random.randint(0, operations_per_thread - 1)}"
                        _ = compute_registry.unregister(plugin_id)
                        local_counts["unregister"] += 1

            except Exception as e:
                if not isinstance(e, ComputeRegistryError):
                    errors.append(e)
            finally:
                with lock:
                    for k, v in local_counts.items():
                        operation_counts[k] += v

        threads = [
            threading.Thread(target=random_operations_task, args=(i,))
            for i in range(num_threads)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Verify no unexpected errors
        unexpected_errors = [
            e for e in errors if not isinstance(e, ComputeRegistryError)
        ]
        assert len(unexpected_errors) == 0, f"Unexpected errors: {unexpected_errors}"

        # Verify operations were performed
        total_ops = sum(operation_counts.values())
        assert total_ops == num_threads * operations_per_thread, (
            f"Expected {num_threads * operations_per_thread} operations, got {total_ops}"
        )

        # Registry should be in a consistent state (no corruption)
        final_keys = compute_registry.list_keys()
        assert len(final_keys) == len(compute_registry)

        # Base plugins should still exist (we didn't unregister them)
        for i in range(10):
            assert compute_registry.is_registered(f"base-plugin-{i}")


class TestComputeRegistryContainsOperatorConcurrency:
    """Tests for concurrent __contains__ operator usage."""

    def test_concurrent_contains_during_registration(
        self, compute_registry: RegistryCompute
    ) -> None:
        """Test 'in' operator during concurrent registration."""
        errors: list[Exception] = []
        check_results: list[bool] = []
        lock = threading.Lock()

        # Pre-register a known plugin
        compute_registry.register_plugin(
            plugin_id="known-plugin",
            plugin_class=MockComputePlugin,  # type: ignore[arg-type]
            version="1.0.0",
        )

        def checker_task() -> None:
            """Check if known plugin is registered."""
            try:
                for _ in range(100):
                    result = "known-plugin" in compute_registry
                    with lock:
                        check_results.append(result)
            except Exception as e:
                errors.append(e)

        def writer_task(thread_id: int) -> None:
            """Register new plugins."""
            try:
                for i in range(30):
                    compute_registry.register_plugin(
                        plugin_id=f"new-{thread_id}-{i}",
                        plugin_class=MockComputePlugin,  # type: ignore[arg-type]
                        version="1.0.0",
                    )
            except Exception as e:
                errors.append(e)

        checkers = [threading.Thread(target=checker_task) for _ in range(5)]
        writers = [threading.Thread(target=writer_task, args=(i,)) for i in range(3)]

        for t in checkers + writers:
            t.start()
        for t in checkers + writers:
            t.join()

        assert len(errors) == 0, f"Errors: {errors}"

        # All checks should return True (known-plugin was never unregistered)
        assert len(check_results) == 500
        assert all(check_results), "Some checks returned False unexpectedly"


class TestComputeRegistryClearConcurrency:
    """Tests for concurrent clear() operations."""

    def test_concurrent_clear_during_operations(
        self, compute_registry: RegistryCompute
    ) -> None:
        """Test clear() during concurrent read/write operations."""
        # Pre-register some plugins
        for i in range(20):
            compute_registry.register_plugin(
                plugin_id=f"clear-test-{i}",
                plugin_class=MockComputePlugin,  # type: ignore[arg-type]
                version="1.0.0",
            )

        errors: list[Exception] = []

        def read_operations() -> None:
            """Continuously read the registry."""
            try:
                for _ in range(100):
                    try:
                        _ = compute_registry.get("clear-test-0")
                    except ComputeRegistryError:
                        pass  # Expected after clear
                    _ = compute_registry.list_keys()
                    _ = len(compute_registry)
            except Exception as e:
                errors.append(e)

        def write_operations(thread_id: int) -> None:
            """Continuously write to the registry."""
            try:
                for i in range(50):
                    compute_registry.register_plugin(
                        plugin_id=f"post-clear-{thread_id}-{i}",
                        plugin_class=MockComputePlugin,  # type: ignore[arg-type]
                        version="1.0.0",
                    )
            except Exception as e:
                errors.append(e)

        def clear_operation() -> None:
            """Clear the registry."""
            try:
                time.sleep(0.01)  # Let some operations start
                compute_registry.clear()
            except Exception as e:
                errors.append(e)

        readers = [threading.Thread(target=read_operations) for _ in range(3)]
        writers = [
            threading.Thread(target=write_operations, args=(i,)) for i in range(2)
        ]
        clear_thread = threading.Thread(target=clear_operation)

        for t in readers + writers + [clear_thread]:
            t.start()
        for t in readers + writers + [clear_thread]:
            t.join()

        # Should not have any unexpected errors
        unexpected_errors = [
            e for e in errors if not isinstance(e, ComputeRegistryError)
        ]
        assert len(unexpected_errors) == 0, f"Unexpected errors: {unexpected_errors}"

        # Registry should be in consistent state
        final_keys = compute_registry.list_keys()
        assert len(final_keys) == len(compute_registry)
