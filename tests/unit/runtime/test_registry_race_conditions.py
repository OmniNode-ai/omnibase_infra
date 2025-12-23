# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Race condition and concurrent access tests for registry components.

This module provides comprehensive race condition tests for:
- PolicyRegistry: Thread-safe policy registration and versioning
- ProtocolBindingRegistry: Thread-safe handler registration
- EventBusBindingRegistry: Thread-safe event bus registration
- Singleton factory functions: Thread-safe lazy initialization

Test Categories:
1. Concurrent Registration: Multiple threads registering simultaneously
2. Concurrent Read/Write: Readers and writers operating together
3. State Consistency: Verifying shared state under concurrent modifications
4. Boundary Conditions: Testing at threshold boundaries (e.g., circuit breaker)
5. Stress Tests: High-volume concurrent operations
6. Secondary Index Consistency: PolicyRegistry index integrity under load

All tests are designed to be deterministic and not flaky.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING

import pytest

from omnibase_infra.enums import EnumPolicyType
from omnibase_infra.errors import PolicyRegistryError
from omnibase_infra.runtime import handler_registry as registry_module
from omnibase_infra.runtime.handler_registry import (
    EVENT_BUS_INMEMORY,
    EVENT_BUS_KAFKA,
    HANDLER_TYPE_CONSUL,
    HANDLER_TYPE_DATABASE,
    HANDLER_TYPE_GRPC,
    HANDLER_TYPE_HTTP,
    HANDLER_TYPE_KAFKA,
    HANDLER_TYPE_VALKEY,
    HANDLER_TYPE_VAULT,
    EventBusBindingRegistry,
    ProtocolBindingRegistry,
    RegistryError,
    get_event_bus_registry,
    get_handler_registry,
)
from omnibase_infra.runtime.policy_registry import PolicyRegistry

if TYPE_CHECKING:
    from omnibase_infra.runtime.protocol_policy import ProtocolPolicy


# =============================================================================
# Mock Classes for Testing
# =============================================================================


class MockSyncPolicy:
    """Mock synchronous policy for testing."""

    def evaluate(self, context: dict[str, object]) -> dict[str, object]:
        return {"result": "sync"}

    def decide(self, context: dict[str, object]) -> dict[str, object]:
        return self.evaluate(context)


class MockSyncPolicyV2:
    """Second mock policy for version testing."""

    def evaluate(self, context: dict[str, object]) -> dict[str, object]:
        return {"result": "v2"}


class MockHandler:
    """Generic mock handler for testing."""


class MockEventBus:
    """Generic mock event bus for testing."""


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def policy_registry() -> PolicyRegistry:
    """Provide a fresh PolicyRegistry instance."""
    # Reset the semver cache to ensure test isolation
    PolicyRegistry._reset_semver_cache()
    return PolicyRegistry()


@pytest.fixture
def handler_registry() -> ProtocolBindingRegistry:
    """Provide a fresh ProtocolBindingRegistry instance."""
    return ProtocolBindingRegistry()


@pytest.fixture
def event_bus_registry() -> EventBusBindingRegistry:
    """Provide a fresh EventBusBindingRegistry instance."""
    return EventBusBindingRegistry()


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
# PolicyRegistry Race Condition Tests
# =============================================================================


class TestPolicyRegistryConcurrentRegistration:
    """Tests for concurrent policy registration scenarios."""

    def test_concurrent_registration_different_policies(
        self, policy_registry: PolicyRegistry
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
        self, policy_registry: PolicyRegistry
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
        self, policy_registry: PolicyRegistry
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
        self, policy_registry: PolicyRegistry
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
            assert (
                len(versions) == 5
            ), f"Policy {policy_id} has {len(versions)} versions, expected 5"

    def test_secondary_index_consistency_during_unregister(
        self, policy_registry: PolicyRegistry
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
            assert (
                len(versions) == 2
            ), f"Policy policy-{i} should have 2 versions remaining"


class TestPolicyRegistrySemverCacheRaceConditions:
    """Tests for semver cache thread safety."""

    def test_semver_cache_concurrent_initialization(self) -> None:
        """Test semver cache is initialized safely under concurrent access."""
        PolicyRegistry._reset_semver_cache()

        results: list[tuple[int, int, int, str]] = []
        errors: list[Exception] = []

        def parse_version() -> None:
            try:
                for i in range(50):
                    result = PolicyRegistry._parse_semver(f"{i % 10}.{i % 5}.{i % 3}")
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
        PolicyRegistry._reset_semver_cache()

        results: dict[str, list[tuple[int, int, int, str]]] = {}
        lock = threading.Lock()
        errors: list[Exception] = []

        def parse_and_collect(version: str) -> None:
            try:
                result = PolicyRegistry._parse_semver(version)
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
            assert all(
                r == first for r in version_results
            ), f"Inconsistent results for {version}: {set(version_results)}"


class TestPolicyRegistryStressTest:
    """Stress tests for PolicyRegistry under high concurrent load."""

    def test_high_volume_concurrent_operations(
        self, policy_registry: PolicyRegistry
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
# ProtocolBindingRegistry Race Condition Tests
# =============================================================================


class TestHandlerRegistryConcurrentOperations:
    """Tests for concurrent operations on ProtocolBindingRegistry."""

    def test_concurrent_registration_multiple_handlers(
        self, handler_registry: ProtocolBindingRegistry
    ) -> None:
        """Test concurrent registration of multiple handlers is thread-safe."""
        handlers = [
            (HANDLER_TYPE_HTTP, type(f"MockHttp{i}", (), {})) for i in range(20)
        ]
        errors: list[Exception] = []

        def register_handler(protocol: str, cls: type) -> None:
            try:
                handler_registry.register(protocol, cls)
            except Exception as e:
                errors.append(e)

        # Use unique protocol types to avoid overwrite conflicts
        handlers_with_unique_keys = [
            (f"custom-{i}", type(f"MockHandler{i}", (), {})) for i in range(50)
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
        self, handler_registry: ProtocolBindingRegistry
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
        self, handler_registry: ProtocolBindingRegistry
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
# EventBusBindingRegistry Race Condition Tests
# =============================================================================


class TestEventBusRegistryConcurrentOperations:
    """Tests for concurrent operations on EventBusBindingRegistry."""

    def test_concurrent_registration_unique_kinds(
        self, event_bus_registry: EventBusBindingRegistry
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
        self, event_bus_registry: EventBusBindingRegistry
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
        # Remaining should have raised RuntimeHostError
        assert len(errors) == num_threads - 1

    def test_concurrent_is_registered_during_registration(
        self, event_bus_registry: EventBusBindingRegistry
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
        registries: list[ProtocolBindingRegistry] = []
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
        registries: list[EventBusBindingRegistry] = []
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
        handler_registries: list[ProtocolBindingRegistry] = []
        event_bus_registries: list[EventBusBindingRegistry] = []
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
        self, policy_registry: PolicyRegistry
    ) -> None:
        """Stress test PolicyRegistry with ThreadPoolExecutor."""
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
        self, handler_registry: ProtocolBindingRegistry
    ) -> None:
        """Stress test ProtocolBindingRegistry with ThreadPoolExecutor."""
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
        policy_registry: PolicyRegistry,
        handler_registry: ProtocolBindingRegistry,
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
        self, policy_registry: PolicyRegistry
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
        self, policy_registry: PolicyRegistry
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
        self, handler_registry: ProtocolBindingRegistry
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
