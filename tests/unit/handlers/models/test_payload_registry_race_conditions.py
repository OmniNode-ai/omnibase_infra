# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Race condition tests for payload registries.

This module tests thread safety for:
- RegistryPayloadHttp
- RegistryPayloadConsul
- RegistryPayloadVault

Tests verify that:
1. Concurrent registrations don't cause race conditions
2. Error raising happens inside lock context (atomic check-and-set)
3. Clear operations are thread-safe
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, ClassVar

import pytest
from pydantic import BaseModel

from omnibase_infra.handlers.models.consul.model_payload_consul import (
    ModelPayloadConsul,
)
from omnibase_infra.handlers.models.consul.registry_payload_consul import (
    RegistryPayloadConsul,
)
from omnibase_infra.handlers.models.http.model_payload_http import ModelPayloadHttp
from omnibase_infra.handlers.models.http.registry_payload_http import (
    RegistryPayloadHttp,
)
from omnibase_infra.handlers.models.vault.model_payload_vault import ModelPayloadVault
from omnibase_infra.handlers.models.vault.registry_payload_vault import (
    RegistryPayloadVault,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def reset_registries():
    """Reset all payload registries before and after each test."""
    # Save original state
    http_types = dict(RegistryPayloadHttp._types)
    consul_types = dict(RegistryPayloadConsul._types)
    vault_types = dict(RegistryPayloadVault._types)

    yield

    # Restore original state
    RegistryPayloadHttp._types = http_types
    RegistryPayloadConsul._types = consul_types
    RegistryPayloadVault._types = vault_types


# =============================================================================
# Mock Payload Classes for Testing
# =============================================================================


def create_http_payload_class(op_type: str) -> type[ModelPayloadHttp]:
    """Create a mock HTTP payload class for testing."""
    # Use a different parameter name to avoid shadowing issues in class scope
    _op_type = op_type

    class MockHttpPayload(ModelPayloadHttp):
        pass

    # Set the class variable after class creation to avoid scoping issues
    MockHttpPayload.__name__ = f"MockHttpPayload_{_op_type}"
    return MockHttpPayload


def create_consul_payload_class(op_type: str) -> type[ModelPayloadConsul]:
    """Create a mock Consul payload class for testing."""
    _op_type = op_type

    class MockConsulPayload(ModelPayloadConsul):
        pass

    MockConsulPayload.__name__ = f"MockConsulPayload_{_op_type}"
    return MockConsulPayload


def create_vault_payload_class(op_type: str) -> type[ModelPayloadVault]:
    """Create a mock Vault payload class for testing."""
    _op_type = op_type

    class MockVaultPayload(ModelPayloadVault):
        pass

    MockVaultPayload.__name__ = f"MockVaultPayload_{_op_type}"
    return MockVaultPayload


# =============================================================================
# HTTP Payload Registry Race Condition Tests
# =============================================================================


class TestRegistryPayloadHttpRaceConditions:
    """Race condition tests for RegistryPayloadHttp."""

    def test_concurrent_registration_different_types(self) -> None:
        """Test concurrent registration of different operation types is thread-safe."""
        num_threads = 20
        errors: list[Exception] = []
        lock = threading.Lock()

        def register_payload(index: int) -> None:
            try:
                op_type = f"test_op_{index}"
                # Use factory function to create class with proper closure
                payload_cls = create_http_payload_class(op_type)
                RegistryPayloadHttp.register(op_type)(payload_cls)
            except Exception as e:
                with lock:
                    errors.append(e)

        threads = [
            threading.Thread(target=register_payload, args=(i,))
            for i in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors during registration: {errors}"
        # Verify all registrations completed
        for i in range(num_threads):
            assert RegistryPayloadHttp.is_registered(f"test_op_{i}")

    def test_concurrent_duplicate_registration_raises_error_atomically(self) -> None:
        """Test that duplicate registration errors are raised atomically.

        This verifies the race condition fix where:
        1. Thread A acquires lock, checks if type exists, releases lock
        2. Thread B registers the same type between lock release and error raise
        3. Thread A raises ValueError (incorrectly if not atomic)

        With the fix, the check and error are atomic within the lock.
        """
        num_threads = 10
        errors: list[ValueError] = []
        successes = 0
        lock = threading.Lock()
        operation_type = "duplicate_test"

        def try_register() -> None:
            nonlocal successes
            try:

                @RegistryPayloadHttp.register(operation_type)
                class Payload(ModelPayloadHttp):
                    operation_type: ClassVar[str] = "duplicate_test"  # type: ignore[misc]

                with lock:
                    successes += 1
            except ValueError as e:
                with lock:
                    errors.append(e)
            except Exception as e:
                pytest.fail(f"Unexpected error: {e}")

        threads = [threading.Thread(target=try_register) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Exactly one should succeed, rest should raise ValueError
        assert successes == 1, f"Expected 1 success, got {successes}"
        assert len(errors) == num_threads - 1, (
            f"Expected {num_threads - 1} errors, got {len(errors)}"
        )

        # All errors should mention the operation type
        for error in errors:
            assert "duplicate_test" in str(error)

    def test_concurrent_clear_during_registration(self) -> None:
        """Test clear() during concurrent registrations is thread-safe."""
        errors: list[Exception] = []
        lock = threading.Lock()

        def register_many() -> None:
            try:
                for i in range(50):
                    try:

                        @RegistryPayloadHttp.register(f"clear_test_{i}")
                        class Payload(ModelPayloadHttp):
                            operation_type: ClassVar[str] = f"clear_test_{i}"  # type: ignore[misc]

                    except ValueError:
                        pass  # Expected if already registered
            except Exception as e:
                with lock:
                    errors.append(e)

        def clear_periodically() -> None:
            try:
                for _ in range(10):
                    time.sleep(0.001)
                    RegistryPayloadHttp.clear()
            except Exception as e:
                with lock:
                    errors.append(e)

        registerers = [threading.Thread(target=register_many) for _ in range(5)]
        clearer = threading.Thread(target=clear_periodically)

        for t in registerers + [clearer]:
            t.start()
        for t in registerers + [clearer]:
            t.join()

        # Should not have any unexpected errors
        assert len(errors) == 0, f"Unexpected errors: {errors}"


# =============================================================================
# Consul Payload Registry Race Condition Tests
# =============================================================================


class TestRegistryPayloadConsulRaceConditions:
    """Race condition tests for RegistryPayloadConsul."""

    def test_concurrent_registration_different_types(self) -> None:
        """Test concurrent registration of different operation types is thread-safe."""
        num_threads = 20
        errors: list[Exception] = []
        lock = threading.Lock()

        def register_payload(index: int) -> None:
            try:
                op_type = f"consul_op_{index}"
                # Use factory function to create class with proper closure
                payload_cls = create_consul_payload_class(op_type)
                RegistryPayloadConsul.register(op_type)(payload_cls)
            except Exception as e:
                with lock:
                    errors.append(e)

        threads = [
            threading.Thread(target=register_payload, args=(i,))
            for i in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors during registration: {errors}"
        for i in range(num_threads):
            assert RegistryPayloadConsul.is_registered(f"consul_op_{i}")

    def test_concurrent_duplicate_registration_raises_error_atomically(self) -> None:
        """Test that duplicate registration errors are raised atomically."""
        num_threads = 10
        errors: list[ValueError] = []
        successes = 0
        lock = threading.Lock()
        operation_type = "consul_duplicate_test"

        def try_register() -> None:
            nonlocal successes
            try:

                @RegistryPayloadConsul.register(operation_type)
                class Payload(ModelPayloadConsul):
                    operation_type: ClassVar[str] = "consul_duplicate_test"  # type: ignore[misc]

                with lock:
                    successes += 1
            except ValueError as e:
                with lock:
                    errors.append(e)
            except Exception as e:
                pytest.fail(f"Unexpected error: {e}")

        threads = [threading.Thread(target=try_register) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert successes == 1, f"Expected 1 success, got {successes}"
        assert len(errors) == num_threads - 1


# =============================================================================
# Vault Payload Registry Race Condition Tests
# =============================================================================


class TestRegistryPayloadVaultRaceConditions:
    """Race condition tests for RegistryPayloadVault."""

    def test_concurrent_registration_different_types(self) -> None:
        """Test concurrent registration of different operation types is thread-safe."""
        num_threads = 20
        errors: list[Exception] = []
        lock = threading.Lock()

        def register_payload(index: int) -> None:
            try:
                op_type = f"vault_op_{index}"
                # Use factory function to create class with proper closure
                payload_cls = create_vault_payload_class(op_type)
                RegistryPayloadVault.register(op_type)(payload_cls)
            except Exception as e:
                with lock:
                    errors.append(e)

        threads = [
            threading.Thread(target=register_payload, args=(i,))
            for i in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors during registration: {errors}"
        for i in range(num_threads):
            assert RegistryPayloadVault.is_registered(f"vault_op_{i}")

    def test_concurrent_duplicate_registration_raises_error_atomically(self) -> None:
        """Test that duplicate registration errors are raised atomically."""
        num_threads = 10
        errors: list[ValueError] = []
        successes = 0
        lock = threading.Lock()
        operation_type = "vault_duplicate_test"

        def try_register() -> None:
            nonlocal successes
            try:

                @RegistryPayloadVault.register(operation_type)
                class Payload(ModelPayloadVault):
                    operation_type: ClassVar[str] = "vault_duplicate_test"  # type: ignore[misc]

                with lock:
                    successes += 1
            except ValueError as e:
                with lock:
                    errors.append(e)
            except Exception as e:
                pytest.fail(f"Unexpected error: {e}")

        threads = [threading.Thread(target=try_register) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert successes == 1, f"Expected 1 success, got {successes}"
        assert len(errors) == num_threads - 1


# =============================================================================
# Stress Tests
# =============================================================================


class TestPayloadRegistryStressTests:
    """High-volume stress tests for payload registries."""

    def test_high_concurrency_mixed_operations(self) -> None:
        """Stress test with high concurrency across all registries."""
        num_threads = 30
        operations_per_thread = 20
        errors: list[Exception] = []
        lock = threading.Lock()

        def mixed_operations(thread_id: int) -> None:
            try:
                for i in range(operations_per_thread):
                    op_type = f"stress_{thread_id}_{i}"

                    # Register to HTTP registry
                    try:

                        @RegistryPayloadHttp.register(op_type)
                        class HttpPayload(ModelPayloadHttp):
                            operation_type: ClassVar[str] = op_type  # type: ignore[misc]

                    except ValueError:
                        pass

                    # Check registration
                    _ = RegistryPayloadHttp.is_registered(op_type)

                    # Get all types
                    _ = RegistryPayloadHttp.get_all_types()

            except Exception as e:
                with lock:
                    errors.append(e)

        threads = [
            threading.Thread(target=mixed_operations, args=(i,))
            for i in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors during stress test: {errors}"

    def test_barrier_synchronized_registration(self) -> None:
        """Test registration with barrier synchronization for maximum contention."""
        num_threads = 10
        barrier = threading.Barrier(num_threads)
        errors: list[Exception] = []
        successes = 0
        lock = threading.Lock()
        operation_type = "barrier_test"

        def synchronized_register() -> None:
            nonlocal successes
            try:
                barrier.wait()  # Synchronize all threads

                @RegistryPayloadHttp.register(operation_type)
                class Payload(ModelPayloadHttp):
                    operation_type: ClassVar[str] = "barrier_test"  # type: ignore[misc]

                with lock:
                    successes += 1
            except ValueError as e:
                with lock:
                    errors.append(e)
            except Exception as e:
                pytest.fail(f"Unexpected error: {e}")

        threads = [
            threading.Thread(target=synchronized_register) for _ in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Exactly one should succeed
        assert successes == 1, f"Expected 1 success, got {successes}"
        assert len(errors) == num_threads - 1

        # Verify registration
        assert RegistryPayloadHttp.is_registered(operation_type)
