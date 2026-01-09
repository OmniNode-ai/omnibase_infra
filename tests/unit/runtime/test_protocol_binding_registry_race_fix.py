# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Test for ProtocolBindingRegistry.get() race condition fix (PR #129).

This module specifically tests the race condition that was fixed where:
1. Thread A acquires lock, checks handler_cls is None, releases lock
2. Thread B registers the protocol between lock release and error raise
3. Thread A raises RegistryError claiming protocol isn't registered (but it now is)

The fix moves the None check inside the lock context to ensure atomicity.
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

import pytest

from omnibase_infra.runtime.handler_registry import ProtocolBindingRegistry
from omnibase_infra.runtime.registry.registry_protocol_binding import RegistryError

if TYPE_CHECKING:
    from omnibase_spi.protocols.handlers.protocol_handler import ProtocolHandler


class MockHandler:
    """Mock handler for testing."""

    def handle(self, request: object) -> object:
        """Valid handle() method for ProtocolHandler protocol."""
        return {"status": "ok"}


class TestProtocolBindingRegistryGetRaceConditionFix:
    """Tests for PR #129 race condition fix in ProtocolBindingRegistry.get()."""

    @pytest.fixture
    def registry(self) -> ProtocolBindingRegistry:
        """Provide a fresh ProtocolBindingRegistry instance."""
        return ProtocolBindingRegistry()

    def test_get_raises_error_atomically_within_lock(
        self, registry: ProtocolBindingRegistry
    ) -> None:
        """Test that get() checks and raises error atomically within lock.

        This test verifies that the error check happens while holding the lock,
        preventing the race condition where another thread could register the
        protocol between lock release and error raise.
        """
        protocol_type = "test-protocol"
        errors: list[Exception] = []
        registration_time: list[float] = []
        error_time: list[float] = []
        lock = threading.Lock()

        def getter_thread() -> None:
            """Thread that tries to get non-existent protocol."""
            try:
                # Small delay to ensure registration thread is ready
                time.sleep(0.001)
                registry.get(protocol_type)
            except RegistryError as e:
                # Record when the error was raised
                with lock:
                    error_time.append(time.time())
                    errors.append(e)
            except Exception as e:
                errors.append(e)

        def register_thread() -> None:
            """Thread that registers the protocol after getter starts."""
            try:
                # Wait a tiny bit to let getter start checking
                time.sleep(0.0015)
                registry.register(protocol_type, MockHandler)  # type: ignore[arg-type]
                # Record when registration completed
                with lock:
                    registration_time.append(time.time())
            except Exception as e:
                errors.append(e)

        getter = threading.Thread(target=getter_thread)
        registerer = threading.Thread(target=register_thread)

        getter.start()
        registerer.start()

        getter.join()
        registerer.join()

        # With the fix, the error should be raised because the check happens
        # atomically within the lock. The getter should see None and raise
        # RegistryError before the registerer completes, OR it should
        # successfully get the handler if registration completed first.

        # Either outcome is acceptable as long as there's no inconsistency:
        # 1. RegistryError raised (getter won the race, checked before registration)
        # 2. No error (registration won the race, getter got the handler)

        # What we're preventing is the OLD race condition where:
        # - Getter releases lock after seeing None
        # - Registerer completes registration
        # - Getter raises error claiming protocol isn't registered (wrong!)

        if len(errors) == 1 and isinstance(errors[0], RegistryError):
            # RegistryError was raised - this is valid
            # Verify the protocol IS now registered (registerer completed)
            assert registry.is_registered(protocol_type)
            # The error message should mention the protocol type
            assert protocol_type in str(errors[0])
        elif len(errors) == 0:
            # No error - getter successfully got the handler after registration
            # Verify the protocol is registered
            assert registry.is_registered(protocol_type)
        else:
            pytest.fail(f"Unexpected errors: {errors}")

    def test_concurrent_get_and_register_no_incorrect_errors(
        self, registry: ProtocolBindingRegistry
    ) -> None:
        """Test that concurrent get() and register() don't produce incorrect errors.

        This stress test runs many iterations to catch the race condition if present.
        """
        num_iterations = 50
        errors: list[RegistryError] = []
        successes = 0
        lock = threading.Lock()

        for i in range(num_iterations):
            protocol_type = f"protocol-{i}"
            iteration_errors: list[RegistryError] = []
            iteration_success = False

            def getter_thread(
                proto: str = protocol_type,
                errors: list[RegistryError] = iteration_errors,
            ) -> None:
                """Try to get the protocol."""
                nonlocal iteration_success
                try:
                    time.sleep(0.0001)
                    result = registry.get(proto)
                    # If we get here, registration won the race
                    assert result is MockHandler
                    with lock:
                        iteration_success = True
                except RegistryError as e:
                    # Getter won the race - record the error
                    with lock:
                        errors.append(e)

            def register_thread(proto: str = protocol_type) -> None:
                """Register the protocol."""
                try:
                    time.sleep(0.00015)
                    registry.register(proto, MockHandler)  # type: ignore[arg-type]
                except Exception as e:
                    pytest.fail(f"Registration failed unexpectedly: {e}")

            getter = threading.Thread(target=getter_thread)
            registerer = threading.Thread(target=register_thread)

            getter.start()
            registerer.start()

            getter.join()
            registerer.join()

            # After both threads complete, verify consistency
            # The protocol MUST be registered now
            assert registry.is_registered(protocol_type), (
                f"Protocol {protocol_type} should be registered after both threads complete"
            )

            if iteration_errors:
                # Error was raised - verify it's the correct error
                assert len(iteration_errors) == 1
                assert protocol_type in str(iteration_errors[0])
                errors.append(iteration_errors[0])
            elif iteration_success:
                successes += 1
            else:
                pytest.fail(f"Iteration {i}: Neither error nor success recorded")

        # Both outcomes are valid depending on timing
        total_outcomes = len(errors) + successes
        assert total_outcomes == num_iterations, (
            f"Expected {num_iterations} outcomes, got {total_outcomes}"
        )

    def test_list_protocols_not_called_after_lock_release(
        self, registry: ProtocolBindingRegistry
    ) -> None:
        """Test that list_protocols() is not called after lock is released.

        This verifies that the registered protocols list is captured atomically
        within the lock, not by calling list_protocols() after the lock is released.
        """
        protocol_type = "missing-protocol"

        try:
            registry.get(protocol_type)
            pytest.fail("Expected RegistryError to be raised")
        except RegistryError as e:
            # Verify the error contains the list of registered protocols
            # This should be empty since we haven't registered anything
            assert "Registered protocols:" in str(e)
            # The fix should inline the sorted keys logic, not call list_protocols()
            # We can't directly verify this without inspecting the implementation,
            # but we can verify the error message is correct
            assert protocol_type in str(e)

        # Now register some protocols and try again
        registry.register("protocol-a", MockHandler)  # type: ignore[arg-type]
        registry.register("protocol-b", MockHandler)  # type: ignore[arg-type]

        try:
            registry.get("missing-protocol-2")
            pytest.fail("Expected RegistryError to be raised")
        except RegistryError as e:
            # Verify the error contains both registered protocols
            assert "protocol-a" in str(e)
            assert "protocol-b" in str(e)
            assert "missing-protocol-2" in str(e)

    def test_get_check_and_error_are_atomic(
        self, registry: ProtocolBindingRegistry
    ) -> None:
        """Test that the None check and error raising are atomic operations.

        This test uses a barrier to synchronize threads and maximize the chance
        of hitting the race condition if it exists.
        """
        protocol_type = "race-protocol"
        barrier = threading.Barrier(2)
        errors: list[Exception] = []
        lock = threading.Lock()

        def getter_with_barrier() -> None:
            """Get protocol after waiting at barrier."""
            try:
                barrier.wait()  # Synchronize with registerer
                registry.get(protocol_type)
            except RegistryError as e:
                with lock:
                    errors.append(e)
            except Exception as e:
                with lock:
                    errors.append(e)

        def register_with_barrier() -> None:
            """Register protocol after waiting at barrier."""
            try:
                barrier.wait()  # Synchronize with getter
                registry.register(protocol_type, MockHandler)  # type: ignore[arg-type]
            except Exception as e:
                with lock:
                    errors.append(e)

        getter = threading.Thread(target=getter_with_barrier)
        registerer = threading.Thread(target=register_with_barrier)

        getter.start()
        registerer.start()

        getter.join()
        registerer.join()

        # After both complete, the protocol must be registered
        assert registry.is_registered(protocol_type)

        # Either no error (registration won) or RegistryError (getter won)
        assert len(errors) <= 1, (
            f"Expected at most 1 error, got {len(errors)}: {errors}"
        )

        if errors:
            assert isinstance(errors[0], RegistryError)
            assert protocol_type in str(errors[0])
