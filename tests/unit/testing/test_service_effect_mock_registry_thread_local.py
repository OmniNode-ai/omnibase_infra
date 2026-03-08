# SPDX-License-Identifier: Apache-2.0
"""Unit tests for thread-local EffectMockRegistry utilities.

Tests cover thread isolation, the scoped context manager, and
cleanup behavior of the thread-local registry helpers.

Related:
    - OMN-1336: Add thread-local utility for EffectMockRegistry
"""

from __future__ import annotations

import threading
from collections.abc import Generator

import pytest

from omnibase_infra.testing.service_effect_mock_registry import (
    EffectMockRegistry,
)
from omnibase_infra.testing.service_effect_mock_registry_thread_local import (
    clear_thread_local_registry,
    get_thread_local_registry,
    scoped_effect_mock_registry,
)


@pytest.fixture(autouse=True)
def _clean_thread_local() -> Generator[None, None, None]:
    """Ensure thread-local state is clean before and after each test."""
    clear_thread_local_registry()
    yield
    clear_thread_local_registry()


@pytest.mark.unit
class TestGetThreadLocalRegistry:
    """Tests for get_thread_local_registry()."""

    def test_returns_registry_instance(self) -> None:
        """Returns a EffectMockRegistry instance."""
        registry = get_thread_local_registry()
        assert isinstance(registry, EffectMockRegistry)

    def test_returns_same_instance_on_repeated_calls(self) -> None:
        """Returns the same instance when called multiple times in same thread."""
        registry1 = get_thread_local_registry()
        registry2 = get_thread_local_registry()
        assert registry1 is registry2

    def test_different_threads_get_different_instances(self) -> None:
        """Different threads receive different registry instances."""
        main_registry = get_thread_local_registry()
        main_registry.register("MainThreadProtocol", object())

        thread_registry: EffectMockRegistry | None = None
        thread_has_main_protocol: bool | None = None

        def worker() -> None:
            nonlocal thread_registry, thread_has_main_protocol
            thread_registry = get_thread_local_registry()
            thread_has_main_protocol = thread_registry.has("MainThreadProtocol")

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()

        assert thread_registry is not None
        assert thread_registry is not main_registry
        assert thread_has_main_protocol is False

    def test_thread_isolation_of_registrations(self) -> None:
        """Registrations in one thread do not affect another thread."""
        main_registry = get_thread_local_registry()

        results: dict[str, bool] = {}

        def thread_a() -> None:
            reg = get_thread_local_registry()
            reg.register("ProtocolA", "mock_a")
            results["a_has_a"] = reg.has("ProtocolA")
            results["a_has_b"] = reg.has("ProtocolB")

        def thread_b() -> None:
            reg = get_thread_local_registry()
            reg.register("ProtocolB", "mock_b")
            results["b_has_a"] = reg.has("ProtocolA")
            results["b_has_b"] = reg.has("ProtocolB")

        t_a = threading.Thread(target=thread_a)
        t_b = threading.Thread(target=thread_b)
        t_a.start()
        t_a.join()
        t_b.start()
        t_b.join()

        # Each thread should only see its own registrations
        assert results["a_has_a"] is True
        assert results["a_has_b"] is False
        assert results["b_has_a"] is False
        assert results["b_has_b"] is True

        # Main thread should not see either
        assert main_registry.has("ProtocolA") is False
        assert main_registry.has("ProtocolB") is False


@pytest.mark.unit
class TestClearThreadLocalRegistry:
    """Tests for clear_thread_local_registry()."""

    def test_clear_removes_registry(self) -> None:
        """After clearing, a fresh registry is created on next access."""
        registry1 = get_thread_local_registry()
        registry1.register("ProtocolA", object())

        clear_thread_local_registry()

        registry2 = get_thread_local_registry()
        assert registry2 is not registry1
        assert len(registry2) == 0

    def test_clear_is_safe_when_no_registry_exists(self) -> None:
        """Clearing when no registry exists does not raise."""
        # Should not raise even without prior get_thread_local_registry()
        clear_thread_local_registry()


@pytest.mark.unit
class TestScopedEffectMockRegistry:
    """Tests for scoped_effect_mock_registry() context manager."""

    def test_yields_registry(self) -> None:
        """Context manager yields a EffectMockRegistry."""
        with scoped_effect_mock_registry() as registry:
            assert isinstance(registry, EffectMockRegistry)

    def test_clears_on_exit(self) -> None:
        """Registrations are cleared when context exits."""
        with scoped_effect_mock_registry() as registry:
            registry.register("ProtocolA", object())
            assert len(registry) == 1

        # After exiting, a new registry should be fresh
        fresh = get_thread_local_registry()
        assert len(fresh) == 0

    def test_clears_on_exception(self) -> None:
        """Registrations are cleared even if an exception occurs."""

        class TestError(Exception):
            pass

        with pytest.raises(TestError):
            with scoped_effect_mock_registry() as registry:
                registry.register("ProtocolA", object())
                raise TestError("test")

        fresh = get_thread_local_registry()
        assert len(fresh) == 0

    def test_nested_scopes(self) -> None:
        """Inner scope cleanup does not affect already-exited outer scope."""
        with scoped_effect_mock_registry() as outer:
            outer.register("OuterProtocol", object())

            with scoped_effect_mock_registry() as inner:
                # Inner is the same thread-local instance
                assert inner is outer
                inner.register("InnerProtocol", object())

            # After inner exits, registry is cleared and recreated
            fresh = get_thread_local_registry()
            assert not fresh.has("OuterProtocol")
            assert not fresh.has("InnerProtocol")

    def test_usable_as_fixture_pattern(self) -> None:
        """Demonstrates the recommended pytest fixture pattern."""
        with scoped_effect_mock_registry() as registry:
            registry.register("ProtocolPostgresAdapter", "stub_postgres")
            registry.register("ProtocolConsulClient", "stub_consul")

            assert registry.resolve("ProtocolPostgresAdapter") == "stub_postgres"
            assert registry.resolve("ProtocolConsulClient") == "stub_consul"
