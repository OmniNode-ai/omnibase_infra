# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Tests for RegistryPayload.

Verifies:
1. Registry resolves ("ModelClaudeHookEvent", "1.0.0") to correct class
2. Registry raises clear error for unregistered types
3. Thread-safe (frozen after registration)

Related:
    - OMN-2036: ProtocolPayloadRegistry implementation
"""

from __future__ import annotations

import threading
import warnings

import pytest
from pydantic import BaseModel

from omnibase_infra.errors.error_payload_registry import PayloadRegistryError
from omnibase_infra.protocols.protocol_payload_registry import (
    ProtocolPayloadRegistry,
)
from omnibase_infra.runtime.registry.registry_payload import RegistryPayload

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def registry() -> RegistryPayload:
    """Create a fresh RegistryPayload for each test."""
    return RegistryPayload()


# =============================================================================
# Sample Pydantic Models for Testing
# =============================================================================


class ModelClaudeHookEvent(BaseModel):
    """Sample payload model for testing."""

    event_type: str = "hook"
    payload: dict[str, str] = {}


class ModelClaudeHookEventV2(BaseModel):
    """Sample v2 payload model for testing."""

    event_type: str = "hook_v2"
    payload: dict[str, str] = {}
    metadata: dict[str, str] = {}


class ModelNodeIntrospectionEvent(BaseModel):
    """Another sample payload model for testing."""

    node_id: str = ""
    node_type: str = ""


# =============================================================================
# DoD 1: Registry resolves ("ModelClaudeHookEvent", "1.0.0") to correct class
# =============================================================================


class TestRegistryPayloadResolve:
    """Tests for successful resolution of registered types."""

    def test_resolve_returns_correct_class(self, registry: RegistryPayload) -> None:
        """DoD: Registry resolves ("ModelClaudeHookEvent", "1.0.0") to correct class."""
        registry.register("ModelClaudeHookEvent", "1.0.0", ModelClaudeHookEvent)
        registry.freeze()

        result = registry.resolve("ModelClaudeHookEvent", "1.0.0")
        assert result is ModelClaudeHookEvent

    def test_resolve_different_versions(self, registry: RegistryPayload) -> None:
        """Same payload_type with different versions resolves to different classes."""
        registry.register("ModelClaudeHookEvent", "1.0.0", ModelClaudeHookEvent)
        registry.register("ModelClaudeHookEvent", "2.0.0", ModelClaudeHookEventV2)
        registry.freeze()

        assert registry.resolve("ModelClaudeHookEvent", "1.0.0") is ModelClaudeHookEvent
        assert (
            registry.resolve("ModelClaudeHookEvent", "2.0.0") is ModelClaudeHookEventV2
        )

    def test_resolve_different_types(self, registry: RegistryPayload) -> None:
        """Different payload_types resolve to different classes."""
        registry.register("ModelClaudeHookEvent", "1.0.0", ModelClaudeHookEvent)
        registry.register(
            "ModelNodeIntrospectionEvent", "1.0.0", ModelNodeIntrospectionEvent
        )
        registry.freeze()

        assert registry.resolve("ModelClaudeHookEvent", "1.0.0") is ModelClaudeHookEvent
        assert (
            registry.resolve("ModelNodeIntrospectionEvent", "1.0.0")
            is ModelNodeIntrospectionEvent
        )

    def test_resolved_class_can_instantiate(self, registry: RegistryPayload) -> None:
        """Resolved class can be used to create model instances."""
        registry.register("ModelClaudeHookEvent", "1.0.0", ModelClaudeHookEvent)
        registry.freeze()

        cls = registry.resolve("ModelClaudeHookEvent", "1.0.0")
        instance = cls(event_type="test", payload={"key": "value"})
        assert instance.event_type == "test"
        assert instance.payload == {"key": "value"}


# =============================================================================
# DoD 2: Registry raises clear error for unregistered types
# =============================================================================


class TestRegistryPayloadErrors:
    """Tests for clear error reporting on unregistered types."""

    def test_resolve_unregistered_raises_error(self, registry: RegistryPayload) -> None:
        """DoD: Registry raises clear error for unregistered types."""
        registry.freeze()

        with pytest.raises(PayloadRegistryError, match="Unregistered payload type"):
            registry.resolve("NonExistentType", "1.0.0")

    def test_resolve_unregistered_includes_type_in_error(
        self, registry: RegistryPayload
    ) -> None:
        """Error message includes the unregistered type and version."""
        registry.freeze()

        with pytest.raises(PayloadRegistryError, match="'NonExistentType'"):
            registry.resolve("NonExistentType", "1.0.0")

    def test_resolve_unregistered_lists_registered_types(
        self, registry: RegistryPayload
    ) -> None:
        """Error message lists available registered types."""
        registry.register("ModelClaudeHookEvent", "1.0.0", ModelClaudeHookEvent)
        registry.freeze()

        with pytest.raises(PayloadRegistryError, match="ModelClaudeHookEvent"):
            registry.resolve("NonExistentType", "1.0.0")

    def test_resolve_wrong_version_raises_error(
        self, registry: RegistryPayload
    ) -> None:
        """Resolving a registered type with wrong version raises error."""
        registry.register("ModelClaudeHookEvent", "1.0.0", ModelClaudeHookEvent)
        registry.freeze()

        with pytest.raises(PayloadRegistryError, match="Unregistered payload type"):
            registry.resolve("ModelClaudeHookEvent", "2.0.0")

    def test_resolve_empty_registry_shows_none(self, registry: RegistryPayload) -> None:
        """Error on empty registry shows '(none)' for registered types."""
        registry.freeze()

        with pytest.raises(PayloadRegistryError, match=r"\(none\)"):
            registry.resolve("AnyType", "1.0.0")

    def test_resolve_before_freeze_raises_error(
        self, registry: RegistryPayload
    ) -> None:
        """Calling resolve before freeze raises error."""
        registry.register("ModelClaudeHookEvent", "1.0.0", ModelClaudeHookEvent)

        with pytest.raises(PayloadRegistryError, match="not frozen"):
            registry.resolve("ModelClaudeHookEvent", "1.0.0")

    def test_has_before_freeze_raises_error(self, registry: RegistryPayload) -> None:
        """Calling has before freeze raises error."""
        with pytest.raises(PayloadRegistryError, match="not frozen"):
            registry.has("ModelClaudeHookEvent", "1.0.0")

    def test_list_types_before_freeze_raises_error(
        self, registry: RegistryPayload
    ) -> None:
        """Calling list_types before freeze raises error."""
        with pytest.raises(PayloadRegistryError, match="not frozen"):
            registry.list_types()

    def test_register_after_freeze_raises_error(
        self, registry: RegistryPayload
    ) -> None:
        """Registering after freeze raises error."""
        registry.freeze()

        with pytest.raises(PayloadRegistryError, match="frozen"):
            registry.register("NewType", "1.0.0", ModelClaudeHookEvent)

    def test_register_duplicate_raises_error(self, registry: RegistryPayload) -> None:
        """Registering the same (type, version) twice raises error."""
        registry.register("ModelClaudeHookEvent", "1.0.0", ModelClaudeHookEvent)

        with pytest.raises(PayloadRegistryError, match="already registered"):
            registry.register("ModelClaudeHookEvent", "1.0.0", ModelClaudeHookEventV2)

    def test_register_non_basemodel_raises_error(
        self, registry: RegistryPayload
    ) -> None:
        """Registering a non-BaseModel class raises error."""
        with pytest.raises(PayloadRegistryError, match="BaseModel subclass"):
            registry.register("BadType", "1.0.0", dict)  # type: ignore[arg-type]

    def test_register_basemodel_itself_raises_error(
        self, registry: RegistryPayload
    ) -> None:
        """Registering BaseModel itself (not a subclass) raises error."""
        with pytest.raises(PayloadRegistryError, match="concrete BaseModel subclass"):
            registry.register("BadType", "1.0.0", BaseModel)

    def test_register_empty_payload_type_raises_error(
        self, registry: RegistryPayload
    ) -> None:
        """Empty payload_type raises error."""
        with pytest.raises(PayloadRegistryError, match="non-empty string"):
            registry.register("", "1.0.0", ModelClaudeHookEvent)

    def test_register_empty_version_raises_error(
        self, registry: RegistryPayload
    ) -> None:
        """Empty version raises error."""
        with pytest.raises(PayloadRegistryError, match="non-empty string"):
            registry.register("ModelClaudeHookEvent", "", ModelClaudeHookEvent)

    def test_register_whitespace_payload_type_raises_error(
        self, registry: RegistryPayload
    ) -> None:
        """Whitespace-only payload_type raises error."""
        with pytest.raises(PayloadRegistryError, match="non-empty string"):
            registry.register("   ", "1.0.0", ModelClaudeHookEvent)

    def test_register_whitespace_version_raises_error(
        self, registry: RegistryPayload
    ) -> None:
        """Whitespace-only version raises error."""
        with pytest.raises(PayloadRegistryError, match="non-empty string"):
            registry.register("ModelClaudeHookEvent", "   ", ModelClaudeHookEvent)

    def test_error_includes_payload_type_context(
        self, registry: RegistryPayload
    ) -> None:
        """PayloadRegistryError includes payload_type in extra context."""
        registry.freeze()

        with pytest.raises(PayloadRegistryError) as exc_info:
            registry.resolve("TestType", "1.0.0")

        error = exc_info.value
        assert "TestType" in str(error)


# =============================================================================
# DoD 3: Thread-safe (frozen after registration)
# =============================================================================


class TestRegistryPayloadThreadSafety:
    """Tests for thread safety and freeze semantics."""

    def test_freeze_prevents_registration(self, registry: RegistryPayload) -> None:
        """After freeze, no new registrations are accepted."""
        registry.freeze()

        with pytest.raises(PayloadRegistryError, match="frozen"):
            registry.register("NewType", "1.0.0", ModelClaudeHookEvent)

    def test_freeze_is_idempotent(self, registry: RegistryPayload) -> None:
        """Calling freeze multiple times does not raise."""
        registry.freeze()
        registry.freeze()  # Should not raise
        assert registry.is_frozen

    def test_concurrent_reads_after_freeze(self, registry: RegistryPayload) -> None:
        """Concurrent resolve calls after freeze are thread-safe."""
        registry.register("ModelClaudeHookEvent", "1.0.0", ModelClaudeHookEvent)
        registry.freeze()

        num_threads = 20
        errors: list[Exception] = []
        results: list[type[BaseModel]] = []
        lock = threading.Lock()

        def read_registry() -> None:
            try:
                cls = registry.resolve("ModelClaudeHookEvent", "1.0.0")
                with lock:
                    results.append(cls)
            except Exception as e:
                with lock:
                    errors.append(e)

        threads = [threading.Thread(target=read_registry) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors during concurrent reads: {errors}"
        assert len(results) == num_threads
        assert all(r is ModelClaudeHookEvent for r in results)

    def test_concurrent_registration_different_types(
        self, registry: RegistryPayload
    ) -> None:
        """Concurrent registration of different types is thread-safe."""
        num_threads = 20
        errors: list[Exception] = []
        lock = threading.Lock()

        def register_type(index: int) -> None:
            try:

                class DynamicModel(BaseModel):
                    value: int = index

                DynamicModel.__name__ = f"DynamicModel_{index}"
                registry.register(f"Type_{index}", "1.0.0", DynamicModel)
            except Exception as e:
                with lock:
                    errors.append(e)

        threads = [
            threading.Thread(target=register_type, args=(i,))
            for i in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors during concurrent registration: {errors}"
        assert registry.entry_count == num_threads

    def test_concurrent_duplicate_registration_exactly_one_succeeds(
        self, registry: RegistryPayload
    ) -> None:
        """Duplicate registration: exactly one succeeds, rest get error."""
        num_threads = 10
        errors: list[PayloadRegistryError] = []
        successes = 0
        lock = threading.Lock()

        def try_register() -> None:
            nonlocal successes
            try:

                class Payload(BaseModel):
                    value: str = "test"

                registry.register("DuplicateType", "1.0.0", Payload)
                with lock:
                    successes += 1
            except PayloadRegistryError as e:
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

    def test_barrier_synchronized_registration(self, registry: RegistryPayload) -> None:
        """Barrier-synchronized registration for maximum contention."""
        num_threads = 10
        barrier = threading.Barrier(num_threads)
        errors: list[PayloadRegistryError] = []
        successes = 0
        lock = threading.Lock()

        def synchronized_register() -> None:
            nonlocal successes
            try:
                barrier.wait()  # Synchronize all threads

                class Payload(BaseModel):
                    value: str = "barrier"

                registry.register("BarrierType", "1.0.0", Payload)
                with lock:
                    successes += 1
            except PayloadRegistryError as e:
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

        assert successes == 1, f"Expected 1 success, got {successes}"
        assert len(errors) == num_threads - 1


# =============================================================================
# Protocol Compliance
# =============================================================================


class TestProtocolCompliance:
    """Verify RegistryPayload satisfies ProtocolPayloadRegistry."""

    def test_isinstance_check(self) -> None:
        """RegistryPayload satisfies ProtocolPayloadRegistry protocol."""
        registry = RegistryPayload()
        assert isinstance(registry, ProtocolPayloadRegistry)


# =============================================================================
# Additional Behavior Tests
# =============================================================================


class TestRegistryPayloadBehavior:
    """Tests for has(), list_types(), properties, and dunder methods."""

    def test_has_returns_true_for_registered(self, registry: RegistryPayload) -> None:
        """has() returns True for registered types."""
        registry.register("ModelClaudeHookEvent", "1.0.0", ModelClaudeHookEvent)
        registry.freeze()

        assert registry.has("ModelClaudeHookEvent", "1.0.0") is True

    def test_has_returns_false_for_unregistered(
        self, registry: RegistryPayload
    ) -> None:
        """has() returns False for unregistered types."""
        registry.freeze()

        assert registry.has("NonExistent", "1.0.0") is False

    def test_list_types_returns_sorted(self, registry: RegistryPayload) -> None:
        """list_types() returns sorted list of (payload_type, version) tuples."""
        registry.register("Zebra", "1.0.0", ModelClaudeHookEvent)
        registry.register("Alpha", "2.0.0", ModelClaudeHookEventV2)
        registry.register("Alpha", "1.0.0", ModelNodeIntrospectionEvent)
        registry.freeze()

        result = registry.list_types()
        assert result == [
            ("Alpha", "1.0.0"),
            ("Alpha", "2.0.0"),
            ("Zebra", "1.0.0"),
        ]

    def test_entry_count(self, registry: RegistryPayload) -> None:
        """entry_count reflects number of registrations."""
        assert registry.entry_count == 0

        registry.register("A", "1.0.0", ModelClaudeHookEvent)
        assert registry.entry_count == 1

        registry.register("B", "1.0.0", ModelClaudeHookEventV2)
        assert registry.entry_count == 2

    def test_is_frozen_property(self, registry: RegistryPayload) -> None:
        """is_frozen reflects freeze state."""
        assert registry.is_frozen is False
        registry.freeze()
        assert registry.is_frozen is True

    def test_len(self, registry: RegistryPayload) -> None:
        """__len__ returns entry count after freeze."""
        registry.register("A", "1.0.0", ModelClaudeHookEvent)
        registry.freeze()
        assert len(registry) == 1

    def test_len_before_freeze_raises(self, registry: RegistryPayload) -> None:
        """__len__ raises before freeze to enforce freeze-after-init invariant."""
        registry.register("A", "1.0.0", ModelClaudeHookEvent)
        with pytest.raises(PayloadRegistryError, match="not frozen"):
            len(registry)

    def test_contains(self, registry: RegistryPayload) -> None:
        """__contains__ supports 'in' operator after freeze."""
        registry.register("A", "1.0.0", ModelClaudeHookEvent)
        registry.freeze()
        assert ("A", "1.0.0") in registry
        assert ("B", "1.0.0") not in registry

    def test_contains_before_freeze_raises(self, registry: RegistryPayload) -> None:
        """__contains__ raises before freeze to enforce freeze-after-init invariant."""
        registry.register("A", "1.0.0", ModelClaudeHookEvent)
        with pytest.raises(PayloadRegistryError, match="not frozen"):
            registry.__contains__(("A", "1.0.0"))

    def test_str_representation(self, registry: RegistryPayload) -> None:
        """__str__ produces readable output."""
        result = str(registry)
        assert "RegistryPayload" in result
        assert "entries=0" in result
        assert "frozen=False" in result

    def test_repr_representation(self, registry: RegistryPayload) -> None:
        """__repr__ produces detailed output."""
        result = repr(registry)
        assert "RegistryPayload" in result

    def test_clear_resets_state(self, registry: RegistryPayload) -> None:
        """clear() resets registry to empty unfrozen state."""
        registry.register("A", "1.0.0", ModelClaudeHookEvent)
        registry.freeze()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            registry.clear()

        assert registry.entry_count == 0
        assert registry.is_frozen is False

    def test_clear_emits_warning(self, registry: RegistryPayload) -> None:
        """clear() emits UserWarning about test-only usage."""
        with pytest.warns(UserWarning, match="intended for testing only"):
            registry.clear()
