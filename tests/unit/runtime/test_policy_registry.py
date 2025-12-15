# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for policy_registry module.

Tests follow TDD approach:
1. Write tests first (red phase)
2. Implement registry classes (green phase)
3. Refactor if needed (refactor phase)

All tests validate:
- Policy registration and retrieval
- Sync enforcement for policy plugins
- Version management
- Singleton pattern implementation
- Thread safety
- Error handling for missing registrations
- Convenience functions
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import pytest

from omnibase_infra.enums import EnumPolicyType
from omnibase_infra.errors import PolicyRegistryError
from omnibase_infra.runtime import policy_registry as registry_module
from omnibase_infra.runtime.policy_registry import (
    PolicyRegistry,
    get_policy_class,
    get_policy_registry,
    register_policy,
)

# =============================================================================
# Mock Policy Classes for Testing
# =============================================================================


class MockSyncPolicy:
    """Mock synchronous policy for testing."""

    @property
    def policy_id(self) -> str:
        return "mock-sync"

    @property
    def policy_type(self) -> str:
        return "orchestrator"

    def evaluate(self, context: dict[str, object]) -> dict[str, object]:
        return {"result": "sync"}


class MockAsyncPolicy:
    """Mock async policy for testing sync enforcement."""

    @property
    def policy_id(self) -> str:
        return "mock-async"

    @property
    def policy_type(self) -> str:
        return "reducer"

    async def evaluate(self, context: dict[str, object]) -> dict[str, object]:
        return {"result": "async"}


class MockAsyncDecidePolicy:
    """Mock async policy with decide() method."""

    @property
    def policy_id(self) -> str:
        return "mock-async-decide"

    @property
    def policy_type(self) -> str:
        return "orchestrator"

    async def decide(self, context: dict[str, object]) -> dict[str, object]:
        return {"decision": "async"}


class MockAsyncReducePolicy:
    """Mock async policy with reduce() method."""

    @property
    def policy_id(self) -> str:
        return "mock-async-reduce"

    @property
    def policy_type(self) -> str:
        return "reducer"

    async def reduce(self, states: list[dict[str, object]]) -> dict[str, object]:
        return {"reduced": "async"}


class MockSyncReducerPolicy:
    """Mock synchronous reducer policy for testing."""

    @property
    def policy_id(self) -> str:
        return "mock-sync-reducer"

    @property
    def policy_type(self) -> str:
        return "reducer"

    def reduce(self, states: list[dict[str, object]]) -> dict[str, object]:
        return {"reduced": "sync"}


class MockSyncDecidePolicy:
    """Mock synchronous policy with decide() method."""

    @property
    def policy_id(self) -> str:
        return "mock-sync-decide"

    @property
    def policy_type(self) -> str:
        return "orchestrator"

    def decide(self, context: dict[str, object]) -> dict[str, object]:
        return {"decision": "sync"}


class MockPolicyV1:
    """Mock policy version 1 for version testing."""

    @property
    def policy_id(self) -> str:
        return "mock-versioned"

    @property
    def policy_type(self) -> str:
        return "orchestrator"

    def evaluate(self, context: dict[str, object]) -> dict[str, object]:
        return {"version": "1.0.0"}


class MockPolicyV2:
    """Mock policy version 2 for version testing."""

    @property
    def policy_id(self) -> str:
        return "mock-versioned"

    @property
    def policy_type(self) -> str:
        return "orchestrator"

    def evaluate(self, context: dict[str, object]) -> dict[str, object]:
        return {"version": "2.0.0"}


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def policy_registry() -> PolicyRegistry:
    """Provide a fresh PolicyRegistry instance for each test."""
    return PolicyRegistry()


@pytest.fixture
def populated_policy_registry() -> PolicyRegistry:
    """Provide a PolicyRegistry with pre-registered policies."""
    registry = PolicyRegistry()
    registry.register_policy(
        policy_id="sync-orchestrator",
        policy_class=MockSyncPolicy,
        policy_type=EnumPolicyType.ORCHESTRATOR,
        version="1.0.0",
    )
    registry.register_policy(
        policy_id="sync-reducer",
        policy_class=MockSyncReducerPolicy,
        policy_type=EnumPolicyType.REDUCER,
        version="1.0.0",
    )
    return registry


@pytest.fixture(autouse=True)
def reset_singletons() -> None:  # type: ignore[misc]
    """Reset singleton instances before each test.

    This ensures tests are isolated and don't affect each other
    through the singleton state.
    """
    with registry_module._singleton_lock:
        registry_module._policy_registry = None
    yield
    # Also reset after test
    with registry_module._singleton_lock:
        registry_module._policy_registry = None


# =============================================================================
# TestPolicyRegistryBasics
# =============================================================================


class TestPolicyRegistryBasics:
    """Basic tests for PolicyRegistry class."""

    def test_register_and_get_policy(self, policy_registry: PolicyRegistry) -> None:
        """Test basic registration and retrieval."""
        policy_registry.register_policy(
            policy_id="test-policy",
            policy_class=MockSyncPolicy,
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="1.0.0",
        )
        policy_cls = policy_registry.get("test-policy")
        assert policy_cls is MockSyncPolicy

    def test_get_unregistered_raises_error(
        self, policy_registry: PolicyRegistry
    ) -> None:
        """Test that getting an unregistered policy raises PolicyRegistryError."""
        with pytest.raises(PolicyRegistryError) as exc_info:
            policy_registry.get("unknown-policy")
        assert "unknown-policy" in str(exc_info.value)
        assert "No policy registered" in str(exc_info.value)

    def test_register_orchestrator_policy(
        self, policy_registry: PolicyRegistry
    ) -> None:
        """Test registering an orchestrator policy with EnumPolicyType.ORCHESTRATOR."""
        policy_registry.register_policy(
            policy_id="orchestrator-policy",
            policy_class=MockSyncPolicy,
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="1.0.0",
        )
        policy_cls = policy_registry.get(
            "orchestrator-policy", policy_type=EnumPolicyType.ORCHESTRATOR
        )
        assert policy_cls is MockSyncPolicy
        assert policy_registry.is_registered(
            "orchestrator-policy", policy_type=EnumPolicyType.ORCHESTRATOR
        )

    def test_register_reducer_policy(self, policy_registry: PolicyRegistry) -> None:
        """Test registering a reducer policy with EnumPolicyType.REDUCER."""
        policy_registry.register_policy(
            policy_id="reducer-policy",
            policy_class=MockSyncReducerPolicy,
            policy_type=EnumPolicyType.REDUCER,
            version="1.0.0",
        )
        policy_cls = policy_registry.get(
            "reducer-policy", policy_type=EnumPolicyType.REDUCER
        )
        assert policy_cls is MockSyncReducerPolicy
        assert policy_registry.is_registered(
            "reducer-policy", policy_type=EnumPolicyType.REDUCER
        )


# =============================================================================
# TestPolicyRegistrySyncEnforcement - CRITICAL ACCEPTANCE CRITERIA
# =============================================================================


class TestPolicyRegistrySyncEnforcement:
    """Tests for synchronous-by-default policy enforcement.

    This is CRITICAL functionality per OMN-812 acceptance criteria.
    Policy plugins must be synchronous by default. Async policies
    require explicit deterministic_async=True flag.
    """

    def test_sync_policy_registration_succeeds(
        self, policy_registry: PolicyRegistry
    ) -> None:
        """Test that synchronous policy registers without issues."""
        # Should not raise - sync policy with default deterministic_async=False
        policy_registry.register_policy(
            policy_id="sync-policy",
            policy_class=MockSyncPolicy,
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="1.0.0",
        )
        assert policy_registry.is_registered("sync-policy")
        policy_cls = policy_registry.get("sync-policy")
        assert policy_cls is MockSyncPolicy

    def test_async_policy_without_flag_raises(
        self, policy_registry: PolicyRegistry
    ) -> None:
        """Test that async policy without deterministic_async=True raises error."""
        with pytest.raises(PolicyRegistryError) as exc_info:
            policy_registry.register_policy(
                policy_id="async-policy",
                policy_class=MockAsyncPolicy,
                policy_type=EnumPolicyType.REDUCER,
                version="1.0.0",
                # deterministic_async defaults to False
            )
        error_msg = str(exc_info.value)
        assert "async-policy" in error_msg
        assert "async" in error_msg.lower()
        assert "evaluate" in error_msg.lower() or "deterministic" in error_msg.lower()

    def test_async_policy_with_flag_succeeds(
        self, policy_registry: PolicyRegistry
    ) -> None:
        """Test that async policy with deterministic_async=True registers OK."""
        # Should not raise with explicit flag
        policy_registry.register_policy(
            policy_id="async-policy",
            policy_class=MockAsyncPolicy,
            policy_type=EnumPolicyType.REDUCER,
            version="1.0.0",
            deterministic_async=True,
        )
        assert policy_registry.is_registered("async-policy")
        policy_cls = policy_registry.get("async-policy")
        assert policy_cls is MockAsyncPolicy

    def test_async_evaluate_method_detected(
        self, policy_registry: PolicyRegistry
    ) -> None:
        """Test that async evaluate() method is detected and enforced."""
        with pytest.raises(PolicyRegistryError) as exc_info:
            policy_registry.register_policy(
                policy_id="async-evaluate",
                policy_class=MockAsyncPolicy,
                policy_type=EnumPolicyType.ORCHESTRATOR,
                version="1.0.0",
                deterministic_async=False,
            )
        error_msg = str(exc_info.value)
        # Should mention the async evaluate method
        assert "evaluate" in error_msg.lower()

    def test_async_decide_method_detected(
        self, policy_registry: PolicyRegistry
    ) -> None:
        """Test that async decide() method is detected and enforced."""
        with pytest.raises(PolicyRegistryError) as exc_info:
            policy_registry.register_policy(
                policy_id="async-decide",
                policy_class=MockAsyncDecidePolicy,
                policy_type=EnumPolicyType.ORCHESTRATOR,
                version="1.0.0",
                deterministic_async=False,
            )
        error_msg = str(exc_info.value)
        # Should mention the async decide method
        assert "decide" in error_msg.lower()

    def test_async_reduce_method_detected(
        self, policy_registry: PolicyRegistry
    ) -> None:
        """Test that async reduce() method is detected and enforced."""
        with pytest.raises(PolicyRegistryError) as exc_info:
            policy_registry.register_policy(
                policy_id="async-reduce",
                policy_class=MockAsyncReducePolicy,
                policy_type=EnumPolicyType.REDUCER,
                version="1.0.0",
                deterministic_async=False,
            )
        error_msg = str(exc_info.value)
        # Should mention the async reduce method
        assert "reduce" in error_msg.lower()

    def test_sync_decide_method_succeeds(self, policy_registry: PolicyRegistry) -> None:
        """Test that sync decide() method policy registers successfully."""
        policy_registry.register_policy(
            policy_id="sync-decide",
            policy_class=MockSyncDecidePolicy,
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="1.0.0",
        )
        assert policy_registry.is_registered("sync-decide")

    def test_sync_reduce_method_succeeds(self, policy_registry: PolicyRegistry) -> None:
        """Test that sync reduce() method policy registers successfully."""
        policy_registry.register_policy(
            policy_id="sync-reduce",
            policy_class=MockSyncReducerPolicy,
            policy_type=EnumPolicyType.REDUCER,
            version="1.0.0",
        )
        assert policy_registry.is_registered("sync-reduce")


# =============================================================================
# TestPolicyRegistryList
# =============================================================================


class TestPolicyRegistryList:
    """Tests for list() method."""

    def test_list_all_policies(self, populated_policy_registry: PolicyRegistry) -> None:
        """Test that list returns (id, type, version) tuples."""
        policies = populated_policy_registry.list()
        assert len(policies) == 2
        # Each entry should be a tuple of (policy_id, policy_type, version)
        for entry in policies:
            assert isinstance(entry, tuple)
            assert len(entry) == 3
            policy_id, policy_type, version = entry
            assert isinstance(policy_id, str)
            assert isinstance(policy_type, str)
            assert isinstance(version, str)

    def test_list_by_policy_type(
        self, populated_policy_registry: PolicyRegistry
    ) -> None:
        """Test filtering list by policy type."""
        # List only orchestrator policies
        orchestrator_policies = populated_policy_registry.list(
            policy_type=EnumPolicyType.ORCHESTRATOR
        )
        assert len(orchestrator_policies) == 1
        assert orchestrator_policies[0][1] == "orchestrator"

        # List only reducer policies
        reducer_policies = populated_policy_registry.list(
            policy_type=EnumPolicyType.REDUCER
        )
        assert len(reducer_policies) == 1
        assert reducer_policies[0][1] == "reducer"

    def test_list_empty_registry(self, policy_registry: PolicyRegistry) -> None:
        """Test that empty registry returns empty list."""
        policies = policy_registry.list()
        assert policies == []


# =============================================================================
# TestPolicyRegistryVersioning
# =============================================================================


class TestPolicyRegistryVersioning:
    """Tests for version management."""

    def test_register_multiple_versions(self, policy_registry: PolicyRegistry) -> None:
        """Test registering same policy with different versions."""
        policy_registry.register_policy(
            policy_id="versioned-policy",
            policy_class=MockPolicyV1,
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="1.0.0",
        )
        policy_registry.register_policy(
            policy_id="versioned-policy",
            policy_class=MockPolicyV2,
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="2.0.0",
        )
        assert len(policy_registry) == 2

    def test_get_specific_version(self, policy_registry: PolicyRegistry) -> None:
        """Test retrieving a specific version."""
        policy_registry.register_policy(
            policy_id="versioned-policy",
            policy_class=MockPolicyV1,
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="1.0.0",
        )
        policy_registry.register_policy(
            policy_id="versioned-policy",
            policy_class=MockPolicyV2,
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="2.0.0",
        )

        # Get specific version
        v1_cls = policy_registry.get("versioned-policy", version="1.0.0")
        assert v1_cls is MockPolicyV1

        v2_cls = policy_registry.get("versioned-policy", version="2.0.0")
        assert v2_cls is MockPolicyV2

    def test_get_latest_when_no_version_specified(
        self, policy_registry: PolicyRegistry
    ) -> None:
        """Test that get() returns latest version when version=None."""
        policy_registry.register_policy(
            policy_id="versioned-policy",
            policy_class=MockPolicyV1,
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="1.0.0",
        )
        policy_registry.register_policy(
            policy_id="versioned-policy",
            policy_class=MockPolicyV2,
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="2.0.0",
        )

        # Get without version should return latest (lexicographically highest)
        latest_cls = policy_registry.get("versioned-policy")
        assert latest_cls is MockPolicyV2

    def test_list_versions(self, policy_registry: PolicyRegistry) -> None:
        """Test list_versions() method."""
        policy_registry.register_policy(
            policy_id="versioned-policy",
            policy_class=MockPolicyV1,
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="1.0.0",
        )
        policy_registry.register_policy(
            policy_id="versioned-policy",
            policy_class=MockPolicyV2,
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="2.0.0",
        )

        versions = policy_registry.list_versions("versioned-policy")
        assert "1.0.0" in versions
        assert "2.0.0" in versions
        assert len(versions) == 2

    def test_list_versions_empty_for_unknown_policy(
        self, policy_registry: PolicyRegistry
    ) -> None:
        """Test list_versions() returns empty for unknown policy."""
        versions = policy_registry.list_versions("unknown-policy")
        assert versions == []


# =============================================================================
# TestPolicyRegistryIsRegistered
# =============================================================================


class TestPolicyRegistryIsRegistered:
    """Tests for is_registered() method."""

    def test_is_registered_returns_true(
        self, populated_policy_registry: PolicyRegistry
    ) -> None:
        """Test is_registered returns True when policy exists."""
        assert populated_policy_registry.is_registered("sync-orchestrator")

    def test_is_registered_returns_false(self, policy_registry: PolicyRegistry) -> None:
        """Test is_registered returns False when policy doesn't exist."""
        assert not policy_registry.is_registered("nonexistent-policy")

    def test_is_registered_with_type_filter(
        self, populated_policy_registry: PolicyRegistry
    ) -> None:
        """Test is_registered with policy_type filter."""
        # Policy exists with matching type
        assert populated_policy_registry.is_registered(
            "sync-orchestrator", policy_type=EnumPolicyType.ORCHESTRATOR
        )
        # Policy exists but with different type
        assert not populated_policy_registry.is_registered(
            "sync-orchestrator", policy_type=EnumPolicyType.REDUCER
        )

    def test_is_registered_with_version_filter(
        self, policy_registry: PolicyRegistry
    ) -> None:
        """Test is_registered with version filter."""
        policy_registry.register_policy(
            policy_id="versioned-policy",
            policy_class=MockPolicyV1,
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="1.0.0",
        )
        # Matching version
        assert policy_registry.is_registered("versioned-policy", version="1.0.0")
        # Non-matching version
        assert not policy_registry.is_registered("versioned-policy", version="2.0.0")


# =============================================================================
# TestPolicyRegistryUnregister
# =============================================================================


class TestPolicyRegistryUnregister:
    """Tests for unregister() method."""

    def test_unregister_removes_policy(self, policy_registry: PolicyRegistry) -> None:
        """Test basic unregister removes policy."""
        policy_registry.register_policy(
            policy_id="to-remove",
            policy_class=MockSyncPolicy,
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="1.0.0",
        )
        assert policy_registry.is_registered("to-remove")

        policy_registry.unregister("to-remove")
        assert not policy_registry.is_registered("to-remove")

    def test_unregister_returns_count_when_found(
        self, policy_registry: PolicyRegistry
    ) -> None:
        """Test unregister returns count of removed policies."""
        policy_registry.register_policy(
            policy_id="to-remove",
            policy_class=MockSyncPolicy,
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="1.0.0",
        )
        result = policy_registry.unregister("to-remove")
        assert result == 1

    def test_unregister_returns_zero_when_not_found(
        self, policy_registry: PolicyRegistry
    ) -> None:
        """Test unregister returns 0 when policy not found."""
        result = policy_registry.unregister("nonexistent")
        assert result == 0

    def test_unregister_multiple_versions(
        self, policy_registry: PolicyRegistry
    ) -> None:
        """Test unregister removes all versions by default."""
        policy_registry.register_policy(
            policy_id="versioned",
            policy_class=MockPolicyV1,
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="1.0.0",
        )
        policy_registry.register_policy(
            policy_id="versioned",
            policy_class=MockPolicyV2,
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="2.0.0",
        )
        result = policy_registry.unregister("versioned")
        assert result == 2
        assert not policy_registry.is_registered("versioned")

    def test_unregister_specific_version(self, policy_registry: PolicyRegistry) -> None:
        """Test unregister with specific version."""
        policy_registry.register_policy(
            policy_id="versioned",
            policy_class=MockPolicyV1,
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="1.0.0",
        )
        policy_registry.register_policy(
            policy_id="versioned",
            policy_class=MockPolicyV2,
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="2.0.0",
        )
        result = policy_registry.unregister("versioned", version="1.0.0")
        assert result == 1
        # 2.0.0 should still exist
        assert policy_registry.is_registered("versioned", version="2.0.0")
        assert not policy_registry.is_registered("versioned", version="1.0.0")


# =============================================================================
# TestPolicyRegistryClear
# =============================================================================


class TestPolicyRegistryClear:
    """Tests for clear() method."""

    def test_clear_removes_all_policies(
        self, populated_policy_registry: PolicyRegistry
    ) -> None:
        """Test that clear removes all policies."""
        assert len(populated_policy_registry) > 0
        populated_policy_registry.clear()
        assert len(populated_policy_registry) == 0
        assert populated_policy_registry.list() == []


# =============================================================================
# TestPolicyRegistryLen
# =============================================================================


class TestPolicyRegistryLen:
    """Tests for __len__ method."""

    def test_len_returns_count(self, policy_registry: PolicyRegistry) -> None:
        """Test __len__ returns correct count."""
        assert len(policy_registry) == 0

        policy_registry.register_policy(
            policy_id="policy1",
            policy_class=MockSyncPolicy,
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="1.0.0",
        )
        assert len(policy_registry) == 1

        policy_registry.register_policy(
            policy_id="policy2",
            policy_class=MockSyncReducerPolicy,
            policy_type=EnumPolicyType.REDUCER,
            version="1.0.0",
        )
        assert len(policy_registry) == 2


# =============================================================================
# TestPolicyRegistryContains
# =============================================================================


class TestPolicyRegistryContains:
    """Tests for __contains__ method."""

    def test_contains_returns_true(
        self, populated_policy_registry: PolicyRegistry
    ) -> None:
        """Test __contains__ returns True for registered policy."""
        assert "sync-orchestrator" in populated_policy_registry

    def test_contains_returns_false(self, policy_registry: PolicyRegistry) -> None:
        """Test __contains__ returns False for unregistered policy."""
        assert "nonexistent" not in policy_registry


# =============================================================================
# TestPolicyRegistryThreadSafety
# =============================================================================


class TestPolicyRegistryThreadSafety:
    """Tests for thread safety of PolicyRegistry."""

    def test_concurrent_registration(self, policy_registry: PolicyRegistry) -> None:
        """Test that concurrent registrations are thread-safe."""
        policies = [
            ("policy1", MockSyncPolicy, EnumPolicyType.ORCHESTRATOR),
            ("policy2", MockSyncReducerPolicy, EnumPolicyType.REDUCER),
            ("policy3", MockSyncDecidePolicy, EnumPolicyType.ORCHESTRATOR),
        ]
        errors: list[Exception] = []

        def register_policy_thread(
            policy_id: str, policy_class: type, policy_type: EnumPolicyType
        ) -> None:
            try:
                policy_registry.register_policy(
                    policy_id=policy_id,
                    policy_class=policy_class,
                    policy_type=policy_type,
                    version="1.0.0",
                )
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(
                target=register_policy_thread,
                args=(pid, pcls, ptype),
            )
            for pid, pcls, ptype in policies
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(policy_registry) == len(policies)

    def test_concurrent_get(self, populated_policy_registry: PolicyRegistry) -> None:
        """Test that concurrent gets are thread-safe."""
        results: list[type] = []
        errors: list[Exception] = []

        def get_policy_thread() -> None:
            try:
                for _ in range(50):
                    policy_cls = populated_policy_registry.get("sync-orchestrator")
                    results.append(policy_cls)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=get_policy_thread) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 250  # 5 threads * 50 iterations
        assert all(cls is MockSyncPolicy for cls in results)


# =============================================================================
# TestPolicyRegistrySingleton
# =============================================================================


class TestPolicyRegistrySingleton:
    """Tests for singleton pattern."""

    def test_get_policy_registry_returns_singleton(self) -> None:
        """Test that get_policy_registry returns same instance."""
        registry1 = get_policy_registry()
        registry2 = get_policy_registry()
        assert registry1 is registry2

    def test_get_policy_class_uses_singleton(self) -> None:
        """Test that get_policy_class convenience function uses singleton."""
        # Register via singleton
        get_policy_registry().register_policy(
            policy_id="singleton-test",
            policy_class=MockSyncPolicy,
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="1.0.0",
        )
        # Retrieve via convenience function
        policy_cls = get_policy_class("singleton-test")
        assert policy_cls is MockSyncPolicy

    def test_register_policy_uses_singleton(self) -> None:
        """Test that register_policy convenience function uses singleton."""
        register_policy(
            policy_id="register-test",
            policy_class=MockSyncPolicy,
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="1.0.0",
        )
        # Verify via singleton
        assert get_policy_registry().is_registered("register-test")

    def test_singleton_thread_safe_initialization(self) -> None:
        """Test that singleton initialization is thread-safe."""
        registries: list[PolicyRegistry] = []

        def get_registry() -> None:
            registries.append(get_policy_registry())

        threads = [threading.Thread(target=get_registry) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(registries) == 10
        first = registries[0]
        for reg in registries:
            assert reg is first


# =============================================================================
# TestPolicyRegistryError
# =============================================================================


class TestPolicyRegistryError:
    """Tests for PolicyRegistryError exception class."""

    def test_error_includes_policy_id(self) -> None:
        """Test that PolicyRegistryError context includes policy_id."""
        error = PolicyRegistryError(
            "Policy not found",
            policy_id="missing-policy",
        )
        assert "Policy not found" in str(error)
        assert error.model.context.get("policy_id") == "missing-policy"

    def test_error_includes_policy_type(self) -> None:
        """Test that PolicyRegistryError context includes policy_type."""
        error = PolicyRegistryError(
            "Invalid policy type",
            policy_type="invalid_type",
        )
        assert error.model.context.get("policy_type") == "invalid_type"

    def test_error_with_extra_context(self) -> None:
        """Test PolicyRegistryError with extra context kwargs."""
        error = PolicyRegistryError(
            "Async method detected",
            policy_id="async-policy",
            policy_type="orchestrator",
            async_method="evaluate",
        )
        assert error.model.context.get("async_method") == "evaluate"

    def test_error_is_exception(self) -> None:
        """Test PolicyRegistryError is an Exception."""
        error = PolicyRegistryError("Test error")
        assert isinstance(error, Exception)


# =============================================================================
# TestPolicyRegistryPolicyTypeNormalization
# =============================================================================


class TestPolicyRegistryPolicyTypeNormalization:
    """Tests for policy type normalization."""

    def test_register_with_enum_type(self, policy_registry: PolicyRegistry) -> None:
        """Test registering with EnumPolicyType enum value."""
        policy_registry.register_policy(
            policy_id="enum-type",
            policy_class=MockSyncPolicy,
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="1.0.0",
        )
        assert policy_registry.is_registered("enum-type")

    def test_register_with_string_type(self, policy_registry: PolicyRegistry) -> None:
        """Test registering with string policy type."""
        policy_registry.register_policy(
            policy_id="string-type",
            policy_class=MockSyncPolicy,
            policy_type="orchestrator",
            version="1.0.0",
        )
        assert policy_registry.is_registered("string-type")

    def test_invalid_policy_type_raises_error(
        self, policy_registry: PolicyRegistry
    ) -> None:
        """Test that invalid policy type string raises PolicyRegistryError."""
        with pytest.raises(PolicyRegistryError) as exc_info:
            policy_registry.register_policy(
                policy_id="invalid-type",
                policy_class=MockSyncPolicy,
                policy_type="invalid_type",
                version="1.0.0",
            )
        assert "invalid_type" in str(exc_info.value)

    def test_get_with_enum_and_string_equivalent(
        self, policy_registry: PolicyRegistry
    ) -> None:
        """Test that get() works with both enum and string type."""
        policy_registry.register_policy(
            policy_id="type-test",
            policy_class=MockSyncPolicy,
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="1.0.0",
        )
        # Get with enum
        cls1 = policy_registry.get("type-test", policy_type=EnumPolicyType.ORCHESTRATOR)
        # Get with string
        cls2 = policy_registry.get("type-test", policy_type="orchestrator")
        assert cls1 is cls2 is MockSyncPolicy


# =============================================================================
# TestPolicyRegistryListPolicyTypes
# =============================================================================


class TestPolicyRegistryListPolicyTypes:
    """Tests for list_policy_types() method."""

    def test_list_policy_types_empty(self, policy_registry: PolicyRegistry) -> None:
        """Test list_policy_types returns empty for empty registry."""
        types = policy_registry.list_policy_types()
        assert types == []

    def test_list_policy_types_with_policies(
        self, populated_policy_registry: PolicyRegistry
    ) -> None:
        """Test list_policy_types returns registered types."""
        types = populated_policy_registry.list_policy_types()
        assert "orchestrator" in types
        assert "reducer" in types
        assert len(types) == 2

    def test_list_policy_types_unique(self, policy_registry: PolicyRegistry) -> None:
        """Test list_policy_types returns unique types only."""
        policy_registry.register_policy(
            policy_id="policy1",
            policy_class=MockSyncPolicy,
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="1.0.0",
        )
        policy_registry.register_policy(
            policy_id="policy2",
            policy_class=MockSyncDecidePolicy,
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="1.0.0",
        )
        types = policy_registry.list_policy_types()
        assert types == ["orchestrator"]


# =============================================================================
# TestPolicyRegistryOverwrite
# =============================================================================


class TestPolicyRegistryOverwrite:
    """Tests for policy overwrite behavior."""

    def test_register_same_key_overwrites(
        self, policy_registry: PolicyRegistry
    ) -> None:
        """Test that registering same (id, type, version) overwrites."""
        policy_registry.register_policy(
            policy_id="overwrite-test",
            policy_class=MockPolicyV1,
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="1.0.0",
        )
        policy_registry.register_policy(
            policy_id="overwrite-test",
            policy_class=MockPolicyV2,
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="1.0.0",
        )
        # Should return the new class
        policy_cls = policy_registry.get("overwrite-test")
        assert policy_cls is MockPolicyV2
        # Count should remain 1
        assert len(policy_registry) == 1


# =============================================================================
# Integration Tests
# =============================================================================


# =============================================================================
# TestPolicyRegistryEdgeCases
# =============================================================================


class TestPolicyRegistryEdgeCases:
    """Edge case tests for PolicyRegistry."""

    def test_get_not_found_with_policy_type_filter(
        self, policy_registry: PolicyRegistry
    ) -> None:
        """Test error message includes policy_type when specified in get()."""
        policy_registry.register_policy(
            policy_id="typed-policy",
            policy_class=MockSyncPolicy,
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="1.0.0",
        )
        with pytest.raises(PolicyRegistryError) as exc_info:
            policy_registry.get("typed-policy", policy_type=EnumPolicyType.REDUCER)
        error_msg = str(exc_info.value)
        assert "typed-policy" in error_msg

    def test_get_not_found_with_version_filter(
        self, policy_registry: PolicyRegistry
    ) -> None:
        """Test error message includes version when specified in get()."""
        policy_registry.register_policy(
            policy_id="versioned-policy",
            policy_class=MockSyncPolicy,
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="1.0.0",
        )
        with pytest.raises(PolicyRegistryError) as exc_info:
            policy_registry.get("versioned-policy", version="2.0.0")
        error_msg = str(exc_info.value)
        assert "versioned-policy" in error_msg

    def test_is_registered_with_invalid_policy_type(
        self, policy_registry: PolicyRegistry
    ) -> None:
        """Test is_registered returns False for invalid policy type."""
        policy_registry.register_policy(
            policy_id="test-policy",
            policy_class=MockSyncPolicy,
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="1.0.0",
        )
        # Invalid policy type should return False, not raise
        result = policy_registry.is_registered(
            "test-policy", policy_type="invalid_type"
        )
        assert result is False

    def test_unregister_with_invalid_policy_type(
        self, policy_registry: PolicyRegistry
    ) -> None:
        """Test unregister returns 0 for invalid policy type."""
        policy_registry.register_policy(
            policy_id="test-policy",
            policy_class=MockSyncPolicy,
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="1.0.0",
        )
        # Invalid policy type should return 0, not raise
        result = policy_registry.unregister("test-policy", policy_type="invalid_type")
        assert result == 0
        # Original policy should still be there
        assert policy_registry.is_registered("test-policy")

    def test_unregister_with_policy_type_filter(
        self, policy_registry: PolicyRegistry
    ) -> None:
        """Test unregister with policy_type filter only removes matching."""
        # Register same policy_id with different types
        policy_registry.register_policy(
            policy_id="multi-type",
            policy_class=MockSyncPolicy,
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="1.0.0",
        )
        policy_registry.register_policy(
            policy_id="multi-type",
            policy_class=MockSyncReducerPolicy,
            policy_type=EnumPolicyType.REDUCER,
            version="1.0.0",
        )
        assert len(policy_registry) == 2

        # Unregister only orchestrator
        result = policy_registry.unregister(
            "multi-type", policy_type=EnumPolicyType.ORCHESTRATOR
        )
        assert result == 1
        # Reducer should still exist
        assert policy_registry.is_registered(
            "multi-type", policy_type=EnumPolicyType.REDUCER
        )
        assert not policy_registry.is_registered(
            "multi-type", policy_type=EnumPolicyType.ORCHESTRATOR
        )


# =============================================================================
# TestPolicyRegistryIntegration
# =============================================================================


class TestPolicyRegistryIntegration:
    """Integration tests for PolicyRegistry."""

    def test_full_registration_workflow(self, policy_registry: PolicyRegistry) -> None:
        """Test complete workflow: register, get, list, unregister."""
        # Register
        policy_registry.register_policy(
            policy_id="workflow-test",
            policy_class=MockSyncPolicy,
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="1.0.0",
        )
        assert policy_registry.is_registered("workflow-test")

        # Get
        policy_cls = policy_registry.get("workflow-test")
        assert policy_cls is MockSyncPolicy

        # List
        policies = policy_registry.list()
        assert len(policies) == 1
        assert policies[0][0] == "workflow-test"

        # Unregister
        policy_registry.unregister("workflow-test")
        assert not policy_registry.is_registered("workflow-test")

    def test_multiple_policies_different_types(
        self, policy_registry: PolicyRegistry
    ) -> None:
        """Test registering policies of different types."""
        policy_registry.register_policy(
            policy_id="orchestrator",
            policy_class=MockSyncPolicy,
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="1.0.0",
        )
        policy_registry.register_policy(
            policy_id="reducer",
            policy_class=MockSyncReducerPolicy,
            policy_type=EnumPolicyType.REDUCER,
            version="1.0.0",
        )

        # Both should be registered
        assert policy_registry.is_registered("orchestrator")
        assert policy_registry.is_registered("reducer")

        # Filter by type
        orchestrators = policy_registry.list(policy_type=EnumPolicyType.ORCHESTRATOR)
        reducers = policy_registry.list(policy_type=EnumPolicyType.REDUCER)
        assert len(orchestrators) == 1
        assert len(reducers) == 1

    def test_async_policy_workflow_with_flag(
        self, policy_registry: PolicyRegistry
    ) -> None:
        """Test async policy workflow with deterministic_async=True."""
        # This should succeed with the flag
        policy_registry.register_policy(
            policy_id="async-workflow",
            policy_class=MockAsyncPolicy,
            policy_type=EnumPolicyType.REDUCER,
            version="1.0.0",
            deterministic_async=True,
        )
        assert policy_registry.is_registered("async-workflow")
        policy_cls = policy_registry.get("async-workflow")
        assert policy_cls is MockAsyncPolicy
