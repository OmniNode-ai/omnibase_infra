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
from collections.abc import Iterator
from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

from omnibase_infra.enums import EnumPolicyType
from omnibase_infra.errors import PolicyRegistryError
from omnibase_infra.runtime import policy_registry as registry_module
from omnibase_infra.runtime.policy_registry import (
    PolicyRegistry,
    get_policy_class,
    get_policy_registry,
    register_policy,
)

if TYPE_CHECKING:
    from omnibase_core.container import ModelONEXContainer

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

    def decide(self, context: dict[str, object]) -> dict[str, object]:
        return self.evaluate(context)


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

    async def decide(self, context: dict[str, object]) -> dict[str, object]:
        return await self.evaluate(context)


class MockAsyncDecidePolicy:
    """Mock async policy with decide() method."""

    @property
    def policy_id(self) -> str:
        return "mock-async-decide"

    @property
    def policy_type(self) -> str:
        return "orchestrator"

    async def evaluate(self, context: dict[str, object]) -> dict[str, object]:
        return {"decision": "async"}

    async def decide(self, context: dict[str, object]) -> dict[str, object]:
        return await self.evaluate(context)


class MockAsyncReducePolicy:
    """Mock async policy with reduce() method."""

    @property
    def policy_id(self) -> str:
        return "mock-async-reduce"

    @property
    def policy_type(self) -> str:
        return "reducer"

    async def evaluate(self, context: dict[str, object]) -> dict[str, object]:
        return {"reduced": "async"}

    async def decide(self, context: dict[str, object]) -> dict[str, object]:
        return await self.evaluate(context)

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

    def evaluate(self, context: dict[str, object]) -> dict[str, object]:
        return {"reduced": "sync"}

    def decide(self, context: dict[str, object]) -> dict[str, object]:
        return self.evaluate(context)

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

    def evaluate(self, context: dict[str, object]) -> dict[str, object]:
        return {"decision": "sync"}

    def decide(self, context: dict[str, object]) -> dict[str, object]:
        return self.evaluate(context)


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

    def decide(self, context: dict[str, object]) -> dict[str, object]:
        return self.evaluate(context)


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

    def decide(self, context: dict[str, object]) -> dict[str, object]:
        return self.evaluate(context)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def policy_registry() -> PolicyRegistry:
    """Provide a fresh PolicyRegistry instance for each test.

    Note: This fixture uses direct instantiation for unit testing the PolicyRegistry
    class itself. For integration tests that need container-based access, use
    container_with_policy_registry or container_with_registries fixtures from
    conftest.py.
    """
    return PolicyRegistry()


@pytest.fixture
def populated_policy_registry() -> PolicyRegistry:
    """Provide a PolicyRegistry with pre-registered policies.

    Note: This fixture uses direct instantiation for unit testing the PolicyRegistry
    class itself. For integration tests, use container-based fixtures.
    """
    registry = PolicyRegistry()
    registry.register_policy(
        policy_id="sync-orchestrator",
        policy_class=MockSyncPolicy,  # type: ignore[arg-type]
        policy_type=EnumPolicyType.ORCHESTRATOR,
        version="1.0.0",
    )  # type: ignore[arg-type]
    registry.register_policy(
        policy_id="sync-reducer",
        policy_class=MockSyncReducerPolicy,  # type: ignore[arg-type]
        policy_type=EnumPolicyType.REDUCER,
        version="1.0.0",
    )  # type: ignore[arg-type]
    return registry


@pytest.fixture(autouse=True)
def reset_singletons() -> Iterator[None]:
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
            policy_class=MockSyncPolicy,  # type: ignore[arg-type]
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="1.0.0",
        )  # type: ignore[arg-type]
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
            policy_class=MockSyncPolicy,  # type: ignore[arg-type]
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="1.0.0",
        )  # type: ignore[arg-type]
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
            policy_class=MockSyncReducerPolicy,  # type: ignore[arg-type]
            policy_type=EnumPolicyType.REDUCER,
            version="1.0.0",
        )  # type: ignore[arg-type]
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
            policy_class=MockSyncPolicy,  # type: ignore[arg-type]
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="1.0.0",
        )  # type: ignore[arg-type]
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
                policy_class=MockAsyncPolicy,  # type: ignore[arg-type]
                policy_type=EnumPolicyType.REDUCER,
                version="1.0.0",
                # deterministic_async defaults to False
            )  # type: ignore[arg-type]
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
            policy_class=MockAsyncPolicy,  # type: ignore[arg-type]
            policy_type=EnumPolicyType.REDUCER,
            version="1.0.0",
            deterministic_async=True,
        )  # type: ignore[arg-type]
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
                policy_class=MockAsyncPolicy,  # type: ignore[arg-type]
                policy_type=EnumPolicyType.ORCHESTRATOR,
                version="1.0.0",
                deterministic_async=False,
            )  # type: ignore[arg-type]
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
                policy_class=MockAsyncDecidePolicy,  # type: ignore[arg-type]
                policy_type=EnumPolicyType.ORCHESTRATOR,
                version="1.0.0",
                deterministic_async=False,
            )  # type: ignore[arg-type]
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
                policy_class=MockAsyncReducePolicy,  # type: ignore[arg-type]
                policy_type=EnumPolicyType.REDUCER,
                version="1.0.0",
                deterministic_async=False,
            )  # type: ignore[arg-type]
        error_msg = str(exc_info.value)
        # Should mention the async reduce method
        assert "reduce" in error_msg.lower()

    def test_sync_decide_method_succeeds(self, policy_registry: PolicyRegistry) -> None:
        """Test that sync decide() method policy registers successfully."""
        policy_registry.register_policy(
            policy_id="sync-decide",
            policy_class=MockSyncDecidePolicy,  # type: ignore[arg-type]
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="1.0.0",
        )  # type: ignore[arg-type]
        assert policy_registry.is_registered("sync-decide")

    def test_sync_reduce_method_succeeds(self, policy_registry: PolicyRegistry) -> None:
        """Test that sync reduce() method policy registers successfully."""
        policy_registry.register_policy(
            policy_id="sync-reduce",
            policy_class=MockSyncReducerPolicy,  # type: ignore[arg-type]
            policy_type=EnumPolicyType.REDUCER,
            version="1.0.0",
        )  # type: ignore[arg-type]
        assert policy_registry.is_registered("sync-reduce")


# =============================================================================
# TestPolicyRegistryList
# =============================================================================


class TestPolicyRegistryList:
    """Tests for list() method."""

    def test_list_all_policies(self, populated_policy_registry: PolicyRegistry) -> None:
        """Test that list_keys returns (id, type, version) tuples."""
        policies = populated_policy_registry.list_keys()
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
        """Test filtering list_keys by policy type."""
        # List only orchestrator policies
        orchestrator_policies = populated_policy_registry.list_keys(
            policy_type=EnumPolicyType.ORCHESTRATOR
        )
        assert len(orchestrator_policies) == 1
        assert orchestrator_policies[0][1] == "orchestrator"

        # List only reducer policies
        reducer_policies = populated_policy_registry.list_keys(
            policy_type=EnumPolicyType.REDUCER
        )
        assert len(reducer_policies) == 1
        assert reducer_policies[0][1] == "reducer"

    def test_list_empty_registry(self, policy_registry: PolicyRegistry) -> None:
        """Test that empty registry returns empty list."""
        policies = policy_registry.list_keys()
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
            policy_class=MockPolicyV1,  # type: ignore[arg-type]
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="1.0.0",
        )  # type: ignore[arg-type]
        policy_registry.register_policy(
            policy_id="versioned-policy",
            policy_class=MockPolicyV2,  # type: ignore[arg-type]
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="2.0.0",
        )  # type: ignore[arg-type]
        assert len(policy_registry) == 2

    def test_get_specific_version(self, policy_registry: PolicyRegistry) -> None:
        """Test retrieving a specific version."""
        policy_registry.register_policy(
            policy_id="versioned-policy",
            policy_class=MockPolicyV1,  # type: ignore[arg-type]
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="1.0.0",
        )  # type: ignore[arg-type]
        policy_registry.register_policy(
            policy_id="versioned-policy",
            policy_class=MockPolicyV2,  # type: ignore[arg-type]
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="2.0.0",
        )  # type: ignore[arg-type]

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
            policy_class=MockPolicyV1,  # type: ignore[arg-type]
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="1.0.0",
        )  # type: ignore[arg-type]
        policy_registry.register_policy(
            policy_id="versioned-policy",
            policy_class=MockPolicyV2,  # type: ignore[arg-type]
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="2.0.0",
        )  # type: ignore[arg-type]

        # Get without version should return latest (semantically highest)
        latest_cls = policy_registry.get("versioned-policy")
        assert latest_cls is MockPolicyV2

    def test_invalid_version_format_raises_error_on_registration(
        self, policy_registry: PolicyRegistry
    ) -> None:
        """Test that invalid version formats raise ProtocolConfigurationError at registration.

        Version validation happens during registration via _parse_semver,
        preventing invalid versions from being registered.
        """
        from omnibase_infra.errors import ProtocolConfigurationError

        # Attempt to register with invalid version format
        with pytest.raises(ProtocolConfigurationError) as exc_info:
            policy_registry.register_policy(
                policy_id="version-test-policy",
                policy_class=MockPolicyV1,  # type: ignore[arg-type]
                policy_type=EnumPolicyType.ORCHESTRATOR,
                version="not-a-version",  # Invalid format - not semver
            )  # type: ignore[arg-type]

        # Verify error message contains version and guidance
        assert "not-a-version" in str(exc_info.value)
        assert "Invalid semantic version format" in str(exc_info.value)
        assert "Version components must be integers" in str(exc_info.value)

        # Registry should be empty (registration failed)
        assert len(policy_registry) == 0

    def test_malformed_semver_components_raises_error(
        self, policy_registry: PolicyRegistry
    ) -> None:
        """Test that malformed semantic version components raise errors at registration.

        Versions with non-numeric parts should raise ProtocolConfigurationError
        immediately during register_policy() call.
        """
        from omnibase_infra.errors import ProtocolConfigurationError

        # Attempt to register with malformed version components
        with pytest.raises(ProtocolConfigurationError) as exc_info:
            policy_registry.register_policy(
                policy_id="malformed-semver",
                policy_class=MockPolicyV1,  # type: ignore[arg-type]
                policy_type=EnumPolicyType.ORCHESTRATOR,
                version="v1.x.y",  # Non-numeric components
            )  # type: ignore[arg-type]

        # Verify error message
        assert "v1.x.y" in str(exc_info.value)
        assert "Version components must be integers" in str(exc_info.value)

        # Registry should be empty
        assert len(policy_registry) == 0

    def test_empty_version_string_raises_error(
        self, policy_registry: PolicyRegistry
    ) -> None:
        """Test that empty version strings raise ProtocolConfigurationError at registration."""
        from omnibase_infra.errors import ProtocolConfigurationError

        # Attempt to register with empty version
        with pytest.raises(ProtocolConfigurationError) as exc_info:
            policy_registry.register_policy(
                policy_id="empty-version",
                policy_class=MockPolicyV1,  # type: ignore[arg-type]
                policy_type=EnumPolicyType.ORCHESTRATOR,
                version="",  # Empty version
            )  # type: ignore[arg-type]

        # Verify error message
        assert "empty version string" in str(exc_info.value).lower()

        # Registry should be empty
        assert len(policy_registry) == 0

    def test_version_with_too_many_parts_raises_error(
        self, policy_registry: PolicyRegistry
    ) -> None:
        """Test that versions with more than 3 parts raise error."""
        from omnibase_infra.errors import ProtocolConfigurationError

        # Attempt to register with too many version parts
        with pytest.raises(ProtocolConfigurationError) as exc_info:
            policy_registry.register_policy(
                policy_id="too-many-parts",
                policy_class=MockPolicyV1,  # type: ignore[arg-type]
                policy_type=EnumPolicyType.ORCHESTRATOR,
                version="1.2.3.4",  # Four parts (invalid)
            )  # type: ignore[arg-type]

        # Verify error mentions format
        assert "1.2.3.4" in str(exc_info.value)
        assert "Invalid semantic version format" in str(exc_info.value)

    def test_get_latest_with_double_digit_versions(
        self, policy_registry: PolicyRegistry
    ) -> None:
        """Test semver sorting handles double-digit versions correctly.

        This tests the fix for lexicographic sorting which would incorrectly
        sort "10.0.0" before "2.0.0" (because '1' < '2' as strings).
        """
        policy_registry.register_policy(
            policy_id="semver-policy",
            policy_class=MockPolicyV1,  # type: ignore[arg-type]
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="2.0.0",
        )  # type: ignore[arg-type]
        policy_registry.register_policy(
            policy_id="semver-policy",
            policy_class=MockPolicyV2,  # type: ignore[arg-type]
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="10.0.0",
        )  # type: ignore[arg-type]

        # Get without version should return 10.0.0 (semver highest), not 2.0.0
        latest_cls = policy_registry.get("semver-policy")
        assert latest_cls is MockPolicyV2, (
            "10.0.0 should be considered later than 2.0.0"
        )

    def test_get_latest_with_prerelease_versions(
        self, policy_registry: PolicyRegistry
    ) -> None:
        """Test semver sorting prefers release over prerelease versions."""
        policy_registry.register_policy(
            policy_id="prerelease-policy",
            policy_class=MockPolicyV1,  # type: ignore[arg-type]
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="1.0.0-alpha",
        )  # type: ignore[arg-type]
        policy_registry.register_policy(
            policy_id="prerelease-policy",
            policy_class=MockPolicyV2,  # type: ignore[arg-type]
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="1.0.0",
        )  # type: ignore[arg-type]

        # Get without version should return release (1.0.0), not prerelease (1.0.0-alpha)
        latest_cls = policy_registry.get("prerelease-policy")
        assert latest_cls is MockPolicyV2, "Release should be preferred over prerelease"

    def test_list_versions(self, policy_registry: PolicyRegistry) -> None:
        """Test list_versions() method."""
        policy_registry.register_policy(
            policy_id="versioned-policy",
            policy_class=MockPolicyV1,  # type: ignore[arg-type]
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="1.0.0",
        )  # type: ignore[arg-type]
        policy_registry.register_policy(
            policy_id="versioned-policy",
            policy_class=MockPolicyV2,  # type: ignore[arg-type]
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="2.0.0",
        )  # type: ignore[arg-type]

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
            policy_class=MockPolicyV1,  # type: ignore[arg-type]
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="1.0.0",
        )  # type: ignore[arg-type]
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
            policy_class=MockSyncPolicy,  # type: ignore[arg-type]
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="1.0.0",
        )  # type: ignore[arg-type]
        assert policy_registry.is_registered("to-remove")

        policy_registry.unregister("to-remove")
        assert not policy_registry.is_registered("to-remove")

    def test_unregister_returns_count_when_found(
        self, policy_registry: PolicyRegistry
    ) -> None:
        """Test unregister returns count of removed policies."""
        policy_registry.register_policy(
            policy_id="to-remove",
            policy_class=MockSyncPolicy,  # type: ignore[arg-type]
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="1.0.0",
        )  # type: ignore[arg-type]
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
            policy_class=MockPolicyV1,  # type: ignore[arg-type]
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="1.0.0",
        )  # type: ignore[arg-type]
        policy_registry.register_policy(
            policy_id="versioned",
            policy_class=MockPolicyV2,  # type: ignore[arg-type]
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="2.0.0",
        )  # type: ignore[arg-type]
        result = policy_registry.unregister("versioned")
        assert result == 2
        assert not policy_registry.is_registered("versioned")

    def test_unregister_specific_version(self, policy_registry: PolicyRegistry) -> None:
        """Test unregister with specific version."""
        policy_registry.register_policy(
            policy_id="versioned",
            policy_class=MockPolicyV1,  # type: ignore[arg-type]
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="1.0.0",
        )  # type: ignore[arg-type]
        policy_registry.register_policy(
            policy_id="versioned",
            policy_class=MockPolicyV2,  # type: ignore[arg-type]
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="2.0.0",
        )  # type: ignore[arg-type]
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
        assert populated_policy_registry.list_keys() == []


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
            policy_class=MockSyncPolicy,  # type: ignore[arg-type]
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="1.0.0",
        )  # type: ignore[arg-type]
        assert len(policy_registry) == 1

        policy_registry.register_policy(
            policy_id="policy2",
            policy_class=MockSyncReducerPolicy,  # type: ignore[arg-type]
            policy_type=EnumPolicyType.REDUCER,
            version="1.0.0",
        )  # type: ignore[arg-type]
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
            policy_class=MockSyncPolicy,  # type: ignore[arg-type]
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="1.0.0",
        )  # type: ignore[arg-type]
        # Retrieve via convenience function
        policy_cls = get_policy_class("singleton-test")
        assert policy_cls is MockSyncPolicy

    def test_register_policy_uses_singleton(self) -> None:
        """Test that register_policy convenience function uses singleton."""
        register_policy(
            policy_id="register-test",
            policy_class=MockSyncPolicy,  # type: ignore[arg-type]
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="1.0.0",
        )  # type: ignore[arg-type]
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

    def test_error_with_enum_policy_type(self) -> None:
        """Test that PolicyRegistryError accepts EnumPolicyType enum value."""
        error = PolicyRegistryError(
            "Policy operation failed",
            policy_id="test-policy",
            policy_type=EnumPolicyType.ORCHESTRATOR,
        )
        # EnumPolicyType should be converted to string for serialization
        assert error.model.context.get("policy_type") == "orchestrator"
        assert error.model.context.get("policy_id") == "test-policy"

    def test_error_with_enum_reducer_policy_type(self) -> None:
        """Test that PolicyRegistryError accepts EnumPolicyType.REDUCER."""
        error = PolicyRegistryError(
            "Reducer policy failed",
            policy_id="reducer-policy",
            policy_type=EnumPolicyType.REDUCER,
        )
        # EnumPolicyType.REDUCER should be converted to "reducer"
        assert error.model.context.get("policy_type") == "reducer"
        assert error.model.context.get("policy_id") == "reducer-policy"

    def test_error_with_string_and_enum_compatibility(self) -> None:
        """Test that string and enum policy_type produce equivalent errors."""
        error_with_string = PolicyRegistryError(
            "Test error",
            policy_id="test-policy",
            policy_type="orchestrator",
        )
        error_with_enum = PolicyRegistryError(
            "Test error",
            policy_id="test-policy",
            policy_type=EnumPolicyType.ORCHESTRATOR,
        )
        # Both should result in the same serialized policy_type
        assert error_with_string.model.context.get("policy_type") == "orchestrator"
        assert error_with_enum.model.context.get("policy_type") == "orchestrator"
        assert error_with_string.model.context.get(
            "policy_type"
        ) == error_with_enum.model.context.get("policy_type")


# =============================================================================
# TestPolicyRegistryPolicyTypeNormalization
# =============================================================================


class TestPolicyRegistryPolicyTypeNormalization:
    """Tests for policy type normalization."""

    def test_register_with_enum_type(self, policy_registry: PolicyRegistry) -> None:
        """Test registering with EnumPolicyType enum value."""
        policy_registry.register_policy(
            policy_id="enum-type",
            policy_class=MockSyncPolicy,  # type: ignore[arg-type]
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="1.0.0",
        )  # type: ignore[arg-type]
        assert policy_registry.is_registered("enum-type")

    def test_register_with_string_type(self, policy_registry: PolicyRegistry) -> None:
        """Test registering with string policy type."""
        policy_registry.register_policy(
            policy_id="string-type",
            policy_class=MockSyncPolicy,  # type: ignore[arg-type]
            policy_type="orchestrator",
            version="1.0.0",
        )  # type: ignore[arg-type]
        assert policy_registry.is_registered("string-type")

    def test_invalid_policy_type_raises_error(
        self, policy_registry: PolicyRegistry
    ) -> None:
        """Test that invalid policy type string raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            policy_registry.register_policy(
                policy_id="invalid-type",
                policy_class=MockSyncPolicy,  # type: ignore[arg-type]
                policy_type="invalid_type",
                version="1.0.0",
            )  # type: ignore[arg-type]
        assert "invalid_type" in str(exc_info.value)

    def test_get_with_enum_and_string_equivalent(
        self, policy_registry: PolicyRegistry
    ) -> None:
        """Test that get() works with both enum and string type."""
        policy_registry.register_policy(
            policy_id="type-test",
            policy_class=MockSyncPolicy,  # type: ignore[arg-type]
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="1.0.0",
        )  # type: ignore[arg-type]
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
            policy_class=MockSyncPolicy,  # type: ignore[arg-type]
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="1.0.0",
        )  # type: ignore[arg-type]
        policy_registry.register_policy(
            policy_id="policy2",
            policy_class=MockSyncDecidePolicy,  # type: ignore[arg-type]
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="1.0.0",
        )  # type: ignore[arg-type]
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
            policy_class=MockPolicyV1,  # type: ignore[arg-type]
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="1.0.0",
        )  # type: ignore[arg-type]
        policy_registry.register_policy(
            policy_id="overwrite-test",
            policy_class=MockPolicyV2,  # type: ignore[arg-type]
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="1.0.0",
        )  # type: ignore[arg-type]
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
            policy_class=MockSyncPolicy,  # type: ignore[arg-type]
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="1.0.0",
        )  # type: ignore[arg-type]
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
            policy_class=MockSyncPolicy,  # type: ignore[arg-type]
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="1.0.0",
        )  # type: ignore[arg-type]
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
            policy_class=MockSyncPolicy,  # type: ignore[arg-type]
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="1.0.0",
        )  # type: ignore[arg-type]
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
            policy_class=MockSyncPolicy,  # type: ignore[arg-type]
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="1.0.0",
        )  # type: ignore[arg-type]
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
            policy_class=MockSyncPolicy,  # type: ignore[arg-type]
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="1.0.0",
        )  # type: ignore[arg-type]
        policy_registry.register_policy(
            policy_id="multi-type",
            policy_class=MockSyncReducerPolicy,  # type: ignore[arg-type]
            policy_type=EnumPolicyType.REDUCER,
            version="1.0.0",
        )  # type: ignore[arg-type]
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
            policy_class=MockSyncPolicy,  # type: ignore[arg-type]
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="1.0.0",
        )  # type: ignore[arg-type]
        assert policy_registry.is_registered("workflow-test")

        # Get
        policy_cls = policy_registry.get("workflow-test")
        assert policy_cls is MockSyncPolicy

        # List
        policies = policy_registry.list_keys()
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
            policy_class=MockSyncPolicy,  # type: ignore[arg-type]
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="1.0.0",
        )  # type: ignore[arg-type]
        policy_registry.register_policy(
            policy_id="reducer",
            policy_class=MockSyncReducerPolicy,  # type: ignore[arg-type]
            policy_type=EnumPolicyType.REDUCER,
            version="1.0.0",
        )  # type: ignore[arg-type]

        # Both should be registered
        assert policy_registry.is_registered("orchestrator")
        assert policy_registry.is_registered("reducer")

        # Filter by type
        orchestrators = policy_registry.list_keys(
            policy_type=EnumPolicyType.ORCHESTRATOR
        )
        reducers = policy_registry.list_keys(policy_type=EnumPolicyType.REDUCER)
        assert len(orchestrators) == 1
        assert len(reducers) == 1

    def test_async_policy_workflow_with_flag(
        self, policy_registry: PolicyRegistry
    ) -> None:
        """Test async policy workflow with deterministic_async=True."""
        # This should succeed with the flag
        policy_registry.register_policy(
            policy_id="async-workflow",
            policy_class=MockAsyncPolicy,  # type: ignore[arg-type]
            policy_type=EnumPolicyType.REDUCER,
            version="1.0.0",
            deterministic_async=True,
        )  # type: ignore[arg-type]
        assert policy_registry.is_registered("async-workflow")
        policy_cls = policy_registry.get("async-workflow")
        assert policy_cls is MockAsyncPolicy


# =============================================================================
# TestPolicyRegistrySemverCaching
# =============================================================================


class TestPolicyRegistrySemverCaching:
    """Tests for _parse_semver() caching behavior.

    Validates that the LRU cache improves performance and correctly
    handles cache hits/misses for version string parsing.
    """

    def test_parse_semver_returns_consistent_results(self) -> None:
        """Test that _parse_semver returns consistent results for same input."""
        # Clear cache to ensure clean state
        PolicyRegistry._parse_semver.cache_clear()

        # Parse same version multiple times
        result1 = PolicyRegistry._parse_semver("1.2.3")
        result2 = PolicyRegistry._parse_semver("1.2.3")
        result3 = PolicyRegistry._parse_semver("1.2.3")

        # All should return identical tuples
        assert result1 == result2 == result3
        assert result1 == (1, 2, 3, chr(127))  # chr(127) for release version

    def test_parse_semver_cache_info_shows_hits(self) -> None:
        """Test that cache info shows hits for repeated parses."""
        # Clear cache to ensure clean state
        PolicyRegistry._parse_semver.cache_clear()
        initial_info = PolicyRegistry._parse_semver.cache_info()
        assert initial_info.hits == 0
        assert initial_info.misses == 0

        # First parse - should be a cache miss
        PolicyRegistry._parse_semver("1.0.0")
        info_after_first = PolicyRegistry._parse_semver.cache_info()
        assert info_after_first.misses == 1
        assert info_after_first.hits == 0

        # Second parse of same version - should be a cache hit
        PolicyRegistry._parse_semver("1.0.0")
        info_after_second = PolicyRegistry._parse_semver.cache_info()
        assert info_after_second.misses == 1
        assert info_after_second.hits == 1

        # Third parse - another hit
        PolicyRegistry._parse_semver("1.0.0")
        info_after_third = PolicyRegistry._parse_semver.cache_info()
        assert info_after_third.misses == 1
        assert info_after_third.hits == 2

    def test_parse_semver_different_versions_cause_misses(self) -> None:
        """Test that different version strings cause cache misses."""
        # Clear cache to ensure clean state
        PolicyRegistry._parse_semver.cache_clear()

        # Parse different versions
        PolicyRegistry._parse_semver("1.0.0")
        PolicyRegistry._parse_semver("2.0.0")
        PolicyRegistry._parse_semver("3.0.0")

        info = PolicyRegistry._parse_semver.cache_info()
        assert info.misses == 3
        assert info.hits == 0
        assert info.currsize == 3  # 3 entries cached

    def test_parse_semver_cache_improves_get_performance(
        self, policy_registry: PolicyRegistry
    ) -> None:
        """Test that caching improves performance for repeated get() calls."""
        # Register multiple versions of same policy
        for i in range(10):
            policy_registry.register_policy(
                policy_id="perf-test",
                policy_class=MockPolicyV1,  # type: ignore[arg-type]
                policy_type=EnumPolicyType.ORCHESTRATOR,
                version=f"{i}.0.0",
            )  # type: ignore[arg-type]

        # Clear cache to measure baseline
        PolicyRegistry._parse_semver.cache_clear()

        # First get() will parse all versions (cold cache)
        _ = policy_registry.get("perf-test")  # Gets latest version
        first_misses = PolicyRegistry._parse_semver.cache_info().misses

        # Second get() should hit cache (warm cache)
        _ = policy_registry.get("perf-test")
        second_misses = PolicyRegistry._parse_semver.cache_info().misses

        # Cache should have been hit (no new misses)
        assert second_misses == first_misses

    def test_parse_semver_cache_handles_prerelease_versions(self) -> None:
        """Test that cache correctly handles prerelease version strings."""
        # Clear cache to ensure clean state
        PolicyRegistry._parse_semver.cache_clear()

        # Parse prerelease versions
        result1 = PolicyRegistry._parse_semver("1.0.0-alpha")
        result2 = PolicyRegistry._parse_semver("1.0.0-beta")
        result3 = PolicyRegistry._parse_semver("1.0.0")

        # All should be distinct cache entries
        assert result1 != result2 != result3
        info = PolicyRegistry._parse_semver.cache_info()
        assert info.currsize == 3

        # Repeat parse should hit cache
        result1_repeat = PolicyRegistry._parse_semver("1.0.0-alpha")
        assert result1_repeat == result1
        info_after = PolicyRegistry._parse_semver.cache_info()
        assert info_after.hits == 1

    def test_parse_semver_cache_size_limit(self) -> None:
        """Test that cache respects maxsize=128 limit."""
        # Clear cache to ensure clean state
        PolicyRegistry._parse_semver.cache_clear()

        # Parse 150 unique versions (exceeds maxsize=128)
        for i in range(150):
            PolicyRegistry._parse_semver(f"{i}.0.0")

        info = PolicyRegistry._parse_semver.cache_info()
        # Cache size should not exceed maxsize
        assert info.currsize <= 128

    def test_parse_semver_cache_lru_eviction(self) -> None:
        """Test that LRU eviction works correctly."""
        # Clear cache to ensure clean state
        PolicyRegistry._parse_semver.cache_clear()

        # Fill cache to capacity with versions 0-127
        for i in range(128):
            PolicyRegistry._parse_semver(f"{i}.0.0")

        # Access version "0.0.0" to make it most recently used
        PolicyRegistry._parse_semver("0.0.0")

        # Add new version to trigger eviction (should evict "1.0.0", not "0.0.0")
        PolicyRegistry._parse_semver("999.0.0")

        # "0.0.0" should still be in cache (was recently used)
        PolicyRegistry._parse_semver("0.0.0")
        info = PolicyRegistry._parse_semver.cache_info()
        # Last access to "0.0.0" should be a hit
        assert info.hits > 0

    def test_parse_semver_cache_clear_resets_state(self) -> None:
        """Test that cache_clear() resets cache state."""
        # Parse some versions
        PolicyRegistry._parse_semver("1.0.0")
        PolicyRegistry._parse_semver("2.0.0")
        info_before = PolicyRegistry._parse_semver.cache_info()
        assert info_before.currsize > 0

        # Clear cache
        PolicyRegistry._parse_semver.cache_clear()
        info_after = PolicyRegistry._parse_semver.cache_info()

        # Cache should be empty
        assert info_after.currsize == 0
        assert info_after.hits == 0
        assert info_after.misses == 0


# =============================================================================
# TestPolicyRegistryInvalidVersions
# =============================================================================


class TestPolicyRegistryInvalidVersions:
    """Tests for version validation and error handling.

    This tests the PR #36 review feedback requirement:
    - Invalid versions should raise ProtocolConfigurationError
    - No silent fallback to (0, 0, 0)
    """

    def test_invalid_version_format_empty_string(
        self, policy_registry: PolicyRegistry
    ) -> None:
        """Test that empty version string raises ProtocolConfigurationError."""
        from omnibase_infra.errors import ProtocolConfigurationError

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            policy_registry.register_policy(
                policy_id="invalid-version",
                policy_class=MockSyncPolicy,  # type: ignore[arg-type]
                policy_type=EnumPolicyType.ORCHESTRATOR,
                version="",
            )
        assert "Invalid semantic version format" in str(exc_info.value)

    def test_invalid_version_format_non_numeric(
        self, policy_registry: PolicyRegistry
    ) -> None:
        """Test that non-numeric version components raise ProtocolConfigurationError."""
        from omnibase_infra.errors import ProtocolConfigurationError

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            policy_registry.register_policy(
                policy_id="invalid-version",
                policy_class=MockSyncPolicy,  # type: ignore[arg-type]
                policy_type=EnumPolicyType.ORCHESTRATOR,
                version="abc.def.ghi",
            )
        assert "Invalid semantic version format" in str(exc_info.value)
        assert "must be integers" in str(exc_info.value)

    def test_invalid_version_format_negative_numbers(
        self, policy_registry: PolicyRegistry
    ) -> None:
        """Test that negative version numbers raise ProtocolConfigurationError."""
        from omnibase_infra.errors import ProtocolConfigurationError

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            policy_registry.register_policy(
                policy_id="invalid-version",
                policy_class=MockSyncPolicy,  # type: ignore[arg-type]
                policy_type=EnumPolicyType.ORCHESTRATOR,
                version="1.-1.0",
            )
        # Negative numbers cause ValueError in int() conversion
        assert "Invalid semantic version format" in str(exc_info.value)
        assert "1.-1.0" in str(exc_info.value)

    def test_invalid_version_format_too_many_parts(
        self, policy_registry: PolicyRegistry
    ) -> None:
        """Test that version with too many parts raises ProtocolConfigurationError."""
        from omnibase_infra.errors import ProtocolConfigurationError

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            policy_registry.register_policy(
                policy_id="invalid-version",
                policy_class=MockSyncPolicy,  # type: ignore[arg-type]
                policy_type=EnumPolicyType.ORCHESTRATOR,
                version="1.2.3.4",
            )
        assert "Invalid semantic version format" in str(exc_info.value)

    def test_valid_version_major_only(self, policy_registry: PolicyRegistry) -> None:
        """Test that single component version (major only) is valid."""
        policy_registry.register_policy(
            policy_id="major-only",
            policy_class=MockSyncPolicy,  # type: ignore[arg-type]
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="1",
        )
        assert policy_registry.is_registered("major-only", version="1")

    def test_valid_version_major_minor(self, policy_registry: PolicyRegistry) -> None:
        """Test that two component version (major.minor) is valid."""
        policy_registry.register_policy(
            policy_id="major-minor",
            policy_class=MockSyncPolicy,  # type: ignore[arg-type]
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="1.2",
        )
        assert policy_registry.is_registered("major-minor", version="1.2")

    def test_semver_comparison_edge_case_1_9_vs_1_10(
        self, policy_registry: PolicyRegistry
    ) -> None:
        """Test the specific PR #36 case: 1.9.0 vs 1.10.0.

        This is the exact bug from the PR review:
        - Lexicographic: "1.10.0" < "1.9.0" (WRONG - because '1' < '9')
        - Semantic: 1.10.0 > 1.9.0 (CORRECT)
        """
        policy_registry.register_policy(
            policy_id="version-test",
            policy_class=MockPolicyV1,  # type: ignore[arg-type]
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="1.9.0",
        )
        policy_registry.register_policy(
            policy_id="version-test",
            policy_class=MockPolicyV2,  # type: ignore[arg-type]
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="1.10.0",
        )

        # Get latest should return 1.10.0 (MockPolicyV2), not 1.9.0
        latest_cls = policy_registry.get("version-test")
        assert latest_cls is MockPolicyV2, (
            "1.10.0 should be considered later than 1.9.0 (semantic versioning)"
        )

    def test_semver_comparison_minor_version_edge_case(
        self, policy_registry: PolicyRegistry
    ) -> None:
        """Test edge case: 0.9.0 vs 0.10.0."""
        policy_registry.register_policy(
            policy_id="minor-test",
            policy_class=MockPolicyV1,  # type: ignore[arg-type]
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="0.9.0",
        )
        policy_registry.register_policy(
            policy_id="minor-test",
            policy_class=MockPolicyV2,  # type: ignore[arg-type]
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="0.10.0",
        )

        latest_cls = policy_registry.get("minor-test")
        assert latest_cls is MockPolicyV2, "0.10.0 > 0.9.0"

    def test_semver_comparison_patch_version_edge_case(
        self, policy_registry: PolicyRegistry
    ) -> None:
        """Test edge case: 1.0.9 vs 1.0.10."""
        policy_registry.register_policy(
            policy_id="patch-test",
            policy_class=MockPolicyV1,  # type: ignore[arg-type]
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="1.0.9",
        )
        policy_registry.register_policy(
            policy_id="patch-test",
            policy_class=MockPolicyV2,  # type: ignore[arg-type]
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="1.0.10",
        )

        latest_cls = policy_registry.get("patch-test")
        assert latest_cls is MockPolicyV2, "1.0.10 > 1.0.9"


# =============================================================================
# Container-Based DI Integration Tests (OMN-868 Phase 3)
# =============================================================================


class TestPolicyRegistryContainerIntegration:
    """Integration tests using container-based DI patterns (OMN-868 Phase 3).

    These tests demonstrate the container-based access pattern that should be
    used in production code. Unit tests above use direct instantiation to test
    the PolicyRegistry class itself, but integration tests should use containers.
    """

    def test_container_provides_policy_registry_via_mock(
        self, container_with_policy_registry: PolicyRegistry
    ) -> None:
        """Test that container fixture provides PolicyRegistry."""
        assert isinstance(container_with_policy_registry, PolicyRegistry)
        assert len(container_with_policy_registry) == 0

    async def test_container_with_registries_provides_policy_registry(
        self, container_with_registries: ModelONEXContainer
    ) -> None:
        """Test that real container fixture provides PolicyRegistry."""
        # Resolve from container (async in omnibase_core 0.4+)
        registry: PolicyRegistry = (
            await container_with_registries.service_registry.resolve_service(
                PolicyRegistry
            )
        )
        assert isinstance(registry, PolicyRegistry)

    async def test_container_based_policy_registration_workflow(
        self, container_with_registries: ModelONEXContainer
    ) -> None:
        """Test full workflow using container-based DI."""
        # Step 1: Resolve registry from container
        registry: PolicyRegistry = (
            await container_with_registries.service_registry.resolve_service(
                PolicyRegistry
            )
        )

        # Step 2: Register policy
        registry.register_policy(
            policy_id="container-test",
            policy_class=MockSyncPolicy,  # type: ignore[arg-type]
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="1.0.0",
        )

        # Step 3: Verify registration
        assert registry.is_registered("container-test")
        policy_cls = registry.get("container-test")
        assert policy_cls is MockSyncPolicy

    async def test_container_isolation_between_tests(
        self, container_with_registries: ModelONEXContainer
    ) -> None:
        """Test that container provides isolated registry per test."""
        registry: PolicyRegistry = (
            await container_with_registries.service_registry.resolve_service(
                PolicyRegistry
            )
        )

        # This test should start with empty registry (no pollution from other tests)
        assert len(registry) == 0

        # Register a policy
        registry.register_policy(
            policy_id="isolation-test",
            policy_class=MockSyncPolicy,  # type: ignore[arg-type]
            policy_type=EnumPolicyType.ORCHESTRATOR,
        )

        assert len(registry) == 1
