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
- Container-based DI integration
- Thread safety
- Error handling for missing registrations
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import pytest
from omnibase_core.models.primitives.model_semver import ModelSemVer
from pydantic import ValidationError

from omnibase_infra.enums import EnumPolicyType
from omnibase_infra.errors import PolicyRegistryError, ProtocolConfigurationError
from omnibase_infra.runtime.models import ModelPolicyKey
from omnibase_infra.runtime.policy_registry import PolicyRegistry

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
    require explicit allow_async=True flag.
    """

    def test_sync_policy_registration_succeeds(
        self, policy_registry: PolicyRegistry
    ) -> None:
        """Test that synchronous policy registers without issues."""
        # Should not raise - sync policy with default allow_async=False
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
        """Test that async policy without allow_async=True raises error.

        Note: The validation checks methods in order (reduce, decide, evaluate),
        so the error may mention whichever async method is found first.
        MockAsyncPolicy has both async decide() and async evaluate(), so either
        may appear in the error message.
        """
        with pytest.raises(PolicyRegistryError) as exc_info:
            policy_registry.register_policy(
                policy_id="async-policy",
                policy_class=MockAsyncPolicy,  # type: ignore[arg-type]
                policy_type=EnumPolicyType.REDUCER,
                version="1.0.0",
                # allow_async defaults to False
            )  # type: ignore[arg-type]
        error_msg = str(exc_info.value)
        assert "async-policy" in error_msg
        assert "async" in error_msg.lower()
        # The error may mention any async method found (decide or evaluate)
        assert (
            "evaluate" in error_msg.lower()
            or "decide" in error_msg.lower()
            or "deterministic" in error_msg.lower()
        )

    def test_async_policy_with_flag_succeeds(
        self, policy_registry: PolicyRegistry
    ) -> None:
        """Test that async policy with allow_async=True registers OK."""
        # Should not raise with explicit flag
        policy_registry.register_policy(
            policy_id="async-policy",
            policy_class=MockAsyncPolicy,  # type: ignore[arg-type]
            policy_type=EnumPolicyType.REDUCER,
            version="1.0.0",
            allow_async=True,
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
                allow_async=False,
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
                allow_async=False,
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
                allow_async=False,
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
        # Attempt to register with invalid version format
        with pytest.raises(ProtocolConfigurationError) as exc_info:
            policy_registry.register_policy(
                policy_id="version-test-policy",
                policy_class=MockPolicyV1,  # type: ignore[arg-type]
                policy_type=EnumPolicyType.ORCHESTRATOR,
                version="not-a-version",  # Invalid format - not semver
            )  # type: ignore[arg-type]

        # Verify error message indicates invalid version format
        error_msg = str(exc_info.value).lower()
        assert "invalid" in error_msg
        assert "version" in error_msg or "format" in error_msg

        # Registry should be empty (registration failed)
        assert len(policy_registry) == 0

    def test_malformed_semver_components_raises_error(
        self, policy_registry: PolicyRegistry
    ) -> None:
        """Test that malformed semantic version components raise errors at registration.

        Versions with non-numeric parts should raise ProtocolConfigurationError
        immediately during register_policy() call.
        """
        # Attempt to register with malformed version components
        with pytest.raises(ProtocolConfigurationError) as exc_info:
            policy_registry.register_policy(
                policy_id="malformed-semver",
                policy_class=MockPolicyV1,  # type: ignore[arg-type]
                policy_type=EnumPolicyType.ORCHESTRATOR,
                version="v1.x.y",  # Non-numeric components
            )  # type: ignore[arg-type]

        # Verify error message indicates version validation failure
        # Note: 'v' prefix is stripped during version normalization, so error shows "1.x.y"
        error_msg = str(exc_info.value)
        error_msg_lower = error_msg.lower()
        assert "1.x.y" in error_msg
        # Check for either the core ModelSemVer.parse() message format
        # or the validate_version_lenient() message format (case-insensitive)
        assert "invalid" in error_msg_lower and "version" in error_msg_lower

        # Registry should be empty
        assert len(policy_registry) == 0

    def test_empty_version_string_raises_error(
        self, policy_registry: PolicyRegistry
    ) -> None:
        """Test that empty version strings raise ProtocolConfigurationError at registration."""
        # Attempt to register with empty version
        with pytest.raises(ProtocolConfigurationError) as exc_info:
            policy_registry.register_policy(
                policy_id="empty-version",
                policy_class=MockPolicyV1,  # type: ignore[arg-type]
                policy_type=EnumPolicyType.ORCHESTRATOR,
                version="",  # Empty version
            )  # type: ignore[arg-type]

        # Verify error message mentions empty/whitespace
        error_msg = str(exc_info.value).lower()
        assert "empty" in error_msg or "whitespace" in error_msg

        # Registry should be empty
        assert len(policy_registry) == 0

    def test_version_with_too_many_parts_raises_error(
        self, policy_registry: PolicyRegistry
    ) -> None:
        """Test that versions with more than 3 parts raise error."""
        # Attempt to register with too many version parts
        with pytest.raises(ProtocolConfigurationError) as exc_info:
            policy_registry.register_policy(
                policy_id="too-many-parts",
                policy_class=MockPolicyV1,  # type: ignore[arg-type]
                policy_type=EnumPolicyType.ORCHESTRATOR,
                version="1.2.3.4",  # Four parts (invalid)
            )  # type: ignore[arg-type]

        # Verify error mentions format and the invalid version
        error_msg = str(exc_info.value)
        error_msg_lower = error_msg.lower()
        assert "1.2.3.4" in error_msg
        # Case-insensitive check for robustness against minor error message changes
        assert "invalid" in error_msg_lower and "version" in error_msg_lower

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
        """Test semver sorting with prerelease versions.

        Note: omnibase_core's ModelSemVer does NOT compare prerelease fields.
        Versions "1.0.0-alpha" and "1.0.0" are considered EQUAL for comparison.
        When versions are equal, the last registered wins (overwrites).
        """
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

        # Both versions exist in registry (different version strings)
        assert policy_registry.is_registered("prerelease-policy", version="1.0.0-alpha")
        assert policy_registry.is_registered("prerelease-policy", version="1.0.0")

        # Get without version returns latest based on semver comparison
        # Since omnibase_core's ModelSemVer ignores prerelease, both versions
        # are considered equal (major=1, minor=0, patch=0). The max() function
        # returns the first equal element, which depends on iteration order.
        latest_cls = policy_registry.get("prerelease-policy")
        # Verify we get one of the registered policies (either is valid when equal)
        assert latest_cls in (MockPolicyV1, MockPolicyV2)

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
        """Test that invalid policy type string raises ProtocolConfigurationError.

        The PolicyRegistry catches ValidationError and converts it to
        ProtocolConfigurationError for consistent error handling across
        all validation failures.
        """
        with pytest.raises(ProtocolConfigurationError) as exc_info:
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
        """Test async policy workflow with allow_async=True."""
        # This should succeed with the flag
        policy_registry.register_policy(
            policy_id="async-workflow",
            policy_class=MockAsyncPolicy,  # type: ignore[arg-type]
            policy_type=EnumPolicyType.REDUCER,
            version="1.0.0",
            allow_async=True,
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

    Note: Tests use _reset_semver_cache() to ensure clean state between tests
    since the cache is now lazily initialized as a class-level singleton.
    """

    def test_parse_semver_returns_consistent_results(self) -> None:
        """Test that _parse_semver returns consistent results for same input."""
        # Reset cache to ensure clean state
        PolicyRegistry._reset_semver_cache()

        # Parse same version multiple times
        result1 = PolicyRegistry._parse_semver("1.2.3")
        result2 = PolicyRegistry._parse_semver("1.2.3")
        result3 = PolicyRegistry._parse_semver("1.2.3")

        # All should return identical ModelSemVer instances
        assert result1 == result2 == result3
        assert result1 == ModelSemVer(major=1, minor=2, patch=3)

    def test_parse_semver_cache_info_shows_hits(self) -> None:
        """Test that cache info shows hits for repeated parses."""
        # Reset cache to ensure clean state
        PolicyRegistry._reset_semver_cache()

        # Get the parser (initializes the cache)
        PolicyRegistry._get_semver_parser()
        initial_info = PolicyRegistry._get_semver_cache_info()
        assert initial_info is not None
        assert initial_info.hits == 0
        assert initial_info.misses == 0

        # First parse - should be a cache miss
        PolicyRegistry._parse_semver("1.0.0")
        info_after_first = PolicyRegistry._get_semver_cache_info()
        assert info_after_first is not None
        assert info_after_first.misses == 1
        assert info_after_first.hits == 0

        # Second parse of same version - should be a cache hit
        PolicyRegistry._parse_semver("1.0.0")
        info_after_second = PolicyRegistry._get_semver_cache_info()
        assert info_after_second is not None
        assert info_after_second.misses == 1
        assert info_after_second.hits == 1

        # Third parse - another hit
        PolicyRegistry._parse_semver("1.0.0")
        info_after_third = PolicyRegistry._get_semver_cache_info()
        assert info_after_third is not None
        assert info_after_third.misses == 1
        assert info_after_third.hits == 2

    def test_parse_semver_different_versions_cause_misses(self) -> None:
        """Test that different version strings cause cache misses."""
        # Reset cache to ensure clean state
        PolicyRegistry._reset_semver_cache()

        # Parse different versions
        PolicyRegistry._parse_semver("1.0.0")
        PolicyRegistry._parse_semver("2.0.0")
        PolicyRegistry._parse_semver("3.0.0")

        info = PolicyRegistry._get_semver_cache_info()
        assert info is not None
        assert info.misses == 3
        assert info.hits == 0
        assert info.currsize == 3  # 3 entries cached

    def test_parse_semver_cache_improves_get_performance(
        self, policy_registry: PolicyRegistry
    ) -> None:
        """Test that caching improves performance for repeated get() calls."""
        # Reset cache to ensure clean state
        PolicyRegistry._reset_semver_cache()

        # Register multiple versions of same policy
        for i in range(10):
            policy_registry.register_policy(
                policy_id="perf-test",
                policy_class=MockPolicyV1,  # type: ignore[arg-type]
                policy_type=EnumPolicyType.ORCHESTRATOR,
                version=f"{i}.0.0",
            )  # type: ignore[arg-type]

        # First get() will parse all versions (cold cache)
        _ = policy_registry.get("perf-test")  # Gets latest version
        first_info = PolicyRegistry._get_semver_cache_info()
        assert first_info is not None
        first_misses = first_info.misses

        # Second get() should hit cache (warm cache)
        _ = policy_registry.get("perf-test")
        second_info = PolicyRegistry._get_semver_cache_info()
        assert second_info is not None
        second_misses = second_info.misses

        # Cache should have been hit (no new misses)
        assert second_misses == first_misses

    def test_parse_semver_cache_handles_prerelease_versions(self) -> None:
        """Test that cache correctly handles prerelease version strings.

        Note: omnibase_core's ModelSemVer does NOT compare prerelease fields.
        All versions with same major.minor.patch compare equal regardless of prerelease.
        The cache still creates separate entries for different input strings.
        """
        # Reset cache to ensure clean state
        PolicyRegistry._reset_semver_cache()

        # Parse prerelease versions
        result1 = PolicyRegistry._parse_semver("1.0.0-alpha")
        result2 = PolicyRegistry._parse_semver("1.0.0-beta")
        result3 = PolicyRegistry._parse_semver("1.0.0")

        # Note: omnibase_core's ModelSemVer ignores prerelease for comparison
        # All three versions have same major.minor.patch, so they compare equal
        # But cache still creates separate entries for different input strings
        info = PolicyRegistry._get_semver_cache_info()
        assert info is not None
        assert info.currsize == 3  # Three distinct input strings

        # Repeat parse should hit cache
        result1_repeat = PolicyRegistry._parse_semver("1.0.0-alpha")
        # Cache returns same parsed result
        assert result1_repeat == result1
        info_after = PolicyRegistry._get_semver_cache_info()
        assert info_after is not None
        assert info_after.hits == 1

    def test_parse_semver_cache_size_limit(self) -> None:
        """Test that cache respects maxsize=128 limit (default)."""
        # Reset cache to ensure clean state
        PolicyRegistry._reset_semver_cache()

        # Parse 150 unique versions (exceeds default maxsize=128)
        for i in range(150):
            PolicyRegistry._parse_semver(f"{i}.0.0")

        info = PolicyRegistry._get_semver_cache_info()
        assert info is not None
        # Cache size should not exceed maxsize
        assert info.currsize <= 128

    def test_parse_semver_cache_lru_eviction(self) -> None:
        """Test that LRU eviction works correctly."""
        # Reset cache to ensure clean state
        PolicyRegistry._reset_semver_cache()

        # Fill cache to capacity with versions 0-127
        for i in range(128):
            PolicyRegistry._parse_semver(f"{i}.0.0")

        # Access version "0.0.0" to make it most recently used
        PolicyRegistry._parse_semver("0.0.0")

        # Add new version to trigger eviction (should evict "1.0.0", not "0.0.0")
        PolicyRegistry._parse_semver("999.0.0")

        # "0.0.0" should still be in cache (was recently used)
        PolicyRegistry._parse_semver("0.0.0")
        info = PolicyRegistry._get_semver_cache_info()
        assert info is not None
        # Last access to "0.0.0" should be a hit
        assert info.hits > 0

    def test_reset_semver_cache_clears_state(self) -> None:
        """Test that _reset_semver_cache() clears cache state."""
        # Parse some versions
        PolicyRegistry._parse_semver("1.0.0")
        PolicyRegistry._parse_semver("2.0.0")
        info_before = PolicyRegistry._get_semver_cache_info()
        assert info_before is not None
        assert info_before.currsize > 0

        # Reset cache
        PolicyRegistry._reset_semver_cache()

        # After reset, the cache should be None (will be reinitialized on next use)
        assert PolicyRegistry._semver_cache is None
        assert PolicyRegistry._semver_cache_inner is None

        # Next parse initializes a fresh cache
        PolicyRegistry._parse_semver("3.0.0")
        info_after = PolicyRegistry._get_semver_cache_info()
        assert info_after is not None

        # New cache should have only the one entry we just parsed
        assert info_after.currsize == 1
        assert info_after.hits == 0
        assert info_after.misses == 1


class TestPolicyRegistrySemverCacheConfiguration:
    """Tests for configurable semver cache size.

    Validates that the cache size can be configured for large deployments.
    """

    def test_configure_semver_cache_before_use(self) -> None:
        """Test configuring cache size before first use."""
        # Reset cache to allow configuration
        PolicyRegistry._reset_semver_cache()
        original_size = PolicyRegistry.SEMVER_CACHE_SIZE

        try:
            # Configure a smaller cache size
            PolicyRegistry.configure_semver_cache(maxsize=64)
            assert PolicyRegistry.SEMVER_CACHE_SIZE == 64

            # Use the parser (initializes with new size)
            PolicyRegistry._parse_semver("1.0.0")

            # Verify cache was created with the configured size
            # The maxsize is stored in cache_parameters() for newer Python
            # or we can verify by filling it
            for i in range(100):
                PolicyRegistry._parse_semver(f"{i}.0.0")
            info = PolicyRegistry._get_semver_cache_info()
            assert info is not None
            # With maxsize=64, currsize should be <= 64
            assert info.currsize <= 64
        finally:
            # Reset to original state
            PolicyRegistry._reset_semver_cache()
            PolicyRegistry.SEMVER_CACHE_SIZE = original_size

    def test_configure_semver_cache_after_use_raises_error(self) -> None:
        """Test that configuring cache after first use raises RuntimeError."""
        # Reset cache to start fresh
        PolicyRegistry._reset_semver_cache()

        # Use the parser (initializes the cache)
        PolicyRegistry._parse_semver("1.0.0")

        # Attempt to reconfigure should fail
        with pytest.raises(RuntimeError) as exc_info:
            PolicyRegistry.configure_semver_cache(maxsize=256)

        assert "Cannot reconfigure semver cache after first use" in str(exc_info.value)

        # Reset for other tests
        PolicyRegistry._reset_semver_cache()

    def test_semver_cache_size_via_class_attribute(self) -> None:
        """Test setting cache size via class attribute before use."""
        # Reset cache to allow reconfiguration
        PolicyRegistry._reset_semver_cache()
        original_size = PolicyRegistry.SEMVER_CACHE_SIZE

        try:
            # Set via class attribute (alternative to configure_semver_cache)
            PolicyRegistry.SEMVER_CACHE_SIZE = 32

            # Parse enough versions to exceed the small cache
            for i in range(50):
                PolicyRegistry._parse_semver(f"{i}.0.0")

            info = PolicyRegistry._get_semver_cache_info()
            assert info is not None
            # With maxsize=32, currsize should be <= 32
            assert info.currsize <= 32
        finally:
            # Reset to original state
            PolicyRegistry._reset_semver_cache()
            PolicyRegistry.SEMVER_CACHE_SIZE = original_size

    def test_default_cache_size_is_128(self) -> None:
        """Test that default cache size is 128."""
        # Reset to ensure we're checking the class default
        PolicyRegistry._reset_semver_cache()

        # The class-level default should be 128
        # Note: We need to be careful as tests may have modified this
        # Just verify the documented default
        assert PolicyRegistry.SEMVER_CACHE_SIZE >= 64  # Reasonable minimum
        assert PolicyRegistry.SEMVER_CACHE_SIZE <= 512  # Reasonable maximum

    def test_cache_reset_allows_reconfiguration(self) -> None:
        """Test that _reset_semver_cache allows reconfiguration."""
        original_size = PolicyRegistry.SEMVER_CACHE_SIZE

        try:
            # Reset and configure
            PolicyRegistry._reset_semver_cache()
            PolicyRegistry.configure_semver_cache(maxsize=100)

            # Use the cache
            PolicyRegistry._parse_semver("1.0.0")

            # Reset again
            PolicyRegistry._reset_semver_cache()

            # Should be able to reconfigure now
            PolicyRegistry.configure_semver_cache(maxsize=200)
            assert PolicyRegistry.SEMVER_CACHE_SIZE == 200
        finally:
            # Cleanup
            PolicyRegistry._reset_semver_cache()
            PolicyRegistry.SEMVER_CACHE_SIZE = original_size


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
        with pytest.raises(ProtocolConfigurationError) as exc_info:
            policy_registry.register_policy(
                policy_id="invalid-version",
                policy_class=MockSyncPolicy,  # type: ignore[arg-type]
                policy_type=EnumPolicyType.ORCHESTRATOR,
                version="",
            )
        # Empty string triggers "cannot be empty or whitespace-only" error
        error_msg = str(exc_info.value).lower()
        assert "empty" in error_msg or "whitespace" in error_msg

    def test_invalid_version_format_empty_prerelease_suffix(
        self, policy_registry: PolicyRegistry
    ) -> None:
        """Test that dash with no prerelease suffix raises ProtocolConfigurationError.

        Semver prerelease suffix must be non-empty when dash is present:
        - "1.2.3-" is INVALID (dash with no suffix)
        - "1.2.3-alpha" is VALID
        - "1.2.3" is VALID (no prerelease)
        """
        with pytest.raises(ProtocolConfigurationError) as exc_info:
            policy_registry.register_policy(
                policy_id="empty-prerelease",
                policy_class=MockSyncPolicy,  # type: ignore[arg-type]
                policy_type=EnumPolicyType.ORCHESTRATOR,
                version="1.2.3-",  # Invalid: dash with empty prerelease
            )
        error_msg = str(exc_info.value).lower()
        # Policy registry validates trailing dash before calling ModelSemVer.parse()
        assert "prerelease" in error_msg or "empty" in error_msg

        # Verify registry is empty (registration failed)
        assert len(policy_registry) == 0

    def test_valid_prerelease_versions_accepted(
        self, policy_registry: PolicyRegistry
    ) -> None:
        """Test that valid prerelease versions are still accepted."""
        # Valid prerelease formats should work
        policy_registry.register_policy(
            policy_id="alpha-version",
            policy_class=MockSyncPolicy,  # type: ignore[arg-type]
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="1.0.0-alpha",
        )
        policy_registry.register_policy(
            policy_id="beta-version",
            policy_class=MockSyncPolicy,  # type: ignore[arg-type]
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="2.0.0-beta.1",
        )
        policy_registry.register_policy(
            policy_id="rc-version",
            policy_class=MockSyncPolicy,  # type: ignore[arg-type]
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="3.0.0-rc.1.2.3",
        )
        # Version without prerelease should also work
        policy_registry.register_policy(
            policy_id="release-version",
            policy_class=MockSyncPolicy,  # type: ignore[arg-type]
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="4.0.0",
        )

        assert policy_registry.is_registered("alpha-version", version="1.0.0-alpha")
        assert policy_registry.is_registered("beta-version", version="2.0.0-beta.1")
        assert policy_registry.is_registered("rc-version", version="3.0.0-rc.1.2.3")
        assert policy_registry.is_registered("release-version", version="4.0.0")

    def test_empty_prerelease_suffix_various_formats(
        self, policy_registry: PolicyRegistry
    ) -> None:
        """Test empty prerelease suffix is rejected for various version formats."""
        invalid_versions = [
            "1-",  # major only with trailing dash
            "1.2-",  # major.minor with trailing dash
            "1.2.3-",  # full semver with trailing dash
            "0.0.0-",  # zero version with trailing dash
            "10.20.30-",  # larger numbers with trailing dash
        ]

        for version in invalid_versions:
            policy_registry.clear()  # Reset for each test
            with pytest.raises(ProtocolConfigurationError) as exc_info:
                policy_registry.register_policy(
                    policy_id=f"test-{version}",
                    policy_class=MockSyncPolicy,  # type: ignore[arg-type]
                    policy_type=EnumPolicyType.ORCHESTRATOR,
                    version=version,
                )
            error_msg = str(exc_info.value).lower()
            # Policy registry validates trailing dash before calling ModelSemVer.parse()
            assert "prerelease" in error_msg or "empty" in error_msg, (
                f"Expected error for version '{version}', got: {exc_info.value}"
            )

    def test_invalid_version_format_non_numeric(
        self, policy_registry: PolicyRegistry
    ) -> None:
        """Test that non-numeric version components raise ProtocolConfigurationError."""
        with pytest.raises(ProtocolConfigurationError) as exc_info:
            policy_registry.register_policy(
                policy_id="invalid-version",
                policy_class=MockSyncPolicy,  # type: ignore[arg-type]
                policy_type=EnumPolicyType.ORCHESTRATOR,
                version="abc.def.ghi",
            )
        # ModelSemVer.parse() rejects non-numeric components
        error_msg = str(exc_info.value).lower()
        assert "invalid" in error_msg or "format" in error_msg

    def test_invalid_version_format_negative_numbers(
        self, policy_registry: PolicyRegistry
    ) -> None:
        """Test that negative version numbers raise ProtocolConfigurationError."""
        with pytest.raises(ProtocolConfigurationError) as exc_info:
            policy_registry.register_policy(
                policy_id="invalid-version",
                policy_class=MockSyncPolicy,  # type: ignore[arg-type]
                policy_type=EnumPolicyType.ORCHESTRATOR,
                version="1.-1.0",
            )
        # Negative numbers are rejected by ModelSemVer.parse() regex
        error_msg = str(exc_info.value).lower()
        assert "invalid" in error_msg or "format" in error_msg

    def test_invalid_version_format_too_many_parts(
        self, policy_registry: PolicyRegistry
    ) -> None:
        """Test that version with too many parts raises ProtocolConfigurationError."""
        with pytest.raises(ProtocolConfigurationError) as exc_info:
            policy_registry.register_policy(
                policy_id="invalid-version",
                policy_class=MockSyncPolicy,  # type: ignore[arg-type]
                policy_type=EnumPolicyType.ORCHESTRATOR,
                version="1.2.3.4",
            )
        error_msg = str(exc_info.value).lower()
        # Case-insensitive check for robustness against minor error message changes
        assert "invalid" in error_msg and "version" in error_msg

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

    def test_version_with_leading_spaces(self, policy_registry: PolicyRegistry) -> None:
        """Test that leading spaces are trimmed from version strings."""
        policy_registry.register_policy(
            policy_id="leading-space",
            policy_class=MockSyncPolicy,  # type: ignore[arg-type]
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="  1.2.3",
        )
        # The version should be stored as trimmed "1.2.3"
        assert policy_registry.is_registered("leading-space")
        # Lookup with trimmed version should work
        policy_cls = policy_registry.get("leading-space", version="1.2.3")
        assert policy_cls is MockSyncPolicy

    def test_version_with_trailing_spaces(
        self, policy_registry: PolicyRegistry
    ) -> None:
        """Test that trailing spaces are trimmed from version strings."""
        policy_registry.register_policy(
            policy_id="trailing-space",
            policy_class=MockSyncPolicy,  # type: ignore[arg-type]
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="1.2.3  ",
        )
        assert policy_registry.is_registered("trailing-space")
        policy_cls = policy_registry.get("trailing-space", version="1.2.3")
        assert policy_cls is MockSyncPolicy

    def test_version_with_surrounding_spaces(
        self, policy_registry: PolicyRegistry
    ) -> None:
        """Test that surrounding spaces are trimmed from version strings."""
        policy_registry.register_policy(
            policy_id="surrounding-space",
            policy_class=MockSyncPolicy,  # type: ignore[arg-type]
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version=" 1.2.3 ",
        )
        assert policy_registry.is_registered("surrounding-space")
        policy_cls = policy_registry.get("surrounding-space", version="1.2.3")
        assert policy_cls is MockSyncPolicy

    def test_version_with_newline(self, policy_registry: PolicyRegistry) -> None:
        """Test that newlines are trimmed from version strings."""
        policy_registry.register_policy(
            policy_id="newline-version",
            policy_class=MockSyncPolicy,  # type: ignore[arg-type]
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="1.2.3\n",
        )
        assert policy_registry.is_registered("newline-version")
        policy_cls = policy_registry.get("newline-version", version="1.2.3")
        assert policy_cls is MockSyncPolicy

    def test_version_with_leading_newline(
        self, policy_registry: PolicyRegistry
    ) -> None:
        """Test that leading newlines are trimmed from version strings."""
        policy_registry.register_policy(
            policy_id="leading-newline",
            policy_class=MockSyncPolicy,  # type: ignore[arg-type]
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="\n1.2.3",
        )
        assert policy_registry.is_registered("leading-newline")
        policy_cls = policy_registry.get("leading-newline", version="1.2.3")
        assert policy_cls is MockSyncPolicy

    def test_version_with_tabs(self, policy_registry: PolicyRegistry) -> None:
        """Test that tabs are trimmed from version strings."""
        policy_registry.register_policy(
            policy_id="tab-version",
            policy_class=MockSyncPolicy,  # type: ignore[arg-type]
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="\t1.2.3\t",
        )
        assert policy_registry.is_registered("tab-version")
        policy_cls = policy_registry.get("tab-version", version="1.2.3")
        assert policy_cls is MockSyncPolicy

    def test_version_with_mixed_whitespace(
        self, policy_registry: PolicyRegistry
    ) -> None:
        """Test that mixed whitespace is trimmed from version strings."""
        policy_registry.register_policy(
            policy_id="mixed-whitespace",
            policy_class=MockSyncPolicy,  # type: ignore[arg-type]
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version=" \t\n1.2.3\n\t ",
        )
        assert policy_registry.is_registered("mixed-whitespace")
        policy_cls = policy_registry.get("mixed-whitespace", version="1.2.3")
        assert policy_cls is MockSyncPolicy

    def test_prerelease_version_with_whitespace(
        self, policy_registry: PolicyRegistry
    ) -> None:
        """Test that whitespace is trimmed from prerelease version strings."""
        policy_registry.register_policy(
            policy_id="prerelease-whitespace",
            policy_class=MockSyncPolicy,  # type: ignore[arg-type]
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="  1.2.3-alpha  ",
        )
        assert policy_registry.is_registered("prerelease-whitespace")
        policy_cls = policy_registry.get("prerelease-whitespace", version="1.2.3-alpha")
        assert policy_cls is MockSyncPolicy

    def test_whitespace_only_version_raises_error(
        self, policy_registry: PolicyRegistry
    ) -> None:
        """Test that whitespace-only version strings raise ProtocolConfigurationError."""
        with pytest.raises(ProtocolConfigurationError) as exc_info:
            policy_registry.register_policy(
                policy_id="whitespace-only",
                policy_class=MockSyncPolicy,  # type: ignore[arg-type]
                policy_type=EnumPolicyType.ORCHESTRATOR,
                version="   ",
            )
        # Whitespace-only version triggers "cannot be empty or whitespace-only" error
        error_msg = str(exc_info.value).lower()
        assert "empty" in error_msg or "whitespace" in error_msg

    def test_parse_semver_whitespace_trimming(self) -> None:
        """Test _parse_semver directly with whitespace inputs."""
        # Reset cache to ensure clean state
        PolicyRegistry._reset_semver_cache()

        # Test various whitespace scenarios
        result1 = PolicyRegistry._parse_semver(" 1.2.3 ")
        result2 = PolicyRegistry._parse_semver("1.2.3\n")
        result3 = PolicyRegistry._parse_semver("\t1.2.3\t")
        result4 = PolicyRegistry._parse_semver("1.2.3")

        # All should parse to the same ModelSemVer result
        assert result1 == result4
        assert result2 == result4
        assert result3 == result4
        assert result1 == ModelSemVer(major=1, minor=2, patch=3)

    def test_whitespace_versions_with_latest_selection(
        self, policy_registry: PolicyRegistry
    ) -> None:
        """Test that whitespace versions work correctly with latest version selection."""
        policy_registry.register_policy(
            policy_id="whitespace-latest",
            policy_class=MockPolicyV1,  # type: ignore[arg-type]
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version=" 1.0.0 ",
        )
        policy_registry.register_policy(
            policy_id="whitespace-latest",
            policy_class=MockPolicyV2,  # type: ignore[arg-type]
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version=" 2.0.0 ",
        )

        # Get latest should return V2 (2.0.0 > 1.0.0)
        latest_cls = policy_registry.get("whitespace-latest")
        assert latest_cls is MockPolicyV2

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


# =============================================================================
# TestModelPolicyKeyHashUniqueness
# =============================================================================


class TestModelPolicyKeyHashUniqueness:
    """Test ModelPolicyKey hash uniqueness for edge cases.

    ModelPolicyKey is used as a dictionary key in PolicyRegistry.
    These tests verify that hash uniqueness doesn't collide for various
    edge cases, ensuring correct dictionary behavior.
    """

    def test_hash_uniqueness_similar_ids(self) -> None:
        """Similar policy_ids should have different hashes."""
        keys = [
            ModelPolicyKey(
                policy_id="retry", policy_type="orchestrator", version="1.0.0"
            ),
            ModelPolicyKey(
                policy_id="retry1", policy_type="orchestrator", version="1.0.0"
            ),
            ModelPolicyKey(
                policy_id="1retry", policy_type="orchestrator", version="1.0.0"
            ),
            ModelPolicyKey(
                policy_id="retr", policy_type="orchestrator", version="1.0.0"
            ),
        ]
        hashes = {hash(k) for k in keys}
        assert len(hashes) == len(keys), "Hash collision detected for similar IDs"

    def test_hash_uniqueness_type_differs(self) -> None:
        """Same policy_id with different types should have different hashes."""
        key1 = ModelPolicyKey(
            policy_id="test", policy_type="orchestrator", version="1.0.0"
        )
        key2 = ModelPolicyKey(policy_id="test", policy_type="reducer", version="1.0.0")
        assert hash(key1) != hash(key2)

    def test_hash_uniqueness_version_differs(self) -> None:
        """Same policy_id with different versions should have different hashes."""
        keys = [
            ModelPolicyKey(
                policy_id="test", policy_type="orchestrator", version="1.0.0"
            ),
            ModelPolicyKey(
                policy_id="test", policy_type="orchestrator", version="1.0.1"
            ),
            ModelPolicyKey(
                policy_id="test", policy_type="orchestrator", version="2.0.0"
            ),
        ]
        hashes = {hash(k) for k in keys}
        assert len(hashes) == len(keys)

    def test_hash_stability(self) -> None:
        """Same key should always produce same hash."""
        key = ModelPolicyKey(
            policy_id="stable", policy_type="orchestrator", version="1.0.0"
        )
        hash1 = hash(key)
        hash2 = hash(key)
        key_copy = ModelPolicyKey(
            policy_id="stable", policy_type="orchestrator", version="1.0.0"
        )
        hash3 = hash(key_copy)

        assert hash1 == hash2 == hash3

    def test_hash_uniqueness_with_special_characters(self) -> None:
        """Policy IDs with special characters should have unique hashes."""
        keys = [
            ModelPolicyKey(
                policy_id="test-policy", policy_type="orchestrator", version="1.0.0"
            ),
            ModelPolicyKey(
                policy_id="test_policy", policy_type="orchestrator", version="1.0.0"
            ),
            ModelPolicyKey(
                policy_id="test.policy", policy_type="orchestrator", version="1.0.0"
            ),
            ModelPolicyKey(
                policy_id="test:policy", policy_type="orchestrator", version="1.0.0"
            ),
        ]
        hashes = {hash(k) for k in keys}
        assert len(hashes) == len(keys)

    def test_hash_uniqueness_prerelease_versions(self) -> None:
        """Prerelease versions should have unique hashes."""
        keys = [
            ModelPolicyKey(
                policy_id="test", policy_type="orchestrator", version="1.0.0"
            ),
            ModelPolicyKey(
                policy_id="test", policy_type="orchestrator", version="1.0.0-alpha"
            ),
            ModelPolicyKey(
                policy_id="test", policy_type="orchestrator", version="1.0.0-beta"
            ),
            ModelPolicyKey(
                policy_id="test", policy_type="orchestrator", version="1.0.0-rc.1"
            ),
        ]
        hashes = {hash(k) for k in keys}
        assert len(hashes) == len(keys)

    def test_dict_key_usage(self) -> None:
        """ModelPolicyKey should work correctly as dict key."""
        d: dict[ModelPolicyKey, str] = {}

        key1 = ModelPolicyKey(
            policy_id="a", policy_type="orchestrator", version="1.0.0"
        )
        key2 = ModelPolicyKey(
            policy_id="a", policy_type="orchestrator", version="1.0.0"
        )  # same
        key3 = ModelPolicyKey(
            policy_id="b", policy_type="orchestrator", version="1.0.0"
        )  # different

        d[key1] = "value1"
        d[key3] = "value3"

        # key2 should find same value as key1 (they're equal)
        assert d[key2] == "value1"
        assert len(d) == 2

    def test_large_scale_hash_distribution(self) -> None:
        """Test hash distribution with many keys."""
        keys = [
            ModelPolicyKey(
                policy_id=f"policy_{i}",
                policy_type="orchestrator" if i % 2 == 0 else "reducer",
                version=f"{i % 10}.{i % 5}.{i % 3}",
            )
            for i in range(1000)
        ]
        hashes = {hash(k) for k in keys}
        # Allow some collisions but expect >99% unique
        assert len(hashes) > 990, f"Too many collisions: {1000 - len(hashes)}"


# =============================================================================
# TestPolicyRegistryVersionNormalizationIntegration
# =============================================================================


class TestPolicyRegistryVersionNormalizationIntegration:
    """Integration tests for version normalization edge cases.

    These tests verify that the PolicyRegistry correctly normalizes version
    strings during registration and lookup, ensuring that partial versions
    and whitespace-trimmed versions work correctly with ModelPolicyKey.

    Required by PR #92 review feedback.
    """

    # =========================================================================
    # Partial Version Normalization Tests
    # =========================================================================

    def test_partial_version_major_only_registers_and_retrieves(
        self, policy_registry: PolicyRegistry
    ) -> None:
        """Test that major-only version '1' normalizes to '1.0.0' for storage.

        Verifies:
        - Registration with "1" succeeds
        - Lookup with "1" finds the policy
        - Lookup with "1.0.0" also finds the policy (normalized match)

        Note: This test intentionally uses non-normalized versions ('1') to verify
        normalization behavior.
        """
        policy_registry.register_policy(
            policy_id="partial-major",
            policy_class=MockSyncPolicy,  # type: ignore[arg-type]
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="1",
        )

        # Should be registered with partial version
        assert policy_registry.is_registered("partial-major", version="1")

        # Should also match normalized version
        assert policy_registry.is_registered("partial-major", version="1.0.0")

        # get() with partial version should work
        policy_cls = policy_registry.get("partial-major", version="1")
        assert policy_cls is MockSyncPolicy

        # get() with normalized version should work
        policy_cls_normalized = policy_registry.get("partial-major", version="1.0.0")
        assert policy_cls_normalized is MockSyncPolicy

    def test_partial_version_major_minor_registers_and_retrieves(
        self, policy_registry: PolicyRegistry
    ) -> None:
        """Test that major.minor version '1.2' normalizes to '1.2.0' for storage.

        Verifies:
        - Registration with "1.2" succeeds
        - Lookup with "1.2" finds the policy
        - Lookup with "1.2.0" also finds the policy (normalized match)

        Note: This test intentionally uses non-normalized versions ('1.2') to verify
        normalization behavior.
        """
        policy_registry.register_policy(
            policy_id="partial-major-minor",
            policy_class=MockSyncPolicy,  # type: ignore[arg-type]
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="1.2",
        )

        # Should be registered with partial version
        assert policy_registry.is_registered("partial-major-minor", version="1.2")

        # Should also match normalized version
        assert policy_registry.is_registered("partial-major-minor", version="1.2.0")

        # get() should work with both formats
        policy_cls = policy_registry.get("partial-major-minor", version="1.2")
        assert policy_cls is MockSyncPolicy

        policy_cls_normalized = policy_registry.get(
            "partial-major-minor", version="1.2.0"
        )
        assert policy_cls_normalized is MockSyncPolicy

    def test_partial_version_zero_normalizes_correctly(
        self, policy_registry: PolicyRegistry
    ) -> None:
        """Test edge case: version '0' normalizes to '0.0.0'.

        This is an important edge case as zero versions are valid in semver.

        Note: This test intentionally uses non-normalized versions ('0') to verify
        normalization behavior.
        """
        policy_registry.register_policy(
            policy_id="zero-version",
            policy_class=MockSyncPolicy,  # type: ignore[arg-type]
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="0",
        )

        # Should be registered and retrievable
        assert policy_registry.is_registered("zero-version", version="0")
        assert policy_registry.is_registered("zero-version", version="0.0.0")

        policy_cls = policy_registry.get("zero-version", version="0")
        assert policy_cls is MockSyncPolicy

        policy_cls_normalized = policy_registry.get("zero-version", version="0.0.0")
        assert policy_cls_normalized is MockSyncPolicy

    def test_partial_version_zero_major_minor_normalizes_correctly(
        self, policy_registry: PolicyRegistry
    ) -> None:
        """Test edge case: version '0.1' normalizes to '0.1.0'.

        Note: This test intentionally uses non-normalized versions ('0.1') to verify
        normalization behavior.
        """
        policy_registry.register_policy(
            policy_id="zero-minor-version",
            policy_class=MockSyncPolicy,  # type: ignore[arg-type]
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="0.1",
        )

        assert policy_registry.is_registered("zero-minor-version", version="0.1")
        assert policy_registry.is_registered("zero-minor-version", version="0.1.0")

        policy_cls = policy_registry.get("zero-minor-version", version="0.1")
        assert policy_cls is MockSyncPolicy

    # =========================================================================
    # Whitespace Trimming Integration Tests
    # =========================================================================

    def test_whitespace_trimming_space_padded_version(
        self, policy_registry: PolicyRegistry
    ) -> None:
        """Test that ' 1.0.0 ' is trimmed and matches '1.0.0'.

        Verifies full integration: register with whitespace, lookup without.

        Note: This test intentionally uses non-normalized versions (' 1.0.0 ') to verify
        normalization behavior.
        """
        policy_registry.register_policy(
            policy_id="space-padded",
            policy_class=MockSyncPolicy,  # type: ignore[arg-type]
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version=" 1.0.0 ",
        )

        # Lookup without whitespace should work
        assert policy_registry.is_registered("space-padded", version="1.0.0")
        policy_cls = policy_registry.get("space-padded", version="1.0.0")
        assert policy_cls is MockSyncPolicy

        # Lookup with same whitespace should also work (normalized)
        assert policy_registry.is_registered("space-padded", version=" 1.0.0 ")
        policy_cls_with_spaces = policy_registry.get("space-padded", version=" 1.0.0 ")
        assert policy_cls_with_spaces is MockSyncPolicy

    def test_whitespace_trimming_tab_newline_version(
        self, policy_registry: PolicyRegistry
    ) -> None:
        """Test that tab and newline characters are trimmed from versions.

        Verifies: '\\t1.0.0\\n' normalizes to '1.0.0'.

        Note: This test intentionally uses non-normalized versions with whitespace to
        verify normalization behavior.
        """
        policy_registry.register_policy(
            policy_id="tab-newline",
            policy_class=MockSyncPolicy,  # type: ignore[arg-type]
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="\t1.0.0\n",
        )

        # Lookup without whitespace
        assert policy_registry.is_registered("tab-newline", version="1.0.0")
        policy_cls = policy_registry.get("tab-newline", version="1.0.0")
        assert policy_cls is MockSyncPolicy

        # Lookup with different whitespace patterns
        assert policy_registry.is_registered("tab-newline", version=" 1.0.0 ")

    def test_whitespace_with_partial_version_combined(
        self, policy_registry: PolicyRegistry
    ) -> None:
        """Test combined whitespace trimming and partial version normalization.

        Verifies: ' 2 ' normalizes to '2.0.0'.

        Note: This test intentionally uses non-normalized versions (' 2 ') to verify
        normalization behavior.
        """
        policy_registry.register_policy(
            policy_id="whitespace-partial",
            policy_class=MockSyncPolicy,  # type: ignore[arg-type]
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version=" 2 ",
        )

        # Should match all normalized forms
        assert policy_registry.is_registered("whitespace-partial", version="2")
        assert policy_registry.is_registered("whitespace-partial", version="2.0.0")
        assert policy_registry.is_registered("whitespace-partial", version=" 2 ")
        assert policy_registry.is_registered("whitespace-partial", version=" 2.0.0 ")

        policy_cls = policy_registry.get("whitespace-partial", version="2.0.0")
        assert policy_cls is MockSyncPolicy

    # =========================================================================
    # ModelPolicyKey Lookup Equivalence Tests
    # =========================================================================

    def test_model_policy_key_equivalence_partial_versions(
        self, policy_registry: PolicyRegistry
    ) -> None:
        """Test that ModelPolicyKey treats normalized versions as equivalent.

        This tests the core of ModelPolicyKey normalization - that registering
        with '1' and looking up with '1.0.0' uses the same key internally.

        Note: This test intentionally uses non-normalized versions ('1') to verify
        normalization behavior.
        """
        # Register with partial version
        policy_registry.register_policy(
            policy_id="key-equiv-partial",
            policy_class=MockPolicyV1,  # type: ignore[arg-type]
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="1",
        )

        # Verify list_versions returns the stored version
        versions = policy_registry.list_versions("key-equiv-partial")
        # The registry normalizes during storage
        assert len(versions) == 1

        # Both partial and full versions should work for retrieval
        policy_cls_partial = policy_registry.get("key-equiv-partial", version="1")
        policy_cls_full = policy_registry.get("key-equiv-partial", version="1.0.0")
        assert policy_cls_partial is policy_cls_full is MockPolicyV1

    def test_model_policy_key_equivalence_whitespace_versions(
        self, policy_registry: PolicyRegistry
    ) -> None:
        """Test that ModelPolicyKey treats whitespace-padded versions equivalently.

        Registering with ' 1.0.0 ' and looking up with '1.0.0' should match.

        Note: This test intentionally uses non-normalized versions ('  1.0.0  ') to
        verify normalization behavior.
        """
        policy_registry.register_policy(
            policy_id="key-equiv-whitespace",
            policy_class=MockPolicyV1,  # type: ignore[arg-type]
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="  1.0.0  ",
        )

        # Various whitespace patterns should all resolve to same policy
        lookups = ["1.0.0", " 1.0.0", "1.0.0 ", "  1.0.0  ", "\t1.0.0\n"]
        for version in lookups:
            policy_cls = policy_registry.get("key-equiv-whitespace", version=version)
            assert policy_cls is MockPolicyV1, f"Failed for version: {version!r}"

    def test_multiple_versions_with_normalization(
        self, policy_registry: PolicyRegistry
    ) -> None:
        """Test that multiple normalized versions don't collide unexpectedly.

        Registering '1' and '1.0.0' should be treated as the same version.

        Note: This test intentionally uses non-normalized versions ('1') to verify
        normalization behavior.
        """
        policy_registry.register_policy(
            policy_id="multi-norm",
            policy_class=MockPolicyV1,  # type: ignore[arg-type]
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="1",
        )

        # Re-registering with '1.0.0' should overwrite (same normalized key)
        policy_registry.register_policy(
            policy_id="multi-norm",
            policy_class=MockPolicyV2,  # type: ignore[arg-type]
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="1.0.0",
        )

        # Should have only one entry (overwritten)
        versions = policy_registry.list_versions("multi-norm")
        assert len(versions) == 1

        # Should return the newer registration
        policy_cls = policy_registry.get("multi-norm")
        assert policy_cls is MockPolicyV2

    def test_latest_version_selection_with_partial_versions(
        self, policy_registry: PolicyRegistry
    ) -> None:
        """Test that get() returns latest version with mixed partial/full formats.

        Register '1', '2.0', '3.0.0' - get() should return the policy for '3.0.0'.

        Note: This test intentionally uses non-normalized versions ('1', '2.0') to
        verify normalization behavior.
        """
        policy_registry.register_policy(
            policy_id="latest-partial",
            policy_class=MockPolicyV1,  # type: ignore[arg-type]
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="1",  # Normalizes to 1.0.0
        )
        policy_registry.register_policy(
            policy_id="latest-partial",
            policy_class=MockSyncPolicy,  # type: ignore[arg-type]
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="2.0",  # Normalizes to 2.0.0
        )
        policy_registry.register_policy(
            policy_id="latest-partial",
            policy_class=MockPolicyV2,  # type: ignore[arg-type]
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="3.0.0",
        )

        # get() without version should return latest (3.0.0)
        latest_cls = policy_registry.get("latest-partial")
        assert latest_cls is MockPolicyV2

    # =========================================================================
    # Leading Zeros Edge Cases
    # =========================================================================

    def test_version_with_leading_zeros_in_numbers(
        self, policy_registry: PolicyRegistry
    ) -> None:
        """Test versions with leading zeros in numeric components.

        Note: Semver spec says leading zeros are NOT allowed, but we test
        what the registry does with them for documentation purposes.
        """
        # This may raise an error depending on ModelSemVer strictness
        # or may normalize. Document the actual behavior.
        try:
            policy_registry.register_policy(
                policy_id="leading-zeros",
                policy_class=MockSyncPolicy,  # type: ignore[arg-type]
                policy_type=EnumPolicyType.ORCHESTRATOR,
                version="01.02.03",
            )
            # If it succeeds, verify retrieval
            assert policy_registry.is_registered("leading-zeros")
            # Lookup behavior depends on normalization
        except ProtocolConfigurationError:
            # Leading zeros are rejected - this is valid semver behavior
            assert len(policy_registry) == 0

    def test_version_zero_zero_zero_explicit(
        self, policy_registry: PolicyRegistry
    ) -> None:
        """Test that '0.0.0' is a valid version (initial/prerelease indicator)."""
        policy_registry.register_policy(
            policy_id="zero-zero-zero",
            policy_class=MockSyncPolicy,  # type: ignore[arg-type]
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="0.0.0",
        )

        assert policy_registry.is_registered("zero-zero-zero", version="0.0.0")
        policy_cls = policy_registry.get("zero-zero-zero", version="0.0.0")
        assert policy_cls is MockSyncPolicy

    # =========================================================================
    # v-prefix Normalization Tests
    # =========================================================================

    def test_v_prefix_stripped_during_normalization(
        self, policy_registry: PolicyRegistry
    ) -> None:
        """Test that 'v1.0.0' is normalized to '1.0.0'.

        Common in git tags but not strictly semver.

        Note: This test intentionally uses non-normalized versions ('v1.0.0') to
        verify normalization behavior.
        """
        policy_registry.register_policy(
            policy_id="v-prefix",
            policy_class=MockSyncPolicy,  # type: ignore[arg-type]
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="v1.0.0",
        )

        # Should be retrievable with or without v prefix
        assert policy_registry.is_registered("v-prefix", version="1.0.0")
        policy_cls = policy_registry.get("v-prefix", version="1.0.0")
        assert policy_cls is MockSyncPolicy

        # Should also work with v prefix in lookup (both normalized)
        assert policy_registry.is_registered("v-prefix", version="v1.0.0")

    def test_v_prefix_with_partial_version(
        self, policy_registry: PolicyRegistry
    ) -> None:
        """Test that 'v2' normalizes to '2.0.0'.

        Note: This test intentionally uses non-normalized versions ('v2') to
        verify normalization behavior.
        """
        policy_registry.register_policy(
            policy_id="v-partial",
            policy_class=MockSyncPolicy,  # type: ignore[arg-type]
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="v2",
        )

        # All these lookups should work
        assert policy_registry.is_registered("v-partial", version="2")
        assert policy_registry.is_registered("v-partial", version="2.0.0")
        assert policy_registry.is_registered("v-partial", version="v2")
        assert policy_registry.is_registered("v-partial", version="v2.0.0")

    # =========================================================================
    # _parse_semver Direct Tests for Normalization
    # =========================================================================

    def test_parse_semver_partial_version_normalization(self) -> None:
        """Test _parse_semver directly handles partial version normalization."""
        PolicyRegistry._reset_semver_cache()

        # Major only
        result_1 = PolicyRegistry._parse_semver("1")
        assert result_1 == ModelSemVer(major=1, minor=0, patch=0)

        # Major.minor
        result_1_2 = PolicyRegistry._parse_semver("1.2")
        assert result_1_2 == ModelSemVer(major=1, minor=2, patch=0)

        # Zero versions
        result_0 = PolicyRegistry._parse_semver("0")
        assert result_0 == ModelSemVer(major=0, minor=0, patch=0)

        result_0_1 = PolicyRegistry._parse_semver("0.1")
        assert result_0_1 == ModelSemVer(major=0, minor=1, patch=0)

    def test_parse_semver_whitespace_normalization(self) -> None:
        """Test _parse_semver trims whitespace before parsing."""
        PolicyRegistry._reset_semver_cache()

        # All should parse to same result
        expected = ModelSemVer(major=1, minor=0, patch=0)

        assert PolicyRegistry._parse_semver(" 1.0.0 ") == expected
        assert PolicyRegistry._parse_semver("1.0.0\n") == expected
        assert PolicyRegistry._parse_semver("\t1.0.0") == expected
        assert PolicyRegistry._parse_semver("  1  ") == expected  # Partial with spaces

    def test_parse_semver_combined_normalization(self) -> None:
        """Test _parse_semver with combined whitespace and partial versions.

        Note: _parse_semver does NOT strip v-prefix - that's handled by
        _normalize_version and ModelPolicyKey. This test only verifies
        whitespace trimming + partial version expansion within _parse_semver.
        """
        PolicyRegistry._reset_semver_cache()

        # Whitespace + partial
        result = PolicyRegistry._parse_semver("  2  ")
        assert result == ModelSemVer(major=2, minor=0, patch=0)

        # Whitespace + major.minor partial
        result_2 = PolicyRegistry._parse_semver(" 3.1 ")
        assert result_2 == ModelSemVer(major=3, minor=1, patch=0)

        # Whitespace + full version
        result_3 = PolicyRegistry._parse_semver("\t4.5.6\n")
        assert result_3 == ModelSemVer(major=4, minor=5, patch=6)
