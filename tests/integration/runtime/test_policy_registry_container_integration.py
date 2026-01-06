# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Integration tests for PolicyRegistry with real ModelONEXContainer.

These tests verify PolicyRegistry works correctly with actual omnibase_core
container implementation, not mocks. Tests cover:

1. Container wiring and service resolution
2. Policy registration through resolved registries
3. Container isolation (separate containers have separate registries)
4. Lazy initialization via get_or_create_policy_registry
5. Handler registry wiring alongside policy registry

Design Principles:
- Use real ModelONEXContainer from omnibase_core
- No mocking of container internals
- Verify actual async container API behavior
- Test isolation via fresh container instances per test
"""

from __future__ import annotations

import pytest
from omnibase_core.container import ModelONEXContainer

# Skip message for omnibase_core 0.6.2 circular import bug
_SKIP_SERVICE_REGISTRY_NONE = (
    "Skipped: omnibase_core circular import bug - service_registry is None. "
    "See: model_onex_container.py -> container_service_registry.py -> "
    "container/__init__.py -> container_service_resolver.py -> ModelONEXContainer. "
    "Upgrade to omnibase_core >= 0.6.3 to run these tests."
)


def _skip_if_service_registry_none(container: ModelONEXContainer) -> None:
    """Skip test if container.service_registry is None due to circular import bug."""
    if container.service_registry is None:
        pytest.skip(_SKIP_SERVICE_REGISTRY_NONE)


from omnibase_infra.enums import EnumPolicyType
from omnibase_infra.runtime.container_wiring import (
    get_handler_registry_from_container,
    get_or_create_policy_registry,
    get_policy_registry_from_container,
    wire_infrastructure_services,
)
from omnibase_infra.runtime.handler_registry import ProtocolBindingRegistry
from omnibase_infra.runtime.policy_registry import PolicyRegistry

# Import shared conformance helpers from conftest
from tests.conftest import (
    assert_handler_registry_interface,
    assert_policy_registry_interface,
)

# Module-level markers - all tests in this file are integration tests
pytestmark = [pytest.mark.integration]


def _check_service_registry_available() -> bool:
    """Check if ServiceRegistry is available in omnibase_core.

    Creates a temporary container to check for service_registry availability,
    then explicitly cleans up the container to prevent resource leaks.

    Returns:
        True if service_registry is available and not None, False otherwise.
    """
    container = None
    try:
        container = ModelONEXContainer()
        return container.service_registry is not None
    except AttributeError:
        # service_registry attribute removed in omnibase_core 0.6.x
        return False
    except TypeError:
        # ModelONEXContainer.__init__ signature changed (new required params)
        return False
    finally:
        # Explicit cleanup of temporary container
        del container


# Module-level flag for pytest.mark.skipif decorators
_service_registry_available = _check_service_registry_available()


@pytest.mark.skipif(
    not _service_registry_available,
    reason="ServiceRegistry not available in omnibase_core (removed in 0.6.x)",
)
class TestPolicyRegistryContainerIntegration:
    """Integration tests for PolicyRegistry with real container."""

    @pytest.mark.asyncio
    async def test_wire_and_resolve_policy_registry_real_container(self) -> None:
        """Test wiring and resolving PolicyRegistry with real container.

        This test verifies the complete container-based DI workflow:
        1. Create real ModelONEXContainer
        2. Wire infrastructure services
        3. Resolve PolicyRegistry from container
        4. Verify instance type
        """
        # Create real container
        container = ModelONEXContainer()

        # Skip if service_registry is None (circular import bug)
        _skip_if_service_registry_none(container)

        # Wire infrastructure services (async operation)
        summary = await wire_infrastructure_services(container)

        # Verify PolicyRegistry is in the summary
        assert "PolicyRegistry" in summary["services"]
        assert "ProtocolBindingRegistry" in summary["services"]

        # Resolve PolicyRegistry from container
        registry = await get_policy_registry_from_container(container)

        # Verify PolicyRegistry interface via shared conformance helper
        assert_policy_registry_interface(registry)
        assert len(registry) == 0  # Empty initially (also verifies __len__ works)

    @pytest.mark.asyncio
    async def test_policy_registration_through_container(self) -> None:
        """Test registering and retrieving policies via container-resolved registry.

        This test verifies that:
        1. Wire services to container
        2. Resolve registry from container
        3. Register a policy
        4. Retrieve the same policy
        """
        # Wire container
        container = ModelONEXContainer()

        # Skip if service_registry is None (circular import bug)
        _skip_if_service_registry_none(container)

        await wire_infrastructure_services(container)

        # Resolve registry
        registry = await get_policy_registry_from_container(container)

        # Create a mock policy class for testing
        class TestPolicy:
            """Test policy implementing minimal ProtocolPolicy interface."""

            @property
            def policy_id(self) -> str:
                return "integration_test_policy"

            @property
            def policy_type(self) -> EnumPolicyType:
                return EnumPolicyType.ORCHESTRATOR

            def evaluate(self, context: dict[str, object]) -> dict[str, object]:
                return {"evaluated": True}

            def decide(self, context: dict[str, object]) -> dict[str, object]:
                return self.evaluate(context)

        # Register policy
        registry.register_policy(
            policy_id="integration_test_policy",
            policy_class=TestPolicy,  # type: ignore[arg-type]
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="1.0.0",
        )

        # Verify policy can be retrieved
        assert registry.is_registered("integration_test_policy")
        policy_cls = registry.get("integration_test_policy")
        assert policy_cls is TestPolicy

        # Verify policy instance works correctly
        policy = policy_cls()
        result = policy.evaluate({"test": "context"})
        assert result == {"evaluated": True}
        assert policy.policy_id == "integration_test_policy"
        assert policy.policy_type == EnumPolicyType.ORCHESTRATOR

    @pytest.mark.asyncio
    async def test_container_isolation_real_containers(self) -> None:
        """Test that separate containers have isolated registries.

        This test verifies container isolation:
        1. Create two separate real containers
        2. Wire both
        3. Register different policies in each
        4. Verify they are completely isolated
        """
        # Create first container and wire it
        container1 = ModelONEXContainer()

        # Skip if service_registry is None (circular import bug)
        _skip_if_service_registry_none(container1)

        await wire_infrastructure_services(container1)
        registry1 = await get_policy_registry_from_container(container1)

        # Create second container and wire it
        container2 = ModelONEXContainer()

        # Skip if service_registry is None (circular import bug)
        _skip_if_service_registry_none(container2)

        await wire_infrastructure_services(container2)
        registry2 = await get_policy_registry_from_container(container2)

        # Verify registries are different instances
        assert registry1 is not registry2

        # Create distinct test policies
        class PolicyA:
            """Policy for container 1."""

            @property
            def policy_id(self) -> str:
                return "policy_a"

            @property
            def policy_type(self) -> EnumPolicyType:
                return EnumPolicyType.ORCHESTRATOR

            def evaluate(self, context: dict[str, object]) -> dict[str, object]:
                return {"source": "container1"}

        class PolicyB:
            """Policy for container 2."""

            @property
            def policy_id(self) -> str:
                return "policy_b"

            @property
            def policy_type(self) -> EnumPolicyType:
                return EnumPolicyType.REDUCER

            def evaluate(self, context: dict[str, object]) -> dict[str, object]:
                return {"source": "container2"}

        # Register different policies in each registry
        registry1.register_policy(
            policy_id="policy_a",
            policy_class=PolicyA,  # type: ignore[arg-type]
            policy_type=EnumPolicyType.ORCHESTRATOR,
        )

        registry2.register_policy(
            policy_id="policy_b",
            policy_class=PolicyB,  # type: ignore[arg-type]
            policy_type=EnumPolicyType.REDUCER,
        )

        # Verify isolation - each registry only has its own policy
        assert registry1.is_registered("policy_a")
        assert not registry1.is_registered("policy_b")

        assert registry2.is_registered("policy_b")
        assert not registry2.is_registered("policy_a")

        # Verify policy types are distinct
        assert registry1.list_policy_types() == ["orchestrator"]
        assert registry2.list_policy_types() == ["reducer"]

    @pytest.mark.asyncio
    async def test_get_or_create_policy_registry_real_container(self) -> None:
        """Test get_or_create_policy_registry with unwired container.

        This test verifies lazy initialization:
        1. Create container WITHOUT wiring
        2. Call get_or_create_policy_registry
        3. Verify it creates and registers PolicyRegistry
        4. Call again to verify same instance is returned
        """
        # Create real container WITHOUT wiring
        container = ModelONEXContainer()

        # Skip if service_registry is None (circular import bug)
        _skip_if_service_registry_none(container)

        # get_or_create should create and register PolicyRegistry
        registry1 = await get_or_create_policy_registry(container)

        # Verify PolicyRegistry interface via shared conformance helper
        assert_policy_registry_interface(registry1)
        assert len(registry1) == 0  # Empty initially (also verifies __len__ works)

        # Second call should return same instance
        registry2 = await get_or_create_policy_registry(container)
        assert registry1 is registry2

        # Verify registry is functional
        class LazyPolicy:
            """Policy for lazy initialization test."""

            def evaluate(self, context: dict[str, object]) -> dict[str, object]:
                return {"lazy": True}

        registry1.register_policy(
            policy_id="lazy_policy",
            policy_class=LazyPolicy,  # type: ignore[arg-type]
            policy_type=EnumPolicyType.ORCHESTRATOR,
        )

        # Should be accessible from second reference
        assert registry2.is_registered("lazy_policy")
        assert registry2.get("lazy_policy") is LazyPolicy

    @pytest.mark.asyncio
    async def test_handler_registry_wiring(self) -> None:
        """Test ProtocolBindingRegistry wiring alongside PolicyRegistry.

        This test verifies that both registries are wired correctly:
        1. Wire services to container
        2. Resolve ProtocolBindingRegistry
        3. Verify it's the correct type
        4. Verify basic operations work
        """
        # Create and wire container
        container = ModelONEXContainer()

        # Skip if service_registry is None (circular import bug)
        _skip_if_service_registry_none(container)

        summary = await wire_infrastructure_services(container)

        # Verify both services are registered
        assert "PolicyRegistry" in summary["services"]
        assert "ProtocolBindingRegistry" in summary["services"]

        # Resolve handler registry
        handler_registry = await get_handler_registry_from_container(container)

        # Verify ProtocolBindingRegistry interface via shared conformance helper
        assert_handler_registry_interface(handler_registry)

        # Verify basic operations work
        assert len(handler_registry) == 0  # Empty initially
        assert handler_registry.list_protocols() == []

    @pytest.mark.asyncio
    async def test_both_registries_resolve_to_same_instances(self) -> None:
        """Test that repeated resolution returns same instances.

        This verifies singleton behavior per container:
        1. Wire container
        2. Resolve PolicyRegistry twice
        3. Resolve ProtocolBindingRegistry twice
        4. Verify same instances returned each time
        """
        container = ModelONEXContainer()

        # Skip if service_registry is None (circular import bug)
        _skip_if_service_registry_none(container)

        await wire_infrastructure_services(container)

        # Resolve PolicyRegistry twice
        policy_reg1 = await get_policy_registry_from_container(container)
        policy_reg2 = await get_policy_registry_from_container(container)
        assert policy_reg1 is policy_reg2

        # Resolve ProtocolBindingRegistry twice
        handler_reg1 = await get_handler_registry_from_container(container)
        handler_reg2 = await get_handler_registry_from_container(container)
        assert handler_reg1 is handler_reg2

    @pytest.mark.asyncio
    async def test_policy_registry_version_resolution(self) -> None:
        """Test multi-version policy registration via real container.

        This test verifies that version resolution works correctly:
        1. Register multiple versions of same policy
        2. Verify get() returns latest version by default
        3. Verify specific version can be retrieved
        """
        container = ModelONEXContainer()

        # Skip if service_registry is None (circular import bug)
        _skip_if_service_registry_none(container)

        await wire_infrastructure_services(container)
        registry = await get_policy_registry_from_container(container)

        # Create versioned policies
        class PolicyV1:
            """Version 1.0.0 of test policy."""

            def evaluate(self, context: dict[str, object]) -> dict[str, object]:
                return {"version": "1.0.0"}

        class PolicyV2:
            """Version 2.0.0 of test policy."""

            def evaluate(self, context: dict[str, object]) -> dict[str, object]:
                return {"version": "2.0.0"}

        class PolicyV10:
            """Version 10.0.0 of test policy (tests semantic versioning)."""

            def evaluate(self, context: dict[str, object]) -> dict[str, object]:
                return {"version": "10.0.0"}

        # Register multiple versions
        registry.register_policy(
            policy_id="versioned_policy",
            policy_class=PolicyV1,  # type: ignore[arg-type]
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="1.0.0",
        )
        registry.register_policy(
            policy_id="versioned_policy",
            policy_class=PolicyV2,  # type: ignore[arg-type]
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="2.0.0",
        )
        registry.register_policy(
            policy_id="versioned_policy",
            policy_class=PolicyV10,  # type: ignore[arg-type]
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="10.0.0",
        )

        # Verify list_versions returns all versions
        versions = registry.list_versions("versioned_policy")
        assert "1.0.0" in versions
        assert "2.0.0" in versions
        assert "10.0.0" in versions

        # Verify get() returns latest version (10.0.0, not 2.0.0 - semantic sorting)
        latest = registry.get("versioned_policy")
        assert latest is PolicyV10

        # Verify specific version can be retrieved
        v1 = registry.get("versioned_policy", version="1.0.0")
        assert v1 is PolicyV1

        v2 = registry.get("versioned_policy", version="2.0.0")
        assert v2 is PolicyV2


@pytest.mark.skipif(
    not _service_registry_available,
    reason="ServiceRegistry not available in omnibase_core (removed in 0.6.x)",
)
class TestContainerWiringErrorHandling:
    """Integration tests for error handling with real containers."""

    @pytest.mark.asyncio
    async def test_resolve_before_wire_raises_error(self) -> None:
        """Test that resolving before wiring raises RuntimeError.

        This verifies proper error handling when services not wired.
        """
        container = ModelONEXContainer()

        # Skip if service_registry is None (circular import bug)
        _skip_if_service_registry_none(container)

        # Attempt to resolve without wiring should fail
        with pytest.raises(RuntimeError, match="PolicyRegistry not registered"):
            await get_policy_registry_from_container(container)

    @pytest.mark.asyncio
    async def test_double_wire_preserves_existing_registry(self) -> None:
        """Test that double-wiring preserves existing registry instances.

        This verifies idempotent wiring behavior - the container maintains
        singleton semantics, so wiring twice returns the same instance.
        This is the expected behavior for global-scoped services.
        """
        container = ModelONEXContainer()

        # Skip if service_registry is None (circular import bug)
        _skip_if_service_registry_none(container)

        # First wire
        await wire_infrastructure_services(container)
        registry1 = await get_policy_registry_from_container(container)

        # Register a policy
        class TestPolicy:
            def evaluate(self, context: dict[str, object]) -> dict[str, object]:
                return {}

        registry1.register_policy(
            policy_id="before_rewire",
            policy_class=TestPolicy,  # type: ignore[arg-type]
            policy_type=EnumPolicyType.ORCHESTRATOR,
        )

        # Second wire - container preserves existing singleton
        await wire_infrastructure_services(container)
        registry2 = await get_policy_registry_from_container(container)

        # Registries should be the SAME instance (global scope = singleton per container)
        assert registry1 is registry2

        # Policy from registry1 should still be accessible via registry2
        assert registry2.is_registered("before_rewire")
        assert registry2.get("before_rewire") is TestPolicy


@pytest.mark.skipif(
    not _service_registry_available,
    reason="ServiceRegistry not available in omnibase_core (removed in 0.6.x)",
)
class TestContainerWithRegistriesFixture:
    """Tests using the container_with_registries fixture."""

    @pytest.mark.asyncio
    async def test_fixture_provides_wired_container(
        self, container_with_registries: ModelONEXContainer
    ) -> None:
        """Test that fixture provides properly wired container."""
        # Resolve PolicyRegistry
        policy_registry = (
            await container_with_registries.service_registry.resolve_service(
                PolicyRegistry
            )
        )

        # Verify PolicyRegistry interface via shared conformance helper
        assert_policy_registry_interface(policy_registry)

        # Resolve ProtocolBindingRegistry
        handler_registry = (
            await container_with_registries.service_registry.resolve_service(
                ProtocolBindingRegistry
            )
        )

        # Verify ProtocolBindingRegistry interface via shared conformance helper
        assert_handler_registry_interface(handler_registry)

    @pytest.mark.asyncio
    async def test_fixture_registries_are_functional(
        self, container_with_registries: ModelONEXContainer
    ) -> None:
        """Test that fixture-provided registries work correctly."""
        policy_registry = (
            await container_with_registries.service_registry.resolve_service(
                PolicyRegistry
            )
        )

        class FixtureTestPolicy:
            def evaluate(self, context: dict[str, object]) -> dict[str, object]:
                return {"from_fixture": True}

        policy_registry.register_policy(
            policy_id="fixture_test",
            policy_class=FixtureTestPolicy,  # type: ignore[arg-type]
            policy_type=EnumPolicyType.REDUCER,
        )

        assert policy_registry.is_registered("fixture_test")
        retrieved = policy_registry.get("fixture_test")
        assert retrieved is FixtureTestPolicy
