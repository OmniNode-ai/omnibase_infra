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

from omnibase_infra.enums import EnumPolicyType
from omnibase_infra.runtime.container_wiring import (
    get_handler_registry_from_container,
    get_or_create_policy_registry,
    get_policy_registry_from_container,
    wire_infrastructure_services,
)
from omnibase_infra.runtime.handler_registry import ProtocolBindingRegistry
from omnibase_infra.runtime.policy_registry import PolicyRegistry


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

        # Wire infrastructure services (async operation)
        summary = await wire_infrastructure_services(container)

        # Verify PolicyRegistry is in the summary
        assert "PolicyRegistry" in summary["services"]
        assert "ProtocolBindingRegistry" in summary["services"]

        # Resolve PolicyRegistry from container
        registry = await get_policy_registry_from_container(container)

        # Verify it's a real PolicyRegistry instance
        assert isinstance(registry, PolicyRegistry)
        assert len(registry) == 0  # Empty initially

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
        await wire_infrastructure_services(container1)
        registry1 = await get_policy_registry_from_container(container1)

        # Create second container and wire it
        container2 = ModelONEXContainer()
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

        # get_or_create should create and register PolicyRegistry
        registry1 = await get_or_create_policy_registry(container)
        assert isinstance(registry1, PolicyRegistry)

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
        summary = await wire_infrastructure_services(container)

        # Verify both services are registered
        assert "PolicyRegistry" in summary["services"]
        assert "ProtocolBindingRegistry" in summary["services"]

        # Resolve handler registry
        handler_registry = await get_handler_registry_from_container(container)

        # Verify it implements ProtocolBindingRegistry interface via duck typing
        # Per ONEX conventions, check for required methods rather than isinstance
        required_methods = ["register", "get", "list_protocols", "is_registered"]
        for method_name in required_methods:
            assert hasattr(
                handler_registry, method_name
            ), f"Handler registry must have '{method_name}' method"
            assert callable(
                getattr(handler_registry, method_name)
            ), f"'{method_name}' must be callable"

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


class TestContainerWiringErrorHandling:
    """Integration tests for error handling with real containers."""

    @pytest.mark.asyncio
    async def test_resolve_before_wire_raises_error(self) -> None:
        """Test that resolving before wiring raises RuntimeError.

        This verifies proper error handling when services not wired.
        """
        container = ModelONEXContainer()

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
        assert isinstance(policy_registry, PolicyRegistry)

        # Resolve ProtocolBindingRegistry
        handler_registry = (
            await container_with_registries.service_registry.resolve_service(
                ProtocolBindingRegistry
            )
        )
        # Verify interface via duck typing (ONEX convention)
        assert hasattr(handler_registry, "register"), "Must have 'register' method"
        assert hasattr(
            handler_registry, "list_protocols"
        ), "Must have 'list_protocols' method"

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
