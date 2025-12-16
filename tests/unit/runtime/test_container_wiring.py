# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for container wiring functionality."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from omnibase_infra.enums import EnumPolicyType
from omnibase_infra.runtime.container_wiring import (
    get_or_create_policy_registry,
    get_policy_registry_from_container,
    wire_infrastructure_services,
)
from omnibase_infra.runtime.policy_registry import PolicyRegistry


class TestWireInfrastructureServices:
    """Test wire_infrastructure_services() function."""

    async def test_wire_infrastructure_services_registers_policy_registry(
        self, mock_container: MagicMock
    ) -> None:
        """Test that wire_infrastructure_services registers PolicyRegistry."""
        summary = await wire_infrastructure_services(mock_container)

        # Verify both PolicyRegistry and ProtocolBindingRegistry were registered
        assert "PolicyRegistry" in summary["services"]
        assert "ProtocolBindingRegistry" in summary["services"]

        # Verify register_instance was called twice (once for each registry)
        assert mock_container.service_registry.register_instance.call_count == 2

    async def test_wire_infrastructure_services_returns_summary(
        self, mock_container: MagicMock
    ) -> None:
        """Test that wire_infrastructure_services returns summary dict."""
        summary = await wire_infrastructure_services(mock_container)

        assert "services" in summary
        assert isinstance(summary["services"], list)
        assert (
            len(summary["services"]) >= 2
        )  # At least PolicyRegistry and ProtocolBindingRegistry


class TestGetPolicyRegistryFromContainer:
    """Test get_policy_registry_from_container() function."""

    async def test_resolve_policy_registry_from_container(
        self, container_with_policy_registry: PolicyRegistry, mock_container: MagicMock
    ) -> None:
        """Test resolving PolicyRegistry from container."""
        registry = await get_policy_registry_from_container(mock_container)

        assert registry is container_with_policy_registry
        assert isinstance(registry, PolicyRegistry)

    async def test_resolve_raises_error_if_not_registered(
        self, mock_container: MagicMock
    ) -> None:
        """Test that resolve raises RuntimeError if PolicyRegistry not registered."""
        # Configure mock to raise exception (not side_effect which would return coroutine)
        mock_container.service_registry.resolve_service.return_value = None

        async def raise_error(*args: object, **kwargs: object) -> None:
            raise ValueError("Service not registered")

        mock_container.service_registry.resolve_service = raise_error  # type: ignore[method-assign]

        with pytest.raises(RuntimeError, match="PolicyRegistry not registered"):
            await get_policy_registry_from_container(mock_container)


class TestGetOrCreatePolicyRegistry:
    """Test get_or_create_policy_registry() function."""

    async def test_returns_existing_registry_if_found(
        self, container_with_policy_registry: PolicyRegistry, mock_container: MagicMock
    ) -> None:
        """Test that existing PolicyRegistry is returned if found."""
        registry = await get_or_create_policy_registry(mock_container)

        assert registry is container_with_policy_registry
        assert isinstance(registry, PolicyRegistry)

    async def test_creates_and_registers_if_not_found(
        self, mock_container: MagicMock
    ) -> None:
        """Test that PolicyRegistry is created and registered if not found."""
        # Configure mock to raise exception first, then return None
        mock_container.service_registry.resolve_service.side_effect = ValueError(
            "Service not registered"
        )

        registry = await get_or_create_policy_registry(mock_container)

        # Verify registry was created
        assert isinstance(registry, PolicyRegistry)

        # Verify register_instance was called
        mock_container.service_registry.register_instance.assert_called_once()
        call_kwargs = mock_container.service_registry.register_instance.call_args[1]
        assert call_kwargs["interface"] == PolicyRegistry
        assert call_kwargs["instance"] is registry
        assert call_kwargs["scope"] == "global"
        assert call_kwargs["metadata"]["auto_registered"] is True

    async def test_raises_error_if_registration_fails(
        self, mock_container: MagicMock
    ) -> None:
        """Test that RuntimeError is raised if registration fails."""
        # Configure mock to raise exception on resolve, and on register_instance
        mock_container.service_registry.resolve_service.side_effect = ValueError(
            "Service not registered"
        )
        mock_container.service_registry.register_instance.side_effect = RuntimeError(
            "Registration failed"
        )

        with pytest.raises(
            RuntimeError, match="Failed to create and register PolicyRegistry"
        ):
            await get_or_create_policy_registry(mock_container)


class TestContainerBasedPolicyUsage:
    """Integration tests demonstrating container-based policy usage."""

    async def test_full_container_based_workflow(
        self, container_with_policy_registry: PolicyRegistry, mock_container: MagicMock
    ) -> None:
        """Test full workflow: wire -> resolve -> register -> retrieve policy."""

        # Step 1: Resolve registry from container (async call)
        registry = await get_policy_registry_from_container(mock_container)
        assert registry is container_with_policy_registry

        # Step 2: Register a policy
        class MockPolicy:
            """Mock policy for testing."""

            @property
            def policy_id(self) -> str:
                return "test_policy"

            @property
            def policy_type(self) -> str:
                return "orchestrator"

            def evaluate(self, context: dict[str, object]) -> dict[str, object]:
                return {"result": True}

            def decide(self, context: dict[str, object]) -> dict[str, object]:
                return {"result": True}

        registry.register_policy(
            policy_id="test_policy",
            policy_class=MockPolicy,  # type: ignore[arg-type]
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="1.0.0",
        )

        # Step 3: Retrieve and verify policy
        policy_cls = registry.get("test_policy")
        assert policy_cls is MockPolicy

        # Step 4: Instantiate and use policy
        policy = policy_cls()
        result = policy.evaluate({"test": "context"})
        assert result == {"result": True}

    async def test_multiple_container_instances_isolated(
        self, mock_container: MagicMock
    ) -> None:
        """Test that multiple containers have isolated registries."""
        from unittest.mock import AsyncMock

        # Create first registry
        mock_container.service_registry.resolve_service.side_effect = ValueError(
            "Service not registered"
        )
        registry1 = await get_or_create_policy_registry(mock_container)

        # Create second mock container
        mock_container2 = MagicMock()
        mock_container2.service_registry = MagicMock()
        mock_container2.service_registry.resolve_service.side_effect = ValueError(
            "Service not registered"
        )
        mock_container2.service_registry.register_instance = AsyncMock(
            return_value="mock-uuid-2"
        )
        registry2 = await get_or_create_policy_registry(mock_container2)

        # Verify they are different instances
        assert registry1 is not registry2
        assert isinstance(registry1, PolicyRegistry)
        assert isinstance(registry2, PolicyRegistry)
