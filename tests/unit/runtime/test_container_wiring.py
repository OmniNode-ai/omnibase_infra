# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for container wiring functionality."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from omnibase_infra.enums import EnumPolicyType
from omnibase_infra.errors import ServiceRegistrationError, ServiceResolutionError
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
        """Test that wire_infrastructure_services registers all infrastructure services."""
        summary = await wire_infrastructure_services(mock_container)

        # Verify PolicyRegistry, ProtocolBindingRegistry, and RegistryCompute were registered
        assert "PolicyRegistry" in summary["services"]
        assert "ProtocolBindingRegistry" in summary["services"]
        assert "RegistryCompute" in summary["services"]

        # Verify register_instance was called three times (once for each registry)
        assert mock_container.service_registry.register_instance.call_count == 3

    async def test_wire_infrastructure_services_returns_summary(
        self, mock_container: MagicMock
    ) -> None:
        """Test that wire_infrastructure_services returns summary dict."""
        summary = await wire_infrastructure_services(mock_container)

        assert "services" in summary
        assert isinstance(summary["services"], list)
        assert (
            len(summary["services"]) >= 3
        )  # PolicyRegistry, ProtocolBindingRegistry, and RegistryCompute


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
        """Test that resolve raises ServiceResolutionError if PolicyRegistry not registered."""
        # Configure mock to raise exception (not side_effect which would return coroutine)
        mock_container.service_registry.resolve_service.return_value = None

        async def raise_error(*args: object, **kwargs: object) -> None:
            raise ValueError("Service not registered")

        mock_container.service_registry.resolve_service = raise_error

        with pytest.raises(
            ServiceResolutionError, match="PolicyRegistry not registered"
        ):
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
        """Test that ServiceRegistrationError is raised if registration fails."""
        # Configure mock to raise exception on resolve, and on register_instance
        mock_container.service_registry.resolve_service.side_effect = ValueError(
            "Service not registered"
        )
        mock_container.service_registry.register_instance.side_effect = RuntimeError(
            "Registration failed"
        )

        with pytest.raises(
            ServiceRegistrationError,
            match="Failed to create and register PolicyRegistry",
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
            """Mock policy fully implementing ProtocolPolicy for testing."""

            @property
            def policy_id(self) -> str:
                return "test_policy"

            @property
            def policy_type(self) -> EnumPolicyType:
                """Return EnumPolicyType for proper protocol implementation."""
                return EnumPolicyType.ORCHESTRATOR

            def evaluate(self, context: dict[str, object]) -> dict[str, object]:
                return {"result": True}

            def decide(self, context: dict[str, object]) -> dict[str, object]:
                return self.evaluate(context)

        registry.register_policy(
            policy_id="test_policy",
            policy_class=MockPolicy,
            policy_type=EnumPolicyType.ORCHESTRATOR,
            version="1.0.0",
        )

        # Step 3: Retrieve and verify policy class
        policy_cls = registry.get("test_policy")
        # Note: registry.get() returns type[ProtocolPolicy], not MockPolicy directly
        # so we verify by instantiation and usage rather than identity check
        assert policy_cls is not None

        # Step 4: Instantiate and use policy
        policy = policy_cls()
        result = policy.evaluate({"test": "context"})
        assert result == {"result": True}

        # Verify the policy has expected properties
        assert policy.policy_id == "test_policy"
        assert policy.policy_type == EnumPolicyType.ORCHESTRATOR

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


class TestAnalyzeAttributeError:
    """Test _analyze_attribute_error() helper function."""

    def test_service_registry_missing(self) -> None:
        """Test detection of missing service_registry attribute.

        Note: After OMN-1257 refactoring, service_registry missing/None cases
        are handled by _validate_service_registry() which raises
        ServiceRegistryUnavailableError before _analyze_attribute_error is called.
        This test verifies the fallback behavior for service_registry when it
        somehow reaches _analyze_attribute_error (returns generic hint).
        """
        from omnibase_infra.runtime.container_wiring import _analyze_attribute_error

        error_str = "'MockContainer' object has no attribute 'service_registry'"
        missing_attr, hint = _analyze_attribute_error(error_str)

        # After refactoring, service_registry case returns generic hint
        # (actual service_registry validation is in _validate_service_registry)
        assert missing_attr == "service_registry"
        assert "service_registry" in hint

    def test_register_instance_missing(self) -> None:
        """Test detection of missing register_instance method."""
        from omnibase_infra.runtime.container_wiring import _analyze_attribute_error

        error_str = "'MockRegistry' object has no attribute 'register_instance'"
        _missing_attr, hint = _analyze_attribute_error(error_str)

        assert "register_instance" in hint
        assert "v0.5.6" in hint

    def test_unknown_attribute(self) -> None:
        """Test handling of unknown missing attribute."""
        from omnibase_infra.runtime.container_wiring import _analyze_attribute_error

        error_str = "'MockContainer' object has no attribute 'unknown_attr'"
        missing_attr, hint = _analyze_attribute_error(error_str)

        assert missing_attr == "unknown_attr"
        assert "unknown_attr" in hint

    def test_no_quotes_in_error(self) -> None:
        """Test handling of error without quotes."""
        from omnibase_infra.runtime.container_wiring import _analyze_attribute_error

        error_str = "Some attribute error without quotes"
        missing_attr, _hint = _analyze_attribute_error(error_str)

        assert missing_attr == "unknown"


class TestAnalyzeTypeError:
    """Test _analyze_type_error() helper function."""

    def test_interface_argument_error(self) -> None:
        """Test detection of interface argument issues."""
        from omnibase_infra.runtime.container_wiring import _analyze_type_error

        error_str = "interface argument must be a type"
        invalid_arg, hint = _analyze_type_error(error_str)

        assert invalid_arg == "interface"
        assert "type class" in hint

    def test_instance_argument_error(self) -> None:
        """Test detection of instance argument issues."""
        from omnibase_infra.runtime.container_wiring import _analyze_type_error

        # Note: error string must not contain "interface" since that's checked first
        error_str = "instance must be an object of the correct type"
        invalid_arg, hint = _analyze_type_error(error_str)

        assert invalid_arg == "instance"
        assert "instance of the interface" in hint

    def test_scope_argument_error(self) -> None:
        """Test detection of scope argument issues."""
        from omnibase_infra.runtime.container_wiring import _analyze_type_error

        error_str = "scope must be one of: global, request, transient"
        invalid_arg, hint = _analyze_type_error(error_str)

        assert invalid_arg == "scope"
        assert "global" in hint

    def test_metadata_argument_error(self) -> None:
        """Test detection of metadata argument issues."""
        from omnibase_infra.runtime.container_wiring import _analyze_type_error

        error_str = "metadata must be a dict"
        invalid_arg, hint = _analyze_type_error(error_str)

        assert invalid_arg == "metadata"
        assert "dict" in hint

    def test_positional_argument_error(self) -> None:
        """Test detection of positional argument mismatch."""
        from omnibase_infra.runtime.container_wiring import _analyze_type_error

        # Note: error string must not contain other keywords like 'instance'
        error_str = "function() takes 2 positional args but 4 were given"
        invalid_arg, hint = _analyze_type_error(error_str)

        assert invalid_arg == "signature"
        assert "mismatch" in hint.lower()

    def test_argument_keyword_error(self) -> None:
        """Test detection of argument keyword errors."""
        from omnibase_infra.runtime.container_wiring import _analyze_type_error

        error_str = "got an unexpected keyword argument 'invalid_arg'"
        invalid_arg, hint = _analyze_type_error(error_str)

        assert invalid_arg == "signature"
        assert "mismatch" in hint.lower()

    def test_unknown_type_error(self) -> None:
        """Test handling of unknown type errors."""
        from omnibase_infra.runtime.container_wiring import _analyze_type_error

        error_str = "some unexpected type error"
        invalid_arg, hint = _analyze_type_error(error_str)

        assert invalid_arg == "unknown"
        assert "compatibility" in hint
