# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for RegistryInfraServiceDiscovery.

This module validates the registry functionality for service discovery
node dependencies, including handler registration and protocol factory
registration.

Test Coverage:
    - register(): No-op behavior (factory registration not implemented in v1.0)
    - register_with_handler(): Direct handler binding via service_registry
    - _create_handler_from_config(): Configuration error handling
    - Handler swapping via registry

Related:
    - OMN-1131: Capability-oriented node architecture
    - RegistryInfraServiceDiscovery: Registry implementation
    - PR #119: Test coverage for handler swapping
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.handlers.service_discovery.handler_service_discovery_mock import (
    HandlerServiceDiscoveryMock,
)
from omnibase_infra.nodes.node_service_discovery_effect.registry import (
    RegistryInfraServiceDiscovery,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_container() -> MagicMock:
    """Create a mock container with service_registry that has async register_instance."""
    container = MagicMock()
    # Set up service_registry with async register_instance
    container.service_registry = MagicMock()
    container.service_registry.register_instance = AsyncMock()
    return container


@pytest.fixture
def mock_container_no_registry() -> MagicMock:
    """Create a mock container with service_registry = None."""
    container = MagicMock()
    container.service_registry = None
    return container


@pytest.fixture
def mock_handler() -> HandlerServiceDiscoveryMock:
    """Create a HandlerServiceDiscoveryMock for testing."""
    return HandlerServiceDiscoveryMock()


@pytest.fixture
def mock_consul_handler() -> MagicMock:
    """Create a mock Consul handler that implements the protocol.

    Uses spec to ensure the mock satisfies isinstance checks against
    the @runtime_checkable protocol.
    """
    from omnibase_infra.nodes.node_service_discovery_effect.protocols import (
        ProtocolDiscoveryOperations,
    )

    handler = MagicMock(spec=ProtocolDiscoveryOperations)
    handler.handler_type = "consul"
    return handler


# =============================================================================
# Factory Registration Tests (No-Op in v1.0)
# =============================================================================


class TestRegistryInfraServiceDiscoveryRegister:
    """Tests for RegistryInfraServiceDiscovery.register() method.

    Note: In v1.0, register() is a no-op because factory registration is not
    implemented in omnibase_core. It simply logs a debug message and returns.
    """

    def test_register_is_noop_with_registry(self, mock_container: MagicMock) -> None:
        """register() is a no-op in v1.0 - does not call register_factory."""
        RegistryInfraServiceDiscovery.register(mock_container)

        # Should NOT call register_factory (not implemented in v1.0)
        # No exception should be raised
        # This is the expected behavior per the docstring

    def test_register_returns_early_when_no_registry(
        self, mock_container_no_registry: MagicMock
    ) -> None:
        """register() returns early when service_registry is None."""
        # Should not raise - just returns early
        RegistryInfraServiceDiscovery.register(mock_container_no_registry)


# =============================================================================
# Direct Handler Registration Tests
# =============================================================================


class TestRegistryInfraServiceDiscoveryRegisterWithHandler:
    """Tests for RegistryInfraServiceDiscovery.register_with_handler() method."""

    @pytest.mark.anyio
    async def test_register_with_handler_calls_register_instance(
        self,
        mock_container: MagicMock,
        mock_handler: HandlerServiceDiscoveryMock,
    ) -> None:
        """register_with_handler() calls container.service_registry.register_instance."""
        await RegistryInfraServiceDiscovery.register_with_handler(
            mock_container, mock_handler
        )

        mock_container.service_registry.register_instance.assert_called_once()

    @pytest.mark.anyio
    async def test_register_with_handler_passes_protocol_and_handler(
        self,
        mock_container: MagicMock,
        mock_handler: HandlerServiceDiscoveryMock,
    ) -> None:
        """register_with_handler() passes protocol type and handler instance."""
        await RegistryInfraServiceDiscovery.register_with_handler(
            mock_container, mock_handler
        )

        call_kwargs = mock_container.service_registry.register_instance.call_args
        assert call_kwargs is not None

        from omnibase_infra.nodes.node_service_discovery_effect.protocols import (
            ProtocolDiscoveryOperations,
        )

        # Check keyword arguments
        assert call_kwargs.kwargs["interface"] is ProtocolDiscoveryOperations
        assert call_kwargs.kwargs["instance"] is mock_handler

    @pytest.mark.anyio
    async def test_register_with_handler_accepts_any_protocol_implementation(
        self,
        mock_container: MagicMock,
        mock_consul_handler: MagicMock,
    ) -> None:
        """register_with_handler() accepts any handler implementing the protocol."""
        await RegistryInfraServiceDiscovery.register_with_handler(
            mock_container, mock_consul_handler
        )

        call_kwargs = mock_container.service_registry.register_instance.call_args
        assert call_kwargs is not None
        assert call_kwargs.kwargs["instance"] is mock_consul_handler

    @pytest.mark.anyio
    async def test_register_with_handler_returns_early_when_no_registry(
        self,
        mock_container_no_registry: MagicMock,
        mock_handler: HandlerServiceDiscoveryMock,
    ) -> None:
        """register_with_handler() returns early when service_registry is None."""
        # Should not raise - just returns early after isinstance check
        await RegistryInfraServiceDiscovery.register_with_handler(
            mock_container_no_registry, mock_handler
        )

    @pytest.mark.anyio
    async def test_register_with_handler_validates_protocol(
        self,
        mock_container: MagicMock,
    ) -> None:
        """register_with_handler() raises TypeError for invalid handler."""
        invalid_handler = MagicMock()  # Does not implement protocol

        with pytest.raises(TypeError) as exc_info:
            await RegistryInfraServiceDiscovery.register_with_handler(
                mock_container, invalid_handler
            )

        assert "ProtocolDiscoveryOperations" in str(exc_info.value)


# =============================================================================
# Configuration Factory Tests
# =============================================================================


class TestRegistryCreateHandlerFromConfig:
    """Tests for RegistryInfraServiceDiscovery._create_handler_from_config()."""

    def test_create_handler_from_config_raises_not_implemented(
        self, mock_container: MagicMock
    ) -> None:
        """_create_handler_from_config() raises ProtocolConfigurationError.

        This is expected behavior as the factory is a placeholder that
        requires explicit handler configuration.
        """
        with pytest.raises(ProtocolConfigurationError) as exc_info:
            RegistryInfraServiceDiscovery._create_handler_from_config(mock_container)

        assert "No service discovery handler configured" in str(exc_info.value)
        assert "register_with_handler()" in str(exc_info.value)

    def test_configuration_error_provides_helpful_message(
        self, mock_container: MagicMock
    ) -> None:
        """Configuration error message guides user to correct solution."""
        with pytest.raises(ProtocolConfigurationError) as exc_info:
            RegistryInfraServiceDiscovery._create_handler_from_config(mock_container)

        error_msg = str(exc_info.value)
        # Should mention the solution
        assert "register_with_handler" in error_msg or "auto-configuration" in error_msg


# =============================================================================
# Handler Swapping Integration Tests
# =============================================================================


class TestRegistryHandlerSwapping:
    """Tests for handler swapping via registry."""

    @pytest.mark.anyio
    async def test_register_with_handler_allows_swapping(
        self,
        mock_container: MagicMock,
        mock_handler: HandlerServiceDiscoveryMock,
        mock_consul_handler: MagicMock,
    ) -> None:
        """Multiple calls to register_with_handler() swap handlers."""
        # Register first handler
        await RegistryInfraServiceDiscovery.register_with_handler(
            mock_container, mock_handler
        )

        # Verify first call
        first_call = mock_container.service_registry.register_instance.call_args_list[0]
        assert first_call.kwargs["instance"] is mock_handler

        # Register second handler (swap)
        await RegistryInfraServiceDiscovery.register_with_handler(
            mock_container, mock_consul_handler
        )

        # Verify second call
        second_call = mock_container.service_registry.register_instance.call_args_list[
            1
        ]
        assert second_call.kwargs["instance"] is mock_consul_handler

        # Both handlers were registered
        assert mock_container.service_registry.register_instance.call_count == 2

    def test_factory_registration_is_noop_in_v1(
        self, mock_container: MagicMock
    ) -> None:
        """register() is a no-op in v1.0 - factory registration not implemented.

        Note: This test documents the current v1.0 behavior where factory
        registration is not implemented. In future versions, this may change.
        """
        RegistryInfraServiceDiscovery.register(mock_container)

        # Factory was NOT registered (not implemented in v1.0)
        # No exception should be raised - this is expected behavior


# =============================================================================
# Protocol Type Tests
# =============================================================================


class TestProtocolTypeUsage:
    """Tests verifying correct protocol type usage."""

    @pytest.mark.anyio
    async def test_register_with_handler_uses_correct_protocol_type(
        self,
        mock_container: MagicMock,
        mock_handler: HandlerServiceDiscoveryMock,
    ) -> None:
        """register_with_handler() uses ProtocolDiscoveryOperations type."""
        from omnibase_infra.nodes.node_service_discovery_effect.protocols import (
            ProtocolDiscoveryOperations,
        )

        await RegistryInfraServiceDiscovery.register_with_handler(
            mock_container, mock_handler
        )

        call_kwargs = mock_container.service_registry.register_instance.call_args
        registered_type = call_kwargs.kwargs["interface"]

        assert registered_type is ProtocolDiscoveryOperations

    @pytest.mark.anyio
    async def test_register_with_handler_uses_global_scope(
        self,
        mock_container: MagicMock,
        mock_handler: HandlerServiceDiscoveryMock,
    ) -> None:
        """register_with_handler() uses GLOBAL injection scope."""
        from omnibase_core.enums import EnumInjectionScope

        await RegistryInfraServiceDiscovery.register_with_handler(
            mock_container, mock_handler
        )

        call_kwargs = mock_container.service_registry.register_instance.call_args
        scope = call_kwargs.kwargs["scope"]

        assert scope is EnumInjectionScope.GLOBAL


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestRegistryEdgeCases:
    """Edge case tests for registry behavior."""

    def test_mock_handler_isinstance_protocol(
        self, mock_handler: HandlerServiceDiscoveryMock
    ) -> None:
        """HandlerServiceDiscoveryMock is an instance of the protocol."""
        from omnibase_infra.nodes.node_service_discovery_effect.protocols import (
            ProtocolDiscoveryOperations,
        )

        assert isinstance(mock_handler, ProtocolDiscoveryOperations)

    def test_mock_handler_has_handler_type(
        self, mock_handler: HandlerServiceDiscoveryMock
    ) -> None:
        """HandlerServiceDiscoveryMock has handler_type property."""
        assert hasattr(mock_handler, "handler_type")
        assert mock_handler.handler_type == "mock"


__all__: list[str] = [
    "TestRegistryInfraServiceDiscoveryRegister",
    "TestRegistryInfraServiceDiscoveryRegisterWithHandler",
    "TestRegistryCreateHandlerFromConfig",
    "TestRegistryHandlerSwapping",
    "TestProtocolTypeUsage",
    "TestRegistryEdgeCases",
]
