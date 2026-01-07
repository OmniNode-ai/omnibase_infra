# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for RegistryInfraServiceDiscovery.

This module validates the registry functionality for service discovery
node dependencies, including handler registration and protocol factory
registration.

Test Coverage:
    - register(): Factory registration with container
    - register_with_handler(): Direct handler binding
    - _create_handler_from_config(): Configuration error handling
    - Handler swapping via registry

Related:
    - OMN-1131: Capability-oriented node architecture
    - RegistryInfraServiceDiscovery: Registry implementation
    - PR #119: Test coverage for handler swapping
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.handlers.service_discovery.handler_mock_service_discovery import (
    MockServiceDiscoveryHandler,
)
from omnibase_infra.nodes.node_service_discovery_effect.registry import (
    RegistryInfraServiceDiscovery,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_container() -> MagicMock:
    """Create a mock container with register_factory and register_instance."""
    container = MagicMock()
    container.register_factory = MagicMock()
    container.register_instance = MagicMock()
    return container


@pytest.fixture
def mock_handler() -> MockServiceDiscoveryHandler:
    """Create a MockServiceDiscoveryHandler for testing."""
    return MockServiceDiscoveryHandler()


@pytest.fixture
def mock_consul_handler() -> MagicMock:
    """Create a mock Consul handler for testing."""
    handler = MagicMock()
    handler.handler_type = "consul"
    return handler


# =============================================================================
# Factory Registration Tests
# =============================================================================


class TestRegistryInfraServiceDiscoveryRegister:
    """Tests for RegistryInfraServiceDiscovery.register() method."""

    def test_register_calls_register_factory(self, mock_container: MagicMock) -> None:
        """register() calls container.register_factory with protocol and factory."""
        RegistryInfraServiceDiscovery.register(mock_container)

        mock_container.register_factory.assert_called_once()

        # Verify it was called with the protocol type and factory function
        call_args = mock_container.register_factory.call_args
        assert call_args is not None

        # First argument should be the protocol type
        from omnibase_infra.nodes.node_service_discovery_effect.protocols import (
            ProtocolServiceDiscoveryHandler,
        )

        assert call_args[0][0] is ProtocolServiceDiscoveryHandler

    def test_register_factory_function_is_create_handler_from_config(
        self, mock_container: MagicMock
    ) -> None:
        """register() uses _create_handler_from_config as factory function."""
        RegistryInfraServiceDiscovery.register(mock_container)

        call_args = mock_container.register_factory.call_args
        assert call_args is not None

        # Second argument should be the factory function
        factory_fn = call_args[0][1]
        assert factory_fn is RegistryInfraServiceDiscovery._create_handler_from_config


# =============================================================================
# Direct Handler Registration Tests
# =============================================================================


class TestRegistryInfraServiceDiscoveryRegisterWithHandler:
    """Tests for RegistryInfraServiceDiscovery.register_with_handler() method."""

    def test_register_with_handler_calls_register_instance(
        self,
        mock_container: MagicMock,
        mock_handler: MockServiceDiscoveryHandler,
    ) -> None:
        """register_with_handler() calls container.register_instance."""
        RegistryInfraServiceDiscovery.register_with_handler(
            mock_container, mock_handler
        )

        mock_container.register_instance.assert_called_once()

    def test_register_with_handler_passes_protocol_and_handler(
        self,
        mock_container: MagicMock,
        mock_handler: MockServiceDiscoveryHandler,
    ) -> None:
        """register_with_handler() passes protocol type and handler instance."""
        RegistryInfraServiceDiscovery.register_with_handler(
            mock_container, mock_handler
        )

        call_args = mock_container.register_instance.call_args
        assert call_args is not None

        from omnibase_infra.nodes.node_service_discovery_effect.protocols import (
            ProtocolServiceDiscoveryHandler,
        )

        assert call_args[0][0] is ProtocolServiceDiscoveryHandler
        assert call_args[0][1] is mock_handler

    def test_register_with_handler_accepts_any_protocol_implementation(
        self,
        mock_container: MagicMock,
        mock_consul_handler: MagicMock,
    ) -> None:
        """register_with_handler() accepts any handler implementing the protocol."""
        RegistryInfraServiceDiscovery.register_with_handler(
            mock_container, mock_consul_handler
        )

        call_args = mock_container.register_instance.call_args
        assert call_args is not None
        assert call_args[0][1] is mock_consul_handler


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

    def test_register_with_handler_allows_swapping(
        self,
        mock_container: MagicMock,
        mock_handler: MockServiceDiscoveryHandler,
        mock_consul_handler: MagicMock,
    ) -> None:
        """Multiple calls to register_with_handler() swap handlers."""
        # Register first handler
        RegistryInfraServiceDiscovery.register_with_handler(
            mock_container, mock_handler
        )

        # Verify first call
        first_call = mock_container.register_instance.call_args_list[0]
        assert first_call[0][1] is mock_handler

        # Register second handler (swap)
        RegistryInfraServiceDiscovery.register_with_handler(
            mock_container, mock_consul_handler
        )

        # Verify second call
        second_call = mock_container.register_instance.call_args_list[1]
        assert second_call[0][1] is mock_consul_handler

        # Both handlers were registered
        assert mock_container.register_instance.call_count == 2

    def test_factory_registration_allows_lazy_handler_creation(
        self, mock_container: MagicMock
    ) -> None:
        """register() enables lazy handler creation via factory pattern."""
        RegistryInfraServiceDiscovery.register(mock_container)

        # Factory was registered
        mock_container.register_factory.assert_called_once()

        # Handler is NOT created yet (lazy)
        # The factory function would be called when resolving the dependency


# =============================================================================
# Protocol Type Tests
# =============================================================================


class TestProtocolTypeUsage:
    """Tests verifying correct protocol type usage."""

    def test_register_uses_correct_protocol_type(
        self, mock_container: MagicMock
    ) -> None:
        """register() uses ProtocolServiceDiscoveryHandler type."""
        from omnibase_infra.nodes.node_service_discovery_effect.protocols import (
            ProtocolServiceDiscoveryHandler,
        )

        RegistryInfraServiceDiscovery.register(mock_container)

        call_args = mock_container.register_factory.call_args[0]
        registered_type = call_args[0]

        assert registered_type is ProtocolServiceDiscoveryHandler

    def test_register_with_handler_uses_correct_protocol_type(
        self,
        mock_container: MagicMock,
        mock_handler: MockServiceDiscoveryHandler,
    ) -> None:
        """register_with_handler() uses ProtocolServiceDiscoveryHandler type."""
        from omnibase_infra.nodes.node_service_discovery_effect.protocols import (
            ProtocolServiceDiscoveryHandler,
        )

        RegistryInfraServiceDiscovery.register_with_handler(
            mock_container, mock_handler
        )

        call_args = mock_container.register_instance.call_args[0]
        registered_type = call_args[0]

        assert registered_type is ProtocolServiceDiscoveryHandler


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestRegistryEdgeCases:
    """Edge case tests for registry behavior."""

    def test_mock_handler_isinstance_protocol(
        self, mock_handler: MockServiceDiscoveryHandler
    ) -> None:
        """MockServiceDiscoveryHandler is an instance of the protocol."""
        from omnibase_infra.nodes.node_service_discovery_effect.protocols import (
            ProtocolServiceDiscoveryHandler,
        )

        assert isinstance(mock_handler, ProtocolServiceDiscoveryHandler)

    def test_mock_handler_has_handler_type(
        self, mock_handler: MockServiceDiscoveryHandler
    ) -> None:
        """MockServiceDiscoveryHandler has handler_type property."""
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
