# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for RegistryInfraRegistrationStorage.

This module validates the registry functionality for registration storage
node dependencies, including handler registration, retrieval, and protocol
metadata registration.

Test Coverage:
    - register(): Protocol metadata registration with container
    - register_handler(): Handler binding with type-based keys
    - get_handler(): Handler retrieval by type or default
    - Handler swapping via registry

Related:
    - OMN-1131: Capability-oriented node architecture
    - RegistryInfraRegistrationStorage: Registry implementation
    - PR #119: Test coverage for handler swapping
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from omnibase_infra.handlers.registration_storage.handler_mock_registration_storage import (
    MockRegistrationStorageHandler,
)
from omnibase_infra.nodes.node_registration_storage_effect.registry import (
    RegistryInfraRegistrationStorage,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_container() -> MagicMock:
    """Create a mock container with service_registry dict.

    The real ModelONEXContainer has service_registry=None when the
    ServiceRegistry module is not installed. We mock it to test
    the registry behavior.
    """
    container = MagicMock()
    container.service_registry = {}
    return container


@pytest.fixture
def container_with_none_registry() -> MagicMock:
    """Create a mock container with service_registry=None."""
    container = MagicMock()
    container.service_registry = None
    return container


@pytest.fixture
def mock_handler() -> MockRegistrationStorageHandler:
    """Create a MockRegistrationStorageHandler for testing."""
    return MockRegistrationStorageHandler()


@pytest.fixture
def mock_postgres_handler() -> MagicMock:
    """Create a mock PostgreSQL handler that implements the protocol.

    Uses spec to ensure the mock satisfies isinstance checks against
    the @runtime_checkable protocol, simulating a PostgreSQL handler
    without requiring actual database connection.
    """
    from omnibase_infra.nodes.node_registration_storage_effect.protocols import (
        ProtocolRegistrationStorageHandler,
    )

    handler = MagicMock(spec=ProtocolRegistrationStorageHandler)
    handler.handler_type = "postgresql"
    return handler


# =============================================================================
# Protocol Registration Tests
# =============================================================================


class TestRegistryInfraRegistrationStorageRegister:
    """Tests for RegistryInfraRegistrationStorage.register() method."""

    def test_register_adds_protocol_metadata(self, mock_container: MagicMock) -> None:
        """register() adds protocol metadata to container service_registry."""
        RegistryInfraRegistrationStorage.register(mock_container)

        protocol_key = RegistryInfraRegistrationStorage.PROTOCOL_KEY
        assert protocol_key in mock_container.service_registry

    def test_register_metadata_contains_required_fields(
        self, mock_container: MagicMock
    ) -> None:
        """register() metadata contains all required fields."""
        RegistryInfraRegistrationStorage.register(mock_container)

        metadata = mock_container.service_registry[
            RegistryInfraRegistrationStorage.PROTOCOL_KEY
        ]

        assert "protocol" in metadata
        assert metadata["protocol"] == "ProtocolRegistrationStorageHandler"
        assert "module" in metadata
        assert "pluggable" in metadata
        assert metadata["pluggable"] is True
        assert "implementations" in metadata
        assert "postgresql" in metadata["implementations"]

    def test_register_with_none_service_registry(
        self, container_with_none_registry: MagicMock
    ) -> None:
        """register() handles None service_registry gracefully."""
        # Should not raise
        RegistryInfraRegistrationStorage.register(container_with_none_registry)

    def test_register_idempotent(self, mock_container: MagicMock) -> None:
        """register() can be called multiple times without error."""
        RegistryInfraRegistrationStorage.register(mock_container)
        RegistryInfraRegistrationStorage.register(mock_container)

        # Should still have the metadata
        assert (
            RegistryInfraRegistrationStorage.PROTOCOL_KEY
            in mock_container.service_registry
        )


# =============================================================================
# Handler Registration Tests
# =============================================================================


class TestRegistryInfraRegistrationStorageRegisterHandler:
    """Tests for RegistryInfraRegistrationStorage.register_handler() method."""

    def test_register_handler_adds_typed_key(
        self,
        mock_container: MagicMock,
        mock_handler: MockRegistrationStorageHandler,
    ) -> None:
        """register_handler() adds handler under typed key."""
        RegistryInfraRegistrationStorage.register_handler(mock_container, mock_handler)

        expected_key = f"{RegistryInfraRegistrationStorage.PROTOCOL_KEY}.mock"
        assert expected_key in mock_container.service_registry
        assert mock_container.service_registry[expected_key] is mock_handler

    def test_register_handler_sets_default_for_postgresql(
        self,
        mock_container: MagicMock,
        mock_postgres_handler: MagicMock,
    ) -> None:
        """register_handler() sets default key for postgresql handler type."""
        RegistryInfraRegistrationStorage.register_handler(
            mock_container, mock_postgres_handler
        )

        # Typed key should exist
        typed_key = f"{RegistryInfraRegistrationStorage.PROTOCOL_KEY}.postgresql"
        assert typed_key in mock_container.service_registry

        # Default key should also exist for postgresql
        default_key = f"{RegistryInfraRegistrationStorage.PROTOCOL_KEY}.default"
        assert default_key in mock_container.service_registry
        assert mock_container.service_registry[default_key] is mock_postgres_handler

    def test_register_handler_no_default_for_non_postgresql(
        self,
        mock_container: MagicMock,
        mock_handler: MockRegistrationStorageHandler,
    ) -> None:
        """register_handler() does not set default for non-postgresql handlers."""
        RegistryInfraRegistrationStorage.register_handler(mock_container, mock_handler)

        # Typed key should exist
        typed_key = f"{RegistryInfraRegistrationStorage.PROTOCOL_KEY}.mock"
        assert typed_key in mock_container.service_registry

        # Default key should NOT exist for mock handler
        default_key = f"{RegistryInfraRegistrationStorage.PROTOCOL_KEY}.default"
        assert default_key not in mock_container.service_registry

    def test_register_handler_with_none_service_registry(
        self,
        container_with_none_registry: MagicMock,
        mock_handler: MockRegistrationStorageHandler,
    ) -> None:
        """register_handler() handles None service_registry gracefully."""
        # Should not raise
        RegistryInfraRegistrationStorage.register_handler(
            container_with_none_registry, mock_handler
        )

    def test_register_multiple_handlers(
        self,
        mock_container: MagicMock,
        mock_handler: MockRegistrationStorageHandler,
        mock_postgres_handler: MagicMock,
    ) -> None:
        """register_handler() can register multiple handlers of different types."""
        RegistryInfraRegistrationStorage.register_handler(mock_container, mock_handler)
        RegistryInfraRegistrationStorage.register_handler(
            mock_container, mock_postgres_handler
        )

        mock_key = f"{RegistryInfraRegistrationStorage.PROTOCOL_KEY}.mock"
        postgres_key = f"{RegistryInfraRegistrationStorage.PROTOCOL_KEY}.postgresql"

        assert mock_key in mock_container.service_registry
        assert postgres_key in mock_container.service_registry
        assert mock_container.service_registry[mock_key] is mock_handler
        assert mock_container.service_registry[postgres_key] is mock_postgres_handler


# =============================================================================
# Handler Retrieval Tests
# =============================================================================


class TestRegistryInfraRegistrationStorageGetHandler:
    """Tests for RegistryInfraRegistrationStorage.get_handler() method."""

    def test_get_handler_by_type(
        self,
        mock_container: MagicMock,
        mock_handler: MockRegistrationStorageHandler,
    ) -> None:
        """get_handler() retrieves handler by type."""
        RegistryInfraRegistrationStorage.register_handler(mock_container, mock_handler)

        retrieved = RegistryInfraRegistrationStorage.get_handler(
            mock_container, handler_type="mock"
        )

        assert retrieved is mock_handler

    def test_get_handler_default(
        self,
        mock_container: MagicMock,
        mock_postgres_handler: MagicMock,
    ) -> None:
        """get_handler() retrieves default handler when no type specified."""
        RegistryInfraRegistrationStorage.register_handler(
            mock_container, mock_postgres_handler
        )

        retrieved = RegistryInfraRegistrationStorage.get_handler(mock_container)

        assert retrieved is mock_postgres_handler

    def test_get_handler_returns_none_for_unregistered(
        self,
        mock_container: MagicMock,
    ) -> None:
        """get_handler() returns None for unregistered handler type."""
        result = RegistryInfraRegistrationStorage.get_handler(
            mock_container, handler_type="nonexistent"
        )

        assert result is None

    def test_get_handler_returns_none_for_no_default(
        self,
        mock_container: MagicMock,
        mock_handler: MockRegistrationStorageHandler,
    ) -> None:
        """get_handler() returns None when no default is set."""
        # Register mock handler (not postgresql, so no default)
        RegistryInfraRegistrationStorage.register_handler(mock_container, mock_handler)

        # Request default (should return None since mock doesn't set default)
        result = RegistryInfraRegistrationStorage.get_handler(mock_container)

        assert result is None

    def test_get_handler_with_none_service_registry(
        self,
        container_with_none_registry: MagicMock,
    ) -> None:
        """get_handler() returns None when service_registry is None."""
        result = RegistryInfraRegistrationStorage.get_handler(
            container_with_none_registry, handler_type="mock"
        )

        assert result is None


# =============================================================================
# Handler Swapping Integration Tests
# =============================================================================


class TestRegistryHandlerSwapping:
    """Tests for handler swapping via registry."""

    def test_swap_handlers_at_runtime(
        self,
        mock_container: MagicMock,
    ) -> None:
        """Handlers can be swapped by re-registering with same type."""
        # Register first mock handler
        handler1 = MockRegistrationStorageHandler()
        RegistryInfraRegistrationStorage.register_handler(mock_container, handler1)

        retrieved1 = RegistryInfraRegistrationStorage.get_handler(
            mock_container, handler_type="mock"
        )
        assert retrieved1 is handler1

        # Swap with second mock handler
        handler2 = MockRegistrationStorageHandler()
        RegistryInfraRegistrationStorage.register_handler(mock_container, handler2)

        retrieved2 = RegistryInfraRegistrationStorage.get_handler(
            mock_container, handler_type="mock"
        )
        assert retrieved2 is handler2
        assert retrieved2 is not handler1

    def test_protocol_key_constant(self) -> None:
        """PROTOCOL_KEY constant is correctly defined."""
        assert (
            RegistryInfraRegistrationStorage.PROTOCOL_KEY
            == "protocol_registration_storage_handler"
        )

    def test_default_handler_type_constant(self) -> None:
        """DEFAULT_HANDLER_TYPE constant is correctly defined."""
        assert RegistryInfraRegistrationStorage.DEFAULT_HANDLER_TYPE == "postgresql"


__all__: list[str] = [
    "TestRegistryInfraRegistrationStorageRegister",
    "TestRegistryInfraRegistrationStorageRegisterHandler",
    "TestRegistryInfraRegistrationStorageGetHandler",
    "TestRegistryHandlerSwapping",
]
