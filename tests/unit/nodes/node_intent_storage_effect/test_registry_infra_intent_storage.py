# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Unit tests for RegistryInfraIntentStorage.

This module validates the registry functionality for intent storage node
dependencies, including handler registration, retrieval, protocol validation,
and storage clearing.

Test Coverage:
    - register(): Protocol metadata registration with module-level storage
    - register_handler(): Handler binding with protocol-based duck typing
    - register_handler(): ProtocolConfigurationError raised for invalid handlers
    - get_handler(): Handler retrieval by type or default
    - get_handler(): Returns None for non-existent handlers
    - clear(): Clears all handlers and metadata

Related:
    - OMN-1509: Intent classification storage and routing
    - RegistryInfraIntentStorage: Registry implementation
    - HandlerIntent: Intent handler for Memgraph graph operations

Note:
    The registry uses module-level storage (_HANDLER_STORAGE, _PROTOCOL_METADATA)
    instead of container.service_registry. Tests must clear this storage between
    runs to avoid test pollution.

Protocol-Based Validation:
    Handler validation uses duck typing per ONEX conventions. Handlers must
    implement the required methods (initialize, shutdown, execute) but do not
    need to inherit from any specific class. This enables:
    - Mock handlers for testing without subclassing
    - Alternative implementations without coupling to HandlerIntent
    - Clean dependency injection patterns
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.nodes.node_intent_storage_effect.registry import (
    RegistryInfraIntentStorage,
)

# Import module-level storage for testing
from omnibase_infra.nodes.node_intent_storage_effect.registry.registry_infra_intent_storage import (
    _HANDLER_STORAGE,
    _PROTOCOL_METADATA,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def clear_module_storage() -> None:
    """Clear module-level storage before and after each test.

    The registry uses module-level dicts for handler and metadata storage.
    This fixture ensures test isolation by clearing the storage before
    each test runs and after each test completes.
    """
    _HANDLER_STORAGE.clear()
    _PROTOCOL_METADATA.clear()
    yield
    _HANDLER_STORAGE.clear()
    _PROTOCOL_METADATA.clear()


@pytest.fixture
def mock_container() -> MagicMock:
    """Create a mock container.

    The registry uses module-level storage instead of container.service_registry,
    so the container is primarily used for API compatibility.
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
def mock_handler_intent() -> MagicMock:
    """Create a mock HandlerIntent that passes isinstance checks.

    Uses patch context to make MagicMock instances pass the isinstance
    check against HandlerIntent.
    """
    return MagicMock()


# =============================================================================
# Protocol Registration Tests
# =============================================================================


class TestRegistryInfraIntentStorageRegister:
    """Tests for RegistryInfraIntentStorage.register() method."""

    def test_register_adds_protocol_metadata(self, mock_container: MagicMock) -> None:
        """register() adds protocol metadata to module-level storage."""
        RegistryInfraIntentStorage.register(mock_container)

        handler_key = RegistryInfraIntentStorage.HANDLER_KEY
        assert handler_key in _PROTOCOL_METADATA

    def test_register_metadata_contains_required_fields(
        self, mock_container: MagicMock
    ) -> None:
        """register() metadata contains all required fields."""
        RegistryInfraIntentStorage.register(mock_container)

        metadata = _PROTOCOL_METADATA[RegistryInfraIntentStorage.HANDLER_KEY]

        assert "handler" in metadata
        assert metadata["handler"] == "HandlerIntent"
        assert "module" in metadata
        assert metadata["module"] == "omnibase_infra.handlers.handler_intent"
        assert "description" in metadata
        assert "capabilities" in metadata

    def test_register_metadata_contains_capabilities(
        self, mock_container: MagicMock
    ) -> None:
        """register() metadata includes expected capabilities."""
        RegistryInfraIntentStorage.register(mock_container)

        metadata = _PROTOCOL_METADATA[RegistryInfraIntentStorage.HANDLER_KEY]
        capabilities = metadata["capabilities"]

        assert "intent.storage" in capabilities
        assert "intent.storage.store" in capabilities
        assert "intent.storage.query_session" in capabilities
        assert "intent.storage.query_distribution" in capabilities

    def test_register_with_none_service_registry(
        self, container_with_none_registry: MagicMock
    ) -> None:
        """register() handles None service_registry gracefully."""
        # Should not raise
        RegistryInfraIntentStorage.register(container_with_none_registry)

        # Metadata should still be registered
        assert RegistryInfraIntentStorage.HANDLER_KEY in _PROTOCOL_METADATA

    def test_register_idempotent(self, mock_container: MagicMock) -> None:
        """register() can be called multiple times without error."""
        RegistryInfraIntentStorage.register(mock_container)
        RegistryInfraIntentStorage.register(mock_container)

        # Should still have the metadata
        assert RegistryInfraIntentStorage.HANDLER_KEY in _PROTOCOL_METADATA


# =============================================================================
# Handler Registration Tests
# =============================================================================


class TestRegistryInfraIntentStorageRegisterHandler:
    """Tests for RegistryInfraIntentStorage.register_handler() method."""

    def test_register_handler_adds_typed_key(
        self,
        mock_container: MagicMock,
        mock_handler_intent: MagicMock,
    ) -> None:
        """register_handler() adds handler under typed key."""
        # MagicMock passes duck typing validation (has callable attributes)
        RegistryInfraIntentStorage.register_handler(mock_container, mock_handler_intent)

        expected_key = (
            f"{RegistryInfraIntentStorage.HANDLER_KEY}."
            f"{RegistryInfraIntentStorage.DEFAULT_HANDLER_TYPE}"
        )
        assert expected_key in _HANDLER_STORAGE
        assert _HANDLER_STORAGE[expected_key] is mock_handler_intent

    def test_register_handler_sets_default(
        self,
        mock_container: MagicMock,
        mock_handler_intent: MagicMock,
    ) -> None:
        """register_handler() also sets the handler as default."""
        # MagicMock passes duck typing validation (has callable attributes)
        RegistryInfraIntentStorage.register_handler(mock_container, mock_handler_intent)

        default_key = f"{RegistryInfraIntentStorage.HANDLER_KEY}.default"
        assert default_key in _HANDLER_STORAGE
        assert _HANDLER_STORAGE[default_key] is mock_handler_intent

    def test_register_handler_raises_protocol_error_for_invalid_handler(
        self,
        mock_container: MagicMock,
    ) -> None:
        """register_handler() raises ProtocolConfigurationError for handlers
        missing required protocol methods.
        """
        invalid_handler = "not a handler"

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            RegistryInfraIntentStorage.register_handler(mock_container, invalid_handler)  # type: ignore[arg-type]

        assert "missing required protocol methods" in str(exc_info.value)
        assert "str" in str(exc_info.value)

    def test_register_handler_raises_protocol_error_for_none(
        self,
        mock_container: MagicMock,
    ) -> None:
        """register_handler() raises ProtocolConfigurationError when handler is None."""
        with pytest.raises(ProtocolConfigurationError) as exc_info:
            RegistryInfraIntentStorage.register_handler(mock_container, None)  # type: ignore[arg-type]

        assert "missing required protocol methods" in str(exc_info.value)
        assert "NoneType" in str(exc_info.value)

    def test_register_handler_accepts_duck_typed_handler(
        self,
        mock_container: MagicMock,
        mock_handler_intent: MagicMock,
    ) -> None:
        """register_handler() accepts any handler with required protocol methods.

        This verifies duck typing works correctly - MagicMock has callable
        attributes for any method name, so it passes protocol validation.
        This enables mock handlers for testing without subclassing.
        """
        # MagicMock auto-creates callable attributes, so it passes protocol checks
        # No patch needed - duck typing accepts any object with required methods
        RegistryInfraIntentStorage.register_handler(mock_container, mock_handler_intent)

        # Verify handler was registered
        retrieved = RegistryInfraIntentStorage.get_handler(mock_container)
        assert retrieved is mock_handler_intent

    def test_register_handler_with_none_service_registry(
        self,
        container_with_none_registry: MagicMock,
        mock_handler_intent: MagicMock,
    ) -> None:
        """register_handler() handles None service_registry gracefully."""
        # MagicMock passes duck typing validation (has callable attributes)
        # Should not raise
        RegistryInfraIntentStorage.register_handler(
            container_with_none_registry, mock_handler_intent
        )

        # Handler should be registered
        expected_key = (
            f"{RegistryInfraIntentStorage.HANDLER_KEY}."
            f"{RegistryInfraIntentStorage.DEFAULT_HANDLER_TYPE}"
        )
        assert expected_key in _HANDLER_STORAGE


# =============================================================================
# Handler Retrieval Tests
# =============================================================================


class TestRegistryInfraIntentStorageGetHandler:
    """Tests for RegistryInfraIntentStorage.get_handler() method."""

    def test_get_handler_by_type(
        self,
        mock_container: MagicMock,
        mock_handler_intent: MagicMock,
    ) -> None:
        """get_handler() retrieves handler by specific type."""
        # MagicMock passes duck typing validation (has callable attributes)
        RegistryInfraIntentStorage.register_handler(mock_container, mock_handler_intent)

        retrieved = RegistryInfraIntentStorage.get_handler(
            mock_container,
            handler_type=RegistryInfraIntentStorage.DEFAULT_HANDLER_TYPE,
        )

        assert retrieved is mock_handler_intent

    def test_get_handler_default(
        self,
        mock_container: MagicMock,
        mock_handler_intent: MagicMock,
    ) -> None:
        """get_handler() retrieves default handler when no type specified."""
        # MagicMock passes duck typing validation (has callable attributes)
        RegistryInfraIntentStorage.register_handler(mock_container, mock_handler_intent)

        retrieved = RegistryInfraIntentStorage.get_handler(mock_container)

        assert retrieved is mock_handler_intent

    def test_get_handler_returns_none_for_unregistered_type(
        self,
        mock_container: MagicMock,
    ) -> None:
        """get_handler() returns None for unregistered handler type."""
        result = RegistryInfraIntentStorage.get_handler(
            mock_container, handler_type="nonexistent"
        )

        assert result is None

    def test_get_handler_returns_none_for_empty_storage(
        self,
        mock_container: MagicMock,
    ) -> None:
        """get_handler() returns None when no handlers are registered."""
        result = RegistryInfraIntentStorage.get_handler(mock_container)

        assert result is None

    def test_get_handler_with_none_service_registry(
        self,
        container_with_none_registry: MagicMock,
    ) -> None:
        """get_handler() returns None when handler is not registered."""
        result = RegistryInfraIntentStorage.get_handler(
            container_with_none_registry, handler_type="memgraph"
        )

        assert result is None

    def test_get_handler_custom_type(
        self,
        mock_container: MagicMock,
        mock_handler_intent: MagicMock,
    ) -> None:
        """get_handler() retrieves handler by custom type string."""
        # Manually add handler with custom type key
        custom_key = f"{RegistryInfraIntentStorage.HANDLER_KEY}.custom_type"
        _HANDLER_STORAGE[custom_key] = mock_handler_intent

        retrieved = RegistryInfraIntentStorage.get_handler(
            mock_container, handler_type="custom_type"
        )

        assert retrieved is mock_handler_intent


# =============================================================================
# Clear Storage Tests
# =============================================================================


class TestRegistryInfraIntentStorageClear:
    """Tests for RegistryInfraIntentStorage.clear() method."""

    def test_clear_removes_all_handlers(
        self,
        mock_container: MagicMock,
        mock_handler_intent: MagicMock,
    ) -> None:
        """clear() removes all registered handlers."""
        # MagicMock passes duck typing validation (has callable attributes)
        RegistryInfraIntentStorage.register_handler(mock_container, mock_handler_intent)

        # Verify handler is registered
        assert len(_HANDLER_STORAGE) > 0

        # Clear
        RegistryInfraIntentStorage.clear()

        # Verify handlers are cleared
        assert len(_HANDLER_STORAGE) == 0

    def test_clear_removes_all_metadata(
        self,
        mock_container: MagicMock,
    ) -> None:
        """clear() removes all protocol metadata."""
        RegistryInfraIntentStorage.register(mock_container)

        # Verify metadata is registered
        assert len(_PROTOCOL_METADATA) > 0

        # Clear
        RegistryInfraIntentStorage.clear()

        # Verify metadata is cleared
        assert len(_PROTOCOL_METADATA) == 0

    def test_clear_on_empty_storage(self) -> None:
        """clear() does not raise on empty storage."""
        # Should not raise
        RegistryInfraIntentStorage.clear()

        # Verify storage is still empty
        assert len(_HANDLER_STORAGE) == 0
        assert len(_PROTOCOL_METADATA) == 0

    def test_clear_allows_re_registration(
        self,
        mock_container: MagicMock,
        mock_handler_intent: MagicMock,
    ) -> None:
        """clear() allows handlers and metadata to be re-registered."""
        # MagicMock passes duck typing validation (has callable attributes)
        RegistryInfraIntentStorage.register(mock_container)
        RegistryInfraIntentStorage.register_handler(mock_container, mock_handler_intent)

        # Clear
        RegistryInfraIntentStorage.clear()

        # Re-register
        RegistryInfraIntentStorage.register(mock_container)
        RegistryInfraIntentStorage.register_handler(mock_container, mock_handler_intent)

        # Verify re-registration worked
        assert RegistryInfraIntentStorage.HANDLER_KEY in _PROTOCOL_METADATA
        retrieved = RegistryInfraIntentStorage.get_handler(mock_container)
        assert retrieved is mock_handler_intent


# =============================================================================
# Constants Tests
# =============================================================================


class TestRegistryConstants:
    """Tests for registry constants."""

    def test_handler_key_constant(self) -> None:
        """HANDLER_KEY constant is correctly defined."""
        assert RegistryInfraIntentStorage.HANDLER_KEY == "handler_intent"

    def test_default_handler_type_constant(self) -> None:
        """DEFAULT_HANDLER_TYPE constant is correctly defined."""
        assert RegistryInfraIntentStorage.DEFAULT_HANDLER_TYPE == "memgraph"
