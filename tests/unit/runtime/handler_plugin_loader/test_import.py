# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Tests for HandlerPluginLoader import and instantiation.

Part of OMN-1132: Handler Plugin Loader implementation.
"""

from __future__ import annotations


class TestHandlerPluginLoaderImport:
    """Tests for HandlerPluginLoader import and instantiation."""

    def test_handler_plugin_loader_can_be_imported(self) -> None:
        """HandlerPluginLoader should be importable from omnibase_infra.runtime."""
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        assert HandlerPluginLoader is not None

    def test_handler_plugin_loader_implements_protocol(self) -> None:
        """HandlerPluginLoader should implement ProtocolHandlerPluginLoader."""
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader
        from omnibase_infra.runtime.protocol_handler_plugin_loader import (
            ProtocolHandlerPluginLoader,
        )

        loader = HandlerPluginLoader()

        # Protocol compliance via duck typing (ONEX convention)
        assert hasattr(loader, "load_from_contract")
        assert hasattr(loader, "load_from_directory")
        assert hasattr(loader, "discover_and_load")
        assert callable(loader.load_from_contract)
        assert callable(loader.load_from_directory)
        assert callable(loader.discover_and_load)

        # Runtime checkable protocol verification
        assert isinstance(loader, ProtocolHandlerPluginLoader)

    def test_constants_are_exported(self) -> None:
        """Module constants should be exported."""
        from omnibase_infra.runtime.handler_plugin_loader import (
            CONTRACT_YAML_FILENAME,
            HANDLER_CONTRACT_FILENAME,
            MAX_CONTRACT_SIZE,
        )

        assert HANDLER_CONTRACT_FILENAME == "handler_contract.yaml"
        assert CONTRACT_YAML_FILENAME == "contract.yaml"
        assert MAX_CONTRACT_SIZE == 10 * 1024 * 1024  # 10MB
