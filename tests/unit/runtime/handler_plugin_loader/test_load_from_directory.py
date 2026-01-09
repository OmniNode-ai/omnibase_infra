# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Tests for HandlerPluginLoader.load_from_directory method.

Part of OMN-1132: Handler Plugin Loader implementation.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from .conftest import MINIMAL_HANDLER_CONTRACT_YAML, MockValidHandler


class TestHandlerPluginLoaderLoadFromDirectory:
    """Tests for load_from_directory method."""

    def test_load_multiple_handlers_from_directory(
        self, valid_contract_directory: Path
    ) -> None:
        """Test loading multiple handlers from a directory tree."""
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        loader = HandlerPluginLoader()
        handlers = loader.load_from_directory(valid_contract_directory)

        # Should find all 3 handlers (handler1, handler2, nested/deep)
        assert len(handlers) == 3

        # Verify all handler names are present
        handler_names = {h.handler_name for h in handlers}
        assert handler_names == {"handler.one", "handler.two", "handler.nested.deep"}

    def test_empty_directory_returns_empty_list(self, empty_directory: Path) -> None:
        """Test that empty directory returns empty list."""
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        loader = HandlerPluginLoader()
        handlers = loader.load_from_directory(empty_directory)

        assert handlers == []

    def test_graceful_failure_continues_on_invalid_contract(
        self, mixed_valid_invalid_directory: Path
    ) -> None:
        """Test that invalid contracts don't stop the whole load."""
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        loader = HandlerPluginLoader()
        handlers = loader.load_from_directory(mixed_valid_invalid_directory)

        # Should still load the one valid handler
        assert len(handlers) == 1
        assert handlers[0].handler_name == "valid.handler"

    def test_directory_not_found_raises_error(self, tmp_path: Path) -> None:
        """Test that nonexistent directory raises error."""
        from omnibase_infra.errors import ProtocolConfigurationError
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        nonexistent_dir = tmp_path / "does_not_exist"

        loader = HandlerPluginLoader()

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            loader.load_from_directory(nonexistent_dir)

        # Verify error message indicates directory not found
        assert "not found" in str(exc_info.value).lower()

    def test_path_is_file_not_directory_raises_error(
        self, valid_contract_path: Path
    ) -> None:
        """Test that file path instead of directory raises error."""
        from omnibase_infra.errors import ProtocolConfigurationError
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        loader = HandlerPluginLoader()

        # valid_contract_path is a file, not a directory
        with pytest.raises(ProtocolConfigurationError) as exc_info:
            loader.load_from_directory(valid_contract_path)

        # Verify error message indicates path is not a directory
        assert "not a directory" in str(exc_info.value).lower()

    def test_loads_both_handler_contract_and_contract_yaml(
        self, tmp_path: Path
    ) -> None:
        """Test that loader finds both handler_contract.yaml and contract.yaml."""
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        # Create handler_contract.yaml
        handler1_dir = tmp_path / "handler1"
        handler1_dir.mkdir()
        (handler1_dir / "handler_contract.yaml").write_text(
            MINIMAL_HANDLER_CONTRACT_YAML.format(
                handler_name="handler.contract.handler",
                handler_class="tests.unit.runtime.handler_plugin_loader.conftest.MockValidHandler",
            )
        )

        # Create contract.yaml
        handler2_dir = tmp_path / "handler2"
        handler2_dir.mkdir()
        (handler2_dir / "contract.yaml").write_text(
            MINIMAL_HANDLER_CONTRACT_YAML.format(
                handler_name="contract.yaml.handler",
                handler_class="tests.unit.runtime.handler_plugin_loader.conftest.MockValidHandler",
            )
        )

        loader = HandlerPluginLoader()
        handlers = loader.load_from_directory(tmp_path)

        assert len(handlers) == 2
        handler_names = {h.handler_name for h in handlers}
        assert handler_names == {"handler.contract.handler", "contract.yaml.handler"}

    def test_load_with_correlation_id(self, valid_contract_directory: Path) -> None:
        """Test loading with correlation_id parameter."""
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        loader = HandlerPluginLoader()
        handlers = loader.load_from_directory(
            valid_contract_directory, correlation_id="test-correlation-456"
        )

        # Should load all handlers
        assert len(handlers) == 3
