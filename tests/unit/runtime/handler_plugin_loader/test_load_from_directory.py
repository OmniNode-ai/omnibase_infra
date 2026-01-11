# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Tests for HandlerPluginLoader.load_from_directory method.

Part of OMN-1132: Handler Plugin Loader implementation.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from .conftest import MINIMAL_HANDLER_CONTRACT_YAML


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
        """Test loading with correlation_id parameter.

        Verifies that:
        1. Handlers load successfully when correlation_id is provided
        2. The correlation_id is propagated to error context when errors occur
        """
        from uuid import UUID

        from omnibase_infra.errors import ProtocolConfigurationError
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        # Use a valid UUID - the public API expects UUID type, not string
        test_correlation_id = UUID("12345678-1234-5678-1234-567812345678")

        loader = HandlerPluginLoader()

        # Part 1: Verify happy path - handlers load successfully with correlation_id
        handlers = loader.load_from_directory(
            valid_contract_directory, correlation_id=test_correlation_id
        )
        assert len(handlers) == 3

        # Part 2: Verify correlation_id is propagated to error context
        # Trigger an error by passing a non-existent directory
        nonexistent_dir = valid_contract_directory / "does_not_exist_subdir"

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            loader.load_from_directory(
                nonexistent_dir, correlation_id=test_correlation_id
            )

        # Verify the correlation_id was propagated to the error
        assert exc_info.value.model.correlation_id == test_correlation_id

    def test_raises_error_when_both_contract_types_in_same_directory(
        self, tmp_path: Path
    ) -> None:
        """Test that error is raised when both contract types exist in same directory.

        This verifies the fail-fast behavior for ambiguous contract configurations
        as documented in docs/patterns/handler_plugin_loader.md#contract-file-precedence.

        When both handler_contract.yaml and contract.yaml exist in the same directory,
        the loader raises ProtocolConfigurationError with AMBIGUOUS_CONTRACT_CONFIGURATION
        error code to prevent duplicate handler registrations and configuration confusion.
        """
        from omnibase_infra.enums import EnumHandlerLoaderError
        from omnibase_infra.errors import ProtocolConfigurationError
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        # Create directory with BOTH contract types (ambiguous configuration)
        ambiguous_dir = tmp_path / "ambiguous"
        ambiguous_dir.mkdir()

        # Create handler_contract.yaml
        (ambiguous_dir / "handler_contract.yaml").write_text(
            MINIMAL_HANDLER_CONTRACT_YAML.format(
                handler_name="handler.from.handler_contract",
                handler_class="tests.unit.runtime.handler_plugin_loader.conftest.MockValidHandler",
            )
        )

        # Create contract.yaml in SAME directory
        (ambiguous_dir / "contract.yaml").write_text(
            MINIMAL_HANDLER_CONTRACT_YAML.format(
                handler_name="handler.from.contract",
                handler_class="tests.unit.runtime.handler_plugin_loader.conftest.MockValidHandler",
            )
        )

        loader = HandlerPluginLoader()

        # Should raise ProtocolConfigurationError for ambiguous configuration
        with pytest.raises(ProtocolConfigurationError) as exc_info:
            loader.load_from_directory(tmp_path)

        # Verify error code is AMBIGUOUS_CONTRACT_CONFIGURATION
        assert exc_info.value.model.context.get("loader_error") == (
            EnumHandlerLoaderError.AMBIGUOUS_CONTRACT_CONFIGURATION.value
        )

        # Verify error message is clear and actionable
        error_message = str(exc_info.value)
        assert "ambiguous" in error_message.lower()
        assert "handler_contract.yaml" in error_message
        assert "contract.yaml" in error_message
        assert "ONE contract file" in error_message

    def test_no_error_when_single_contract_type_per_directory(
        self, tmp_path: Path
    ) -> None:
        """Test that no error is raised when only one contract type per directory.

        This verifies that the ambiguous contract error is only triggered for the
        specific case of both contract types in the same directory, not for normal
        configurations where handler_contract.yaml and contract.yaml are in
        separate directories.
        """
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        # Create separate directories with one contract type each (correct config)
        dir1 = tmp_path / "handler1"
        dir1.mkdir()
        (dir1 / "handler_contract.yaml").write_text(
            MINIMAL_HANDLER_CONTRACT_YAML.format(
                handler_name="handler.one",
                handler_class="tests.unit.runtime.handler_plugin_loader.conftest.MockValidHandler",
            )
        )

        dir2 = tmp_path / "handler2"
        dir2.mkdir()
        (dir2 / "contract.yaml").write_text(
            MINIMAL_HANDLER_CONTRACT_YAML.format(
                handler_name="handler.two",
                handler_class="tests.unit.runtime.handler_plugin_loader.conftest.MockValidHandler",
            )
        )

        loader = HandlerPluginLoader()

        # Should succeed without raising any errors
        handlers = loader.load_from_directory(tmp_path)

        # Verify both handlers were loaded
        assert len(handlers) == 2
        handler_names = {h.handler_name for h in handlers}
        assert handler_names == {"handler.one", "handler.two"}
