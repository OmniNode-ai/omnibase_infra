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
        """Test loading with correlation_id parameter.

        Verifies that:
        1. Handlers load successfully when correlation_id is provided
        2. The correlation_id is propagated to error context when errors occur
        """
        from uuid import UUID

        from omnibase_infra.errors import ProtocolConfigurationError
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        # Use a valid UUID string that can be parsed by ModelInfraErrorContext
        test_correlation_id = UUID("12345678-1234-5678-1234-567812345678")
        correlation_id_str = str(test_correlation_id)

        loader = HandlerPluginLoader()

        # Part 1: Verify happy path - handlers load successfully with correlation_id
        handlers = loader.load_from_directory(
            valid_contract_directory, correlation_id=correlation_id_str
        )
        assert len(handlers) == 3

        # Part 2: Verify correlation_id is propagated to error context
        # Trigger an error by passing a non-existent directory
        nonexistent_dir = valid_contract_directory / "does_not_exist_subdir"

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            loader.load_from_directory(
                nonexistent_dir, correlation_id=correlation_id_str
            )

        # Verify the correlation_id was propagated to the error
        assert exc_info.value.model.correlation_id == test_correlation_id

    def test_correlation_id_propagated_to_helper_methods(self, tmp_path: Path) -> None:
        """Test that correlation_id is propagated to internal helper methods.

        Verifies correlation_id propagation for:
        - _extract_handler_name (via missing handler_name)
        - _extract_handler_class (via missing handler_class)
        - _extract_handler_type (via missing handler_type)

        Per ONEX coding guidelines: "Always propagate correlation_id from
        incoming requests; include in all error context."
        """
        from uuid import UUID

        from omnibase_infra.errors import ProtocolConfigurationError
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        from .conftest import (
            HANDLER_CONTRACT_WITHOUT_CLASS,
            HANDLER_CONTRACT_WITHOUT_NAME,
            MINIMAL_CONTRACT_WITHOUT_HANDLER_TYPE,
        )

        test_correlation_id = UUID("abcdef12-3456-7890-abcd-ef1234567890")
        correlation_id_str = str(test_correlation_id)

        loader = HandlerPluginLoader()

        # Test 1: Verify correlation_id propagated to _extract_handler_name
        # Error triggered by missing handler_name field in contract
        handler_dir = tmp_path / "missing_name"
        handler_dir.mkdir()
        contract_path = handler_dir / "handler_contract.yaml"
        contract_path.write_text(HANDLER_CONTRACT_WITHOUT_NAME)

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            loader.load_from_contract(contract_path, correlation_id=correlation_id_str)

        assert exc_info.value.model.correlation_id == test_correlation_id
        # Verify error is about missing handler_name (from _extract_handler_name)
        assert "handler_name" in str(exc_info.value).lower()

        # Test 2: Verify correlation_id propagated to _extract_handler_class
        # Error triggered by missing handler_class field in contract
        handler_dir = tmp_path / "missing_class"
        handler_dir.mkdir()
        contract_path = handler_dir / "handler_contract.yaml"
        contract_path.write_text(HANDLER_CONTRACT_WITHOUT_CLASS)

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            loader.load_from_contract(contract_path, correlation_id=correlation_id_str)

        assert exc_info.value.model.correlation_id == test_correlation_id
        # Verify error is about missing handler_class (from _extract_handler_class)
        assert "handler_class" in str(exc_info.value).lower()

        # Test 3: Verify correlation_id propagated to _extract_handler_type
        # Error triggered by missing handler_type field in contract
        handler_dir = tmp_path / "missing_type"
        handler_dir.mkdir()
        contract_path = handler_dir / "handler_contract.yaml"
        contract_path.write_text(
            MINIMAL_CONTRACT_WITHOUT_HANDLER_TYPE.format(
                handler_name="test.handler",
                handler_class="tests.unit.runtime.handler_plugin_loader.conftest.MockValidHandler",
            )
        )

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            loader.load_from_contract(contract_path, correlation_id=correlation_id_str)

        assert exc_info.value.model.correlation_id == test_correlation_id
        # Verify error is about missing handler_type (from _extract_handler_type)
        assert "handler_type" in str(exc_info.value).lower()

    def test_correlation_id_propagated_to_import_handler_class(
        self, tmp_path: Path
    ) -> None:
        """Test that correlation_id is propagated to _import_handler_class.

        Triggers an import error by specifying a non-existent module.
        """
        from uuid import UUID

        from omnibase_infra.errors import InfraConnectionError
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        from .conftest import MINIMAL_HANDLER_CONTRACT_YAML

        test_correlation_id = UUID("fedcba98-7654-3210-fedc-ba9876543210")
        correlation_id_str = str(test_correlation_id)

        loader = HandlerPluginLoader()

        # Create contract with non-existent module
        handler_dir = tmp_path / "bad_import"
        handler_dir.mkdir()
        contract_path = handler_dir / "handler_contract.yaml"
        contract_path.write_text(
            MINIMAL_HANDLER_CONTRACT_YAML.format(
                handler_name="bad.import.handler",
                handler_class="nonexistent.module.path.Handler",
            )
        )

        with pytest.raises(InfraConnectionError) as exc_info:
            loader.load_from_contract(contract_path, correlation_id=correlation_id_str)

        assert exc_info.value.model.correlation_id == test_correlation_id
        # Verify error is about module not found (from _import_handler_class)
        assert "module not found" in str(exc_info.value).lower()
