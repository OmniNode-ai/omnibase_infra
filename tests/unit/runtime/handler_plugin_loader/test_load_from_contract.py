# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Tests for HandlerPluginLoader.load_from_contract method.

Part of OMN-1132: Handler Plugin Loader implementation.
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from .conftest import (
    EMPTY_CONTRACT_YAML,
    HANDLER_CONTRACT_WITHOUT_CLASS,
    HANDLER_CONTRACT_WITHOUT_NAME,
    INVALID_YAML_SYNTAX,
    MINIMAL_CONTRACT_WITHOUT_HANDLER_TYPE,
    MINIMAL_HANDLER_CONTRACT_YAML,
    VALID_HANDLER_CONTRACT_YAML,
    MockInvalidHandler,
    MockValidHandler,
)


class TestHandlerPluginLoaderLoadFromContract:
    """Tests for load_from_contract method."""

    def test_load_valid_handler_contract(self, valid_contract_path: Path) -> None:
        """Test loading a valid handler contract."""
        from omnibase_infra.models.runtime import ModelLoadedHandler
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        loader = HandlerPluginLoader()
        result = loader.load_from_contract(valid_contract_path)

        # Verify result type
        assert isinstance(result, ModelLoadedHandler)

        # Verify handler metadata
        assert result.handler_name == "test.valid.handler"
        assert "MockValidHandler" in result.handler_class
        assert result.contract_path == valid_contract_path.resolve()
        assert "auth" in result.capability_tags
        assert "validation" in result.capability_tags

    def test_load_minimal_contract_with_handler_type(self, tmp_path: Path) -> None:
        """Test that minimal contract with handler_type loads successfully."""
        from omnibase_infra.enums import EnumHandlerTypeCategory
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        contract_file = tmp_path / "handler_contract.yaml"
        contract_file.write_text(
            MINIMAL_HANDLER_CONTRACT_YAML.format(
                handler_name="minimal.handler",
                handler_class="tests.unit.runtime.handler_plugin_loader.conftest.MockValidHandler",
            )
        )

        loader = HandlerPluginLoader()
        result = loader.load_from_contract(contract_file)

        # handler_type is set to compute in MINIMAL_HANDLER_CONTRACT_YAML
        assert result.handler_type == EnumHandlerTypeCategory.COMPUTE
        assert result.capability_tags == []  # No tags specified

    def test_reject_contract_missing_handler_type(self, tmp_path: Path) -> None:
        """Test that contract without handler_type raises error."""
        from omnibase_infra.errors import ProtocolConfigurationError
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        contract_file = tmp_path / "handler_contract.yaml"
        contract_file.write_text(
            MINIMAL_CONTRACT_WITHOUT_HANDLER_TYPE.format(
                handler_name="no.type.handler",
                handler_class="tests.unit.runtime.handler_plugin_loader.conftest.MockValidHandler",
            )
        )

        loader = HandlerPluginLoader()

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            loader.load_from_contract(contract_file)

        # Verify error message indicates missing handler_type
        error_msg = str(exc_info.value).lower()
        assert "handler_type" in error_msg

    def test_reject_missing_contract_file(self, tmp_path: Path) -> None:
        """Test error when contract file doesn't exist."""
        from omnibase_infra.errors import ProtocolConfigurationError
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        nonexistent_path = tmp_path / "does_not_exist" / "handler_contract.yaml"

        loader = HandlerPluginLoader()

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            loader.load_from_contract(nonexistent_path)

        # Verify error details via message content
        assert "not found" in str(exc_info.value).lower()

    def test_reject_contract_path_is_directory(self, tmp_path: Path) -> None:
        """Test error when contract path points to a directory, not a file."""
        from omnibase_infra.errors import ProtocolConfigurationError
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        directory = tmp_path / "some_dir"
        directory.mkdir()

        loader = HandlerPluginLoader()

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            loader.load_from_contract(directory)

        assert "not a file" in str(exc_info.value).lower()

    def test_reject_invalid_yaml(self, tmp_path: Path) -> None:
        """Test error when contract has invalid YAML syntax."""
        from omnibase_infra.enums import EnumHandlerLoaderError
        from omnibase_infra.errors import ProtocolConfigurationError
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        contract_file = tmp_path / "handler_contract.yaml"
        contract_file.write_text(INVALID_YAML_SYNTAX)

        loader = HandlerPluginLoader()

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            loader.load_from_contract(contract_file)

        # Verify error message indicates YAML parsing failure
        assert "Invalid YAML syntax" in str(exc_info.value)

        # Verify correct error code is set
        error = exc_info.value
        assert (
            error.model.context.get("loader_error")
            == EnumHandlerLoaderError.INVALID_YAML_SYNTAX.value
        )

    def test_reject_empty_contract_file(self, tmp_path: Path) -> None:
        """Test error when contract file is empty."""
        from omnibase_infra.errors import ProtocolConfigurationError
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        contract_file = tmp_path / "handler_contract.yaml"
        contract_file.write_text(EMPTY_CONTRACT_YAML)

        loader = HandlerPluginLoader()

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            loader.load_from_contract(contract_file)

        # Verify error message indicates empty contract
        assert "empty" in str(exc_info.value).lower()

    def test_reject_missing_handler_name(self, tmp_path: Path) -> None:
        """Test error when handler_name field is missing."""
        from omnibase_infra.errors import ProtocolConfigurationError
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        contract_file = tmp_path / "handler_contract.yaml"
        contract_file.write_text(HANDLER_CONTRACT_WITHOUT_NAME)

        loader = HandlerPluginLoader()

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            loader.load_from_contract(contract_file)

        # Verify error message indicates missing handler name
        error_msg = str(exc_info.value).lower()
        assert "handler_name" in error_msg or "name" in error_msg

    def test_reject_missing_handler_class(self, tmp_path: Path) -> None:
        """Test error when handler_class field is missing."""
        from omnibase_infra.errors import ProtocolConfigurationError
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        contract_file = tmp_path / "handler_contract.yaml"
        contract_file.write_text(HANDLER_CONTRACT_WITHOUT_CLASS)

        loader = HandlerPluginLoader()

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            loader.load_from_contract(contract_file)

        # Verify error message indicates missing handler class
        assert "handler_class" in str(exc_info.value).lower()

    def test_reject_invalid_handler_class_path(self, tmp_path: Path) -> None:
        """Test error when handler class cannot be imported."""
        from omnibase_infra.errors import InfraConnectionError
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        contract_file = tmp_path / "handler_contract.yaml"
        contract_file.write_text(
            MINIMAL_HANDLER_CONTRACT_YAML.format(
                handler_name="invalid.module.handler",
                handler_class="nonexistent.module.path.Handler",
            )
        )

        loader = HandlerPluginLoader()

        with pytest.raises(InfraConnectionError) as exc_info:
            loader.load_from_contract(contract_file)

        # Verify error message indicates module import failure
        error_msg = str(exc_info.value).lower()
        assert "module" in error_msg or "import" in error_msg

    def test_reject_class_not_found_in_module(self, tmp_path: Path) -> None:
        """Test error when class doesn't exist in module."""
        from omnibase_infra.errors import InfraConnectionError
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        contract_file = tmp_path / "handler_contract.yaml"
        # Use a valid module but nonexistent class
        contract_file.write_text(
            MINIMAL_HANDLER_CONTRACT_YAML.format(
                handler_name="missing.class.handler",
                handler_class="tests.unit.runtime.handler_plugin_loader.conftest.NonexistentClass",
            )
        )

        loader = HandlerPluginLoader()

        with pytest.raises(InfraConnectionError) as exc_info:
            loader.load_from_contract(contract_file)

        # Verify error message indicates class not found
        error_msg = str(exc_info.value).lower()
        assert "class" in error_msg or "not found" in error_msg

    def test_reject_handler_without_describe_method(self, tmp_path: Path) -> None:
        """Test error when handler doesn't implement ProtocolHandler (no describe)."""
        from omnibase_infra.errors import ProtocolConfigurationError
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        contract_file = tmp_path / "handler_contract.yaml"
        contract_file.write_text(
            MINIMAL_HANDLER_CONTRACT_YAML.format(
                handler_name="invalid.protocol.handler",
                handler_class="tests.unit.runtime.handler_plugin_loader.conftest.MockInvalidHandler",
            )
        )

        loader = HandlerPluginLoader()

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            loader.load_from_contract(contract_file)

        # Verify error message indicates protocol validation failure
        error_msg = str(exc_info.value).lower()
        assert "describe" in error_msg or "protocol" in error_msg

    def test_reject_unqualified_class_path(self, tmp_path: Path) -> None:
        """Test error when handler_class is not fully qualified (no dots)."""
        from omnibase_infra.errors import InfraConnectionError
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        contract_file = tmp_path / "handler_contract.yaml"
        contract_file.write_text(
            MINIMAL_HANDLER_CONTRACT_YAML.format(
                handler_name="unqualified.handler",
                handler_class="JustClassName",  # Not fully qualified
            )
        )

        loader = HandlerPluginLoader()

        with pytest.raises(InfraConnectionError) as exc_info:
            loader.load_from_contract(contract_file)

        # Verify error message indicates invalid class path
        error_msg = str(exc_info.value).lower()
        assert "class" in error_msg or "import" in error_msg or "module" in error_msg

    def test_load_with_correlation_id(self, valid_contract_path: Path) -> None:
        """Test loading with correlation_id parameter."""
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        loader = HandlerPluginLoader()
        result = loader.load_from_contract(valid_contract_path, correlation_id=uuid4())

        # Should load successfully
        assert result.handler_name == "test.valid.handler"
