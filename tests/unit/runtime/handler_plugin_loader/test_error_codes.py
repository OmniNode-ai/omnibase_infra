# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Tests for distinct error codes in HandlerPluginLoader.

Part of OMN-1132: Handler Plugin Loader implementation.

These tests verify that the handler plugin loader produces distinct error codes
for different failure scenarios, enabling precise programmatic error handling.

Error Code Ranges:
    - 001-009: File-level errors (load_from_contract)
    - 010-019: Import errors (handler class loading)
    - 020-029: Directory-level errors (load_from_directory)
    - 030-039: Pattern errors (discover_and_load)
    - 040-049: Configuration errors (ambiguous configurations)
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from .conftest import MINIMAL_HANDLER_CONTRACT_YAML


class TestFileNotFoundVsNotAFile:
    """Tests for distinct error codes: FILE_NOT_FOUND (001) vs NOT_A_FILE (007).

    These tests verify that the loader correctly distinguishes between:
    - HANDLER_LOADER_001: Path does not exist at all
    - HANDLER_LOADER_007: Path exists but is not a regular file (e.g., directory)
    """

    def test_file_not_found_uses_001_error_code(self, tmp_path: Path) -> None:
        """Test that non-existent path raises error with HANDLER_LOADER_001."""
        from omnibase_infra.enums import EnumHandlerLoaderError
        from omnibase_infra.errors import ProtocolConfigurationError
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        # Path that doesn't exist
        non_existent_path = tmp_path / "does_not_exist" / "handler_contract.yaml"

        loader = HandlerPluginLoader()

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            loader.load_from_contract(non_existent_path)

        # Verify error code is FILE_NOT_FOUND
        error = exc_info.value
        assert (
            error.model.context.get("loader_error")
            == EnumHandlerLoaderError.FILE_NOT_FOUND.value
        )
        assert "not found" in str(error).lower()

    def test_directory_path_uses_007_error_code(self, tmp_path: Path) -> None:
        """Test that directory path raises error with HANDLER_LOADER_007."""
        from omnibase_infra.enums import EnumHandlerLoaderError
        from omnibase_infra.errors import ProtocolConfigurationError
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        # Create a directory (not a file)
        directory_path = tmp_path / "handlers"
        directory_path.mkdir()

        loader = HandlerPluginLoader()

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            loader.load_from_contract(directory_path)

        # Verify error code is NOT_A_FILE
        error = exc_info.value
        assert (
            error.model.context.get("loader_error")
            == EnumHandlerLoaderError.NOT_A_FILE.value
        )
        assert "not a file" in str(error).lower()

    def test_error_codes_are_distinct(self) -> None:
        """Test that FILE_NOT_FOUND and NOT_A_FILE have different values."""
        from omnibase_infra.enums import EnumHandlerLoaderError

        assert (
            EnumHandlerLoaderError.FILE_NOT_FOUND.value
            != EnumHandlerLoaderError.NOT_A_FILE.value
        )
        assert EnumHandlerLoaderError.FILE_NOT_FOUND.value == "HANDLER_LOADER_001"
        assert EnumHandlerLoaderError.NOT_A_FILE.value == "HANDLER_LOADER_007"


class TestFileOperationErrorCodes:
    """Tests for file operation error codes (008, 009)."""

    def test_stat_error_uses_009_error_code(self, tmp_path: Path) -> None:
        """Test that stat failure raises error with HANDLER_LOADER_009."""
        from omnibase_infra.enums import EnumHandlerLoaderError
        from omnibase_infra.errors import ProtocolConfigurationError
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        # Create a valid file
        contract_file = tmp_path / "handler_contract.yaml"
        contract_file.write_text(
            MINIMAL_HANDLER_CONTRACT_YAML.format(
                handler_name="test.handler",
                handler_class="tests.unit.runtime.handler_plugin_loader.conftest.MockValidHandler",
            )
        )

        loader = HandlerPluginLoader()

        # Get the original stat result to return for initial calls
        original_stat_result = contract_file.stat()
        call_count = [0]

        def mock_stat(self, *args, **kwargs):
            call_count[0] += 1
            # First two calls are for exists() and is_file(), third is for size check
            if call_count[0] <= 2:
                return original_stat_result
            raise OSError("Permission denied")

        with patch.object(Path, "stat", mock_stat):
            with pytest.raises(ProtocolConfigurationError) as exc_info:
                loader.load_from_contract(contract_file)

        # Verify error code is FILE_STAT_ERROR
        error = exc_info.value
        assert (
            error.model.context.get("loader_error")
            == EnumHandlerLoaderError.FILE_STAT_ERROR.value
        )
        assert "stat" in str(error).lower()

    def test_read_error_uses_008_error_code(self, tmp_path: Path) -> None:
        """Test that read failure raises error with HANDLER_LOADER_008."""
        from omnibase_infra.enums import EnumHandlerLoaderError
        from omnibase_infra.errors import ProtocolConfigurationError
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        # Create a valid file
        contract_file = tmp_path / "handler_contract.yaml"
        contract_file.write_text(
            MINIMAL_HANDLER_CONTRACT_YAML.format(
                handler_name="test.handler",
                handler_class="tests.unit.runtime.handler_plugin_loader.conftest.MockValidHandler",
            )
        )

        loader = HandlerPluginLoader()

        # Mock open to raise OSError (not during stat, but during open)
        def mock_open(self, *args, **kwargs):
            raise OSError("Read error")

        with patch.object(Path, "open", mock_open):
            with pytest.raises(ProtocolConfigurationError) as exc_info:
                loader.load_from_contract(contract_file)

        # Verify error code is FILE_READ_ERROR
        error = exc_info.value
        assert (
            error.model.context.get("loader_error")
            == EnumHandlerLoaderError.FILE_READ_ERROR.value
        )
        assert "read" in str(error).lower()


class TestDirectoryErrorCodes:
    """Tests for directory operation error codes (020, 022)."""

    def test_directory_not_found_uses_020_error_code(self, tmp_path: Path) -> None:
        """Test that non-existent directory raises error with HANDLER_LOADER_020."""
        from omnibase_infra.enums import EnumHandlerLoaderError
        from omnibase_infra.errors import ProtocolConfigurationError
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        # Path that doesn't exist
        non_existent_dir = tmp_path / "does_not_exist"

        loader = HandlerPluginLoader()

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            loader.load_from_directory(non_existent_dir)

        # Verify error code is DIRECTORY_NOT_FOUND
        error = exc_info.value
        assert (
            error.model.context.get("loader_error")
            == EnumHandlerLoaderError.DIRECTORY_NOT_FOUND.value
        )
        assert "not found" in str(error).lower()

    def test_file_as_directory_uses_022_error_code(self, tmp_path: Path) -> None:
        """Test that file path raises error with HANDLER_LOADER_022 when used as directory."""
        from omnibase_infra.enums import EnumHandlerLoaderError
        from omnibase_infra.errors import ProtocolConfigurationError
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        # Create a file (not a directory)
        file_path = tmp_path / "not_a_directory.txt"
        file_path.write_text("This is a file, not a directory")

        loader = HandlerPluginLoader()

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            loader.load_from_directory(file_path)

        # Verify error code is NOT_A_DIRECTORY
        error = exc_info.value
        assert (
            error.model.context.get("loader_error")
            == EnumHandlerLoaderError.NOT_A_DIRECTORY.value
        )
        assert "not a directory" in str(error).lower()


class TestEnumProperties:
    """Tests for EnumHandlerLoaderError enum properties."""

    def test_file_error_property(self) -> None:
        """Test is_file_error property returns True for file errors."""
        from omnibase_infra.enums import EnumHandlerLoaderError

        # All file errors (001-009) should return True
        assert EnumHandlerLoaderError.FILE_NOT_FOUND.is_file_error
        assert EnumHandlerLoaderError.INVALID_YAML_SYNTAX.is_file_error
        assert EnumHandlerLoaderError.SCHEMA_VALIDATION_FAILED.is_file_error
        assert EnumHandlerLoaderError.NOT_A_FILE.is_file_error
        assert EnumHandlerLoaderError.FILE_READ_ERROR.is_file_error
        assert EnumHandlerLoaderError.FILE_STAT_ERROR.is_file_error

        # Non-file errors should return False
        assert not EnumHandlerLoaderError.MODULE_NOT_FOUND.is_file_error
        assert not EnumHandlerLoaderError.DIRECTORY_NOT_FOUND.is_file_error

    def test_import_error_property(self) -> None:
        """Test is_import_error property returns True for import errors."""
        from omnibase_infra.enums import EnumHandlerLoaderError

        # All import errors (010-019) should return True
        assert EnumHandlerLoaderError.MODULE_NOT_FOUND.is_import_error
        assert EnumHandlerLoaderError.CLASS_NOT_FOUND.is_import_error
        assert EnumHandlerLoaderError.IMPORT_ERROR.is_import_error

        # Non-import errors should return False
        assert not EnumHandlerLoaderError.FILE_NOT_FOUND.is_import_error
        assert not EnumHandlerLoaderError.DIRECTORY_NOT_FOUND.is_import_error

    def test_directory_error_property(self) -> None:
        """Test is_directory_error property returns True for directory errors."""
        from omnibase_infra.enums import EnumHandlerLoaderError

        # All directory errors (020-029) should return True
        assert EnumHandlerLoaderError.DIRECTORY_NOT_FOUND.is_directory_error
        assert EnumHandlerLoaderError.PERMISSION_DENIED.is_directory_error
        assert EnumHandlerLoaderError.NOT_A_DIRECTORY.is_directory_error

        # Non-directory errors should return False
        assert not EnumHandlerLoaderError.FILE_NOT_FOUND.is_directory_error
        assert not EnumHandlerLoaderError.MODULE_NOT_FOUND.is_directory_error

    def test_pattern_error_property(self) -> None:
        """Test is_pattern_error property returns True for pattern errors."""
        from omnibase_infra.enums import EnumHandlerLoaderError

        # All pattern errors (030-039) should return True
        assert EnumHandlerLoaderError.EMPTY_PATTERNS_LIST.is_pattern_error
        assert EnumHandlerLoaderError.INVALID_GLOB_PATTERN.is_pattern_error

        # Non-pattern errors should return False
        assert not EnumHandlerLoaderError.FILE_NOT_FOUND.is_pattern_error
        assert not EnumHandlerLoaderError.DIRECTORY_NOT_FOUND.is_pattern_error

    def test_configuration_error_property(self) -> None:
        """Test is_configuration_error property returns True for configuration errors."""
        from omnibase_infra.enums import EnumHandlerLoaderError

        # All configuration errors (040-049) should return True
        assert EnumHandlerLoaderError.AMBIGUOUS_CONTRACT_CONFIGURATION.is_configuration_error

        # Non-configuration errors should return False
        assert not EnumHandlerLoaderError.FILE_NOT_FOUND.is_configuration_error
        assert not EnumHandlerLoaderError.DIRECTORY_NOT_FOUND.is_configuration_error
        assert not EnumHandlerLoaderError.EMPTY_PATTERNS_LIST.is_configuration_error


class TestAmbiguousContractConfigurationError:
    """Tests for AMBIGUOUS_CONTRACT_CONFIGURATION error code (040).

    This error is raised when both handler_contract.yaml and contract.yaml
    exist in the same directory, which is an ambiguous configuration that
    could lead to duplicate handler registrations.
    """

    def test_ambiguous_contract_uses_040_error_code(self, tmp_path: Path) -> None:
        """Test that ambiguous contract configuration uses HANDLER_LOADER_040."""
        from omnibase_infra.enums import EnumHandlerLoaderError
        from omnibase_infra.errors import ProtocolConfigurationError
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        # Create directory with BOTH contract types (ambiguous)
        ambiguous_dir = tmp_path / "ambiguous"
        ambiguous_dir.mkdir()

        (ambiguous_dir / "handler_contract.yaml").write_text(
            MINIMAL_HANDLER_CONTRACT_YAML.format(
                handler_name="handler.one",
                handler_class="tests.unit.runtime.handler_plugin_loader.conftest.MockValidHandler",
            )
        )
        (ambiguous_dir / "contract.yaml").write_text(
            MINIMAL_HANDLER_CONTRACT_YAML.format(
                handler_name="handler.two",
                handler_class="tests.unit.runtime.handler_plugin_loader.conftest.MockValidHandler",
            )
        )

        loader = HandlerPluginLoader()

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            loader.load_from_directory(tmp_path)

        # Verify error code is AMBIGUOUS_CONTRACT_CONFIGURATION
        error = exc_info.value
        assert (
            error.model.context.get("loader_error")
            == EnumHandlerLoaderError.AMBIGUOUS_CONTRACT_CONFIGURATION.value
        )
        assert (
            EnumHandlerLoaderError.AMBIGUOUS_CONTRACT_CONFIGURATION.value
            == "HANDLER_LOADER_040"
        )

    def test_ambiguous_contract_includes_correlation_id(self, tmp_path: Path) -> None:
        """Test that AMBIGUOUS_CONTRACT_CONFIGURATION error includes correlation_id."""
        from uuid import UUID

        from omnibase_infra.errors import ProtocolConfigurationError
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        # Create directory with BOTH contract types (ambiguous)
        ambiguous_dir = tmp_path / "ambiguous"
        ambiguous_dir.mkdir()

        (ambiguous_dir / "handler_contract.yaml").write_text(
            MINIMAL_HANDLER_CONTRACT_YAML.format(
                handler_name="handler.one",
                handler_class="tests.unit.runtime.handler_plugin_loader.conftest.MockValidHandler",
            )
        )
        (ambiguous_dir / "contract.yaml").write_text(
            MINIMAL_HANDLER_CONTRACT_YAML.format(
                handler_name="handler.two",
                handler_class="tests.unit.runtime.handler_plugin_loader.conftest.MockValidHandler",
            )
        )

        test_correlation_id = UUID("12345678-1234-5678-1234-567812345678")
        loader = HandlerPluginLoader()

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            loader.load_from_directory(tmp_path, correlation_id=test_correlation_id)

        # Verify correlation_id is in error
        error = exc_info.value
        assert error.model.correlation_id == test_correlation_id

    def test_ambiguous_contract_error_message_is_actionable(
        self, tmp_path: Path
    ) -> None:
        """Test that error message explains the problem and solution."""
        from omnibase_infra.errors import ProtocolConfigurationError
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        # Create directory with BOTH contract types (ambiguous)
        ambiguous_dir = tmp_path / "ambiguous"
        ambiguous_dir.mkdir()

        (ambiguous_dir / "handler_contract.yaml").write_text(
            MINIMAL_HANDLER_CONTRACT_YAML.format(
                handler_name="handler.one",
                handler_class="tests.unit.runtime.handler_plugin_loader.conftest.MockValidHandler",
            )
        )
        (ambiguous_dir / "contract.yaml").write_text(
            MINIMAL_HANDLER_CONTRACT_YAML.format(
                handler_name="handler.two",
                handler_class="tests.unit.runtime.handler_plugin_loader.conftest.MockValidHandler",
            )
        )

        loader = HandlerPluginLoader()

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            loader.load_from_directory(tmp_path)

        error_message = str(exc_info.value)

        # Error should explain what was detected
        assert "handler_contract.yaml" in error_message
        assert "contract.yaml" in error_message
        assert "ambiguous" in error_message.lower()

        # Error should explain how to fix
        assert "ONE contract file" in error_message


class TestCorrelationIdInErrors:
    """Tests for correlation_id propagation in error context."""

    def test_file_not_found_includes_correlation_id(self, tmp_path: Path) -> None:
        """Test that FILE_NOT_FOUND error includes correlation_id."""
        from uuid import uuid4

        from omnibase_infra.errors import ProtocolConfigurationError
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        non_existent_path = tmp_path / "does_not_exist.yaml"
        test_correlation_id = uuid4()

        loader = HandlerPluginLoader()

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            loader.load_from_contract(
                non_existent_path, correlation_id=test_correlation_id
            )

        # Verify correlation_id is in error context
        error = exc_info.value
        # The correlation_id is stored in the model's context or as a top-level attribute
        assert error.model.correlation_id is not None
        # Verify the correlation_id matches what was provided
        assert error.model.correlation_id == test_correlation_id

    def test_not_a_file_includes_correlation_id(self, tmp_path: Path) -> None:
        """Test that NOT_A_FILE error includes correlation_id."""
        from uuid import uuid4

        from omnibase_infra.errors import ProtocolConfigurationError
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        directory_path = tmp_path / "handlers"
        directory_path.mkdir()
        test_correlation_id = uuid4()

        loader = HandlerPluginLoader()

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            loader.load_from_contract(
                directory_path, correlation_id=test_correlation_id
            )

        # Verify correlation_id is in error context
        error = exc_info.value
        assert error.model.correlation_id is not None
        # Verify the correlation_id matches what was provided
        assert error.model.correlation_id == test_correlation_id

    def test_auto_generated_correlation_id_when_not_provided(
        self, tmp_path: Path
    ) -> None:
        """Test that correlation_id is auto-generated when not provided."""
        from omnibase_infra.errors import ProtocolConfigurationError
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        non_existent_path = tmp_path / "does_not_exist.yaml"

        loader = HandlerPluginLoader()

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            loader.load_from_contract(non_existent_path)  # No correlation_id provided

        # Verify a correlation_id was auto-generated
        error = exc_info.value
        assert error.model.correlation_id is not None
