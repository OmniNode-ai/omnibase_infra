# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Tests for early YAML validation in HandlerPluginLoader.

Part of OMN-1132: Handler Plugin Loader implementation.

These tests verify that YAML validation happens early in the discover_and_load
flow, providing fail-fast behavior for malformed YAML files before expensive
operations like path resolution and handler class loading.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from .conftest import INVALID_YAML_SYNTAX, MINIMAL_HANDLER_CONTRACT_YAML


class TestValidateYamlSyntaxMethod:
    """Tests for the _validate_yaml_syntax helper method."""

    def test_valid_yaml_returns_true(self, tmp_path: Path) -> None:
        """Test that valid YAML returns True."""
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        # Create a valid YAML file
        yaml_file = tmp_path / "valid.yaml"
        yaml_file.write_text(
            MINIMAL_HANDLER_CONTRACT_YAML.format(
                handler_name="test.handler",
                handler_class="tests.unit.runtime.handler_plugin_loader.conftest.MockValidHandler",
            )
        )

        loader = HandlerPluginLoader()
        result = loader._validate_yaml_syntax(yaml_file)

        assert result is True

    def test_malformed_yaml_returns_false_graceful_mode(self, tmp_path: Path) -> None:
        """Test that malformed YAML returns False in graceful mode."""
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        # Create a malformed YAML file
        yaml_file = tmp_path / "invalid.yaml"
        yaml_file.write_text(INVALID_YAML_SYNTAX)

        loader = HandlerPluginLoader()
        result = loader._validate_yaml_syntax(yaml_file, raise_on_error=False)

        assert result is False

    def test_malformed_yaml_raises_error_strict_mode(self, tmp_path: Path) -> None:
        """Test that malformed YAML raises error in strict mode."""
        from omnibase_infra.enums import EnumHandlerLoaderError
        from omnibase_infra.errors import ProtocolConfigurationError
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        # Create a malformed YAML file
        yaml_file = tmp_path / "invalid.yaml"
        yaml_file.write_text(INVALID_YAML_SYNTAX)

        loader = HandlerPluginLoader()

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            loader._validate_yaml_syntax(yaml_file, raise_on_error=True)

        # Verify error code
        error = exc_info.value
        assert (
            error.model.context.get("loader_error")
            == EnumHandlerLoaderError.INVALID_YAML_SYNTAX.value
        )

    def test_error_message_includes_file_path(self, tmp_path: Path) -> None:
        """Test that error message includes the file path."""
        from omnibase_infra.errors import ProtocolConfigurationError
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        # Create a malformed YAML file
        yaml_file = tmp_path / "malformed_contract.yaml"
        yaml_file.write_text(INVALID_YAML_SYNTAX)

        loader = HandlerPluginLoader()

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            loader._validate_yaml_syntax(yaml_file, raise_on_error=True)

        # Verify file path is in error message
        error_message = str(exc_info.value)
        assert "malformed_contract.yaml" in error_message

    def test_error_includes_correlation_id(self, tmp_path: Path) -> None:
        """Test that error includes correlation_id."""
        from uuid import uuid4

        from omnibase_infra.errors import ProtocolConfigurationError
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        yaml_file = tmp_path / "invalid.yaml"
        yaml_file.write_text(INVALID_YAML_SYNTAX)

        loader = HandlerPluginLoader()
        test_correlation_id = str(uuid4())

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            loader._validate_yaml_syntax(
                yaml_file, correlation_id=test_correlation_id, raise_on_error=True
            )

        error = exc_info.value
        assert str(error.model.correlation_id) == test_correlation_id

    def test_various_yaml_syntax_errors(self, tmp_path: Path) -> None:
        """Test detection of various YAML syntax errors."""
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        loader = HandlerPluginLoader()

        # Test cases of malformed YAML that will definitely fail parsing
        malformed_cases = [
            # Unclosed bracket
            "key: [unclosed",
            # Bad indentation causing parse error
            "key:\n  nested: value\n wrong: indent",
            # Unclosed single quote
            "key: 'unclosed quote",
            # Unclosed double quote
            'key: "unclosed quote',
            # Invalid flow mapping
            "key: {unclosed",
        ]

        for i, malformed_yaml in enumerate(malformed_cases):
            yaml_file = tmp_path / f"malformed_{i}.yaml"
            yaml_file.write_text(malformed_yaml)

            result = loader._validate_yaml_syntax(yaml_file, raise_on_error=False)
            assert result is False, (
                f"Expected False for case {i}: {malformed_yaml[:30]}..."
            )


class TestEarlyYamlValidationInDiscoverAndLoad:
    """Tests for early YAML validation in discover_and_load."""

    def test_malformed_yaml_skipped_during_discovery(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that malformed YAML files are skipped during discovery."""
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        monkeypatch.chdir(tmp_path)

        # Create a directory structure with one valid and one invalid contract
        valid_dir = tmp_path / "valid_handler"
        valid_dir.mkdir()
        (valid_dir / "handler_contract.yaml").write_text(
            MINIMAL_HANDLER_CONTRACT_YAML.format(
                handler_name="valid.handler",
                handler_class="tests.unit.runtime.handler_plugin_loader.conftest.MockValidHandler",
            )
        )

        invalid_dir = tmp_path / "invalid_handler"
        invalid_dir.mkdir()
        (invalid_dir / "handler_contract.yaml").write_text(INVALID_YAML_SYNTAX)

        loader = HandlerPluginLoader()
        handlers = loader.discover_and_load(["**/handler_contract.yaml"])

        # Only the valid handler should be loaded
        assert len(handlers) == 1
        assert handlers[0].handler_name == "valid.handler"

    def test_early_validation_prevents_expensive_operations(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that early validation skips files before path resolution."""
        from unittest.mock import patch

        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        monkeypatch.chdir(tmp_path)

        # Create a malformed YAML file
        handler_dir = tmp_path / "handler"
        handler_dir.mkdir()
        (handler_dir / "handler_contract.yaml").write_text(INVALID_YAML_SYNTAX)

        loader = HandlerPluginLoader()

        # Track if load_from_contract is called
        load_called = []
        original_load = loader.load_from_contract

        def tracking_load(*args, **kwargs):
            load_called.append(True)
            return original_load(*args, **kwargs)

        with patch.object(loader, "load_from_contract", tracking_load):
            handlers = loader.discover_and_load(["**/handler_contract.yaml"])

        # load_from_contract should NOT be called for malformed YAML
        assert len(load_called) == 0
        assert len(handlers) == 0

    def test_multiple_malformed_files_all_skipped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that multiple malformed YAML files are all skipped."""
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        monkeypatch.chdir(tmp_path)

        # Create multiple malformed YAML files
        for i in range(3):
            handler_dir = tmp_path / f"handler_{i}"
            handler_dir.mkdir()
            (handler_dir / "handler_contract.yaml").write_text(
                f"invalid_yaml_{i}: [unclosed"
            )

        loader = HandlerPluginLoader()
        handlers = loader.discover_and_load(["**/handler_contract.yaml"])

        assert len(handlers) == 0

    def test_empty_yaml_file_passes_syntax_but_fails_schema(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that empty YAML files pass syntax validation but fail schema validation."""
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        monkeypatch.chdir(tmp_path)

        # Create an empty YAML file (valid syntax, empty content)
        handler_dir = tmp_path / "handler"
        handler_dir.mkdir()
        (handler_dir / "handler_contract.yaml").write_text("")

        loader = HandlerPluginLoader()

        # Empty file should pass YAML syntax validation
        result = loader._validate_yaml_syntax(
            handler_dir / "handler_contract.yaml", raise_on_error=False
        )
        assert result is True  # Empty YAML is syntactically valid

        # But discover_and_load should fail on schema validation
        handlers = loader.discover_and_load(["**/handler_contract.yaml"])
        assert len(handlers) == 0  # Failed due to empty content


class TestYamlValidationErrorDetails:
    """Tests for detailed error information from YAML validation."""

    def test_yaml_error_includes_line_info_when_available(self, tmp_path: Path) -> None:
        """Test that YAML error includes line information when available."""
        import re

        from omnibase_infra.errors import ProtocolConfigurationError
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        # Create YAML with error on a specific line (line 4)
        yaml_content = """handler_name: test
handler_class: test.Handler
handler_type: compute
invalid_line: [unclosed bracket
"""
        yaml_file = tmp_path / "contract.yaml"
        yaml_file.write_text(yaml_content)

        loader = HandlerPluginLoader()

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            loader._validate_yaml_syntax(yaml_file, raise_on_error=True)

        # Error should contain line information from YAML parser in format "line N"
        # where N is a digit. The error occurs on line 4 of the YAML content.
        error_message = str(exc_info.value)

        # Match YAML parser line format: "line N" where N is a positive integer
        # This is more specific than checking for "line" substring which could
        # match words like "inline" or "guideline"
        line_info_pattern = re.compile(r"\bline\s+\d+\b", re.IGNORECASE)
        match = line_info_pattern.search(error_message)

        assert match is not None, (
            f"Expected YAML error to include line information in format 'line N', "
            f"but got: {error_message}"
        )

        # Additionally verify the line number is reasonable (lines 1-5 for this content)
        # The error is on line 4 where the unclosed bracket is
        line_number = int(match.group().split()[-1])
        assert 1 <= line_number <= 5, (
            f"Expected line number between 1-5 for 4-line YAML with error on line 4, "
            f"but got line {line_number}"
        )

    def test_file_read_error_uses_correct_error_code(self, tmp_path: Path) -> None:
        """Test that file read errors use FILE_READ_ERROR code."""
        from unittest.mock import patch

        from omnibase_infra.enums import EnumHandlerLoaderError
        from omnibase_infra.errors import ProtocolConfigurationError
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        yaml_file = tmp_path / "contract.yaml"
        yaml_file.write_text("valid: yaml")

        loader = HandlerPluginLoader()

        # Mock file open to raise OSError
        def mock_open(*args, **kwargs):
            raise OSError("Permission denied")

        with patch.object(Path, "open", mock_open):
            with pytest.raises(ProtocolConfigurationError) as exc_info:
                loader._validate_yaml_syntax(yaml_file, raise_on_error=True)

        error = exc_info.value
        assert (
            error.model.context.get("loader_error")
            == EnumHandlerLoaderError.FILE_READ_ERROR.value
        )
