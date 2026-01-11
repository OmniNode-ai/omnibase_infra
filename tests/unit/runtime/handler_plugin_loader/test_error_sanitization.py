# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Tests for error message sanitization in HandlerPluginLoader.

This module tests that error messages do not disclose sensitive information
such as filesystem paths, and that the _sanitize_exception_message helper
correctly sanitizes exception messages.

Part of OMN-1132: Handler Plugin Loader implementation.

Security Focus:
    - Error messages should not expose full filesystem paths
    - Exception details should be sanitized before user-facing messages
    - Correlation IDs should be preserved for tracing
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from omnibase_infra.runtime.handler_plugin_loader import (
    HandlerPluginLoader,
    _sanitize_exception_message,
)


class TestSanitizeExceptionMessage:
    """Tests for the _sanitize_exception_message helper function."""

    def test_sanitizes_unix_absolute_paths(self) -> None:
        """Test that Unix absolute paths are replaced with <path>."""
        exc = OSError("Permission denied: '/etc/secrets/key.pem'")
        result = _sanitize_exception_message(exc)

        # Verify sensitive path is removed (negative assertion)
        assert "/etc/secrets/key.pem" not in result
        assert "/etc/secrets" not in result
        assert "/etc" not in result

        # Verify sanitized placeholder is present (positive assertion)
        assert "<path>" in result, (
            f"Expected <path> placeholder in sanitized result: {result}"
        )

        # Verify the non-path content is preserved
        assert "Permission denied" in result

    def test_sanitizes_windows_paths(self) -> None:
        """Test that Windows paths are replaced with <path>."""
        exc = OSError("Access denied: C:\\Users\\admin\\secrets\\config.yaml")
        result = _sanitize_exception_message(exc)

        # Verify sensitive Windows path components are removed (negative assertions)
        assert "C:\\Users" not in result, (
            f"Windows path prefix should be sanitized: {result}"
        )
        assert "admin\\secrets" not in result, (
            f"Windows path should be fully sanitized: {result}"
        )
        assert "config.yaml" not in result or "<path>" in result, (
            f"Path should be replaced with placeholder: {result}"
        )

        # Verify the non-path content is preserved
        assert "Access denied" in result

    def test_sanitizes_relative_paths(self) -> None:
        """Test that relative paths are replaced with <path>."""
        exc = FileNotFoundError("File not found: ./config/secrets.yaml")
        result = _sanitize_exception_message(exc)

        # Verify sensitive path is removed (negative assertion)
        assert "./config/secrets.yaml" not in result, (
            f"Relative path should be sanitized but was found in: {result}"
        )

        # Verify sanitized placeholder is present (positive assertion)
        assert "<path>" in result, (
            f"Expected <path> placeholder in sanitized result: {result}"
        )

        # Verify the non-path content is preserved
        assert "File not found" in result

    def test_sanitizes_paths_in_nested_directories(self) -> None:
        """Test that deeply nested paths are sanitized."""
        exc = PermissionError(
            "Cannot read /home/user/projects/myapp/config/handlers/handler_contract.yaml"
        )
        result = _sanitize_exception_message(exc)

        assert "/home/user/projects" not in result
        assert "<path>" in result

    def test_preserves_non_path_content(self) -> None:
        """Test that non-path error content is preserved."""
        exc = ValueError("Invalid handler type: expected 'effect', got 'invalid'")
        result = _sanitize_exception_message(exc)

        # Should preserve the original message since there are no paths
        assert "Invalid handler type" in result
        assert "effect" in result
        assert "invalid" in result

    def test_preserves_line_numbers(self) -> None:
        """Test that line number information is preserved."""
        exc = SyntaxError("YAML syntax error at line 42, column 5")
        result = _sanitize_exception_message(exc)

        # Line numbers should be preserved
        assert "42" in result
        assert "5" in result

    def test_handles_empty_exception_message(self) -> None:
        """Test that empty exception messages are handled."""
        exc = Exception("")
        result = _sanitize_exception_message(exc)

        assert result == ""

    def test_handles_exception_with_only_path(self) -> None:
        """Test that exception with only a path is fully sanitized."""
        exc = FileNotFoundError("/var/log/myapp/errors.log")
        result = _sanitize_exception_message(exc)

        assert "/var/log" not in result

    def test_sanitizes_quoted_paths(self) -> None:
        """Test that quoted paths are sanitized."""
        exc = OSError("File '/home/user/config.yaml' not found")
        result = _sanitize_exception_message(exc)

        assert "/home/user/config.yaml" not in result
        assert "'<path>'" in result


class TestErrorMessageSanitizationInLoader:
    """Tests for error message sanitization in HandlerPluginLoader methods."""

    def test_yaml_error_does_not_expose_full_path(self, tmp_path: Path) -> None:
        """Test that YAML parsing errors do not expose full filesystem paths."""
        from omnibase_infra.errors import ProtocolConfigurationError

        # Create a file with truly invalid YAML syntax (tabs mixed with spaces)
        contract_file = tmp_path / "handler_contract.yaml"
        contract_file.write_text(
            "handler_name: test\n"
            "  \t- invalid indentation mixing tabs and spaces\n"
            "handler_class:\n"
            "\t- more invalid: [\n"
        )

        loader = HandlerPluginLoader()

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            loader.load_from_contract(contract_file)

        error_message = str(exc_info.value)

        # The error message should NOT contain the full path
        assert str(tmp_path) not in error_message

        # Should indicate YAML error (either syntax or validation)
        # The important thing is paths are not exposed
        assert "yaml" in error_message.lower() or "validation" in error_message.lower()

    def test_file_not_found_shows_only_filename(self, tmp_path: Path) -> None:
        """Test that file not found errors show only the filename, not full path."""
        from omnibase_infra.errors import ProtocolConfigurationError

        nonexistent = tmp_path / "deep" / "nested" / "handler_contract.yaml"

        loader = HandlerPluginLoader()

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            loader.load_from_contract(nonexistent)

        error_message = str(exc_info.value)

        # Positive assertion: Should show the filename in error message
        assert "handler_contract.yaml" in error_message, (
            f"Expected filename in error message: {error_message}"
        )

        # Negative assertions: Should NOT expose directory structure in the message text
        # (Note: full path may be in context for debugging, but not in the message)
        # Using AND logic: ALL sensitive path components must be absent
        assert "deep/nested" not in error_message, (
            f"Directory structure 'deep/nested' should not be exposed in: {error_message}"
        )
        assert str(tmp_path) not in error_message, (
            f"Full temp path should not be exposed in: {error_message}"
        )

    def test_error_context_contains_full_path_for_debugging(
        self, tmp_path: Path
    ) -> None:
        """Test that error context contains full path for debugging purposes.

        While the error MESSAGE should not expose paths, the error CONTEXT
        should contain the full path for debugging and observability.
        """
        from omnibase_infra.errors import ProtocolConfigurationError

        nonexistent = tmp_path / "handler_contract.yaml"

        loader = HandlerPluginLoader()

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            loader.load_from_contract(nonexistent)

        # Error context should contain full path for debugging
        error = exc_info.value
        context = error.model.context
        assert "contract_path" in context
        assert str(tmp_path) in context["contract_path"]

    def test_import_error_does_not_expose_system_paths(self, tmp_path: Path) -> None:
        """Test that import errors do not expose system paths."""
        from omnibase_infra.errors import InfraConnectionError

        # Create a contract pointing to a non-existent module
        contract_file = tmp_path / "handler_contract.yaml"
        contract_file.write_text(
            """
handler_name: test.handler
handler_class: nonexistent.module.path.Handler
handler_type: compute
"""
        )

        loader = HandlerPluginLoader()

        with pytest.raises(InfraConnectionError) as exc_info:
            loader.load_from_contract(contract_file)

        error_message = str(exc_info.value)

        # Should indicate module not found
        assert "Module not found" in error_message or "module" in error_message.lower()

        # Should NOT contain system Python paths
        assert "/usr/lib" not in error_message
        assert "/site-packages" not in error_message

    def test_correlation_id_preserved_in_sanitized_error(self, tmp_path: Path) -> None:
        """Test that correlation ID is preserved even when error is sanitized."""
        from uuid import uuid4

        from omnibase_infra.errors import ProtocolConfigurationError

        nonexistent = tmp_path / "handler_contract.yaml"
        correlation_id = str(uuid4())

        loader = HandlerPluginLoader()

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            loader.load_from_contract(nonexistent, correlation_id=correlation_id)

        # Correlation ID should be preserved in error model
        error = exc_info.value
        assert error.model.correlation_id is not None
        assert str(error.model.correlation_id) == correlation_id


class TestErrorCodePresenceInSanitizedErrors:
    """Tests to ensure error codes are present even in sanitized errors."""

    def test_file_not_found_has_error_code(self, tmp_path: Path) -> None:
        """Test that FILE_NOT_FOUND error has proper error code."""
        from omnibase_infra.enums import EnumHandlerLoaderError
        from omnibase_infra.errors import ProtocolConfigurationError

        nonexistent = tmp_path / "handler_contract.yaml"

        loader = HandlerPluginLoader()

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            loader.load_from_contract(nonexistent)

        error = exc_info.value
        assert (
            error.model.context.get("loader_error")
            == EnumHandlerLoaderError.FILE_NOT_FOUND.value
        )

    def test_yaml_error_has_error_code(self, tmp_path: Path) -> None:
        """Test that YAML syntax errors have proper error code."""
        from omnibase_infra.enums import EnumHandlerLoaderError
        from omnibase_infra.errors import ProtocolConfigurationError

        contract_file = tmp_path / "handler_contract.yaml"
        contract_file.write_text("handler_name: test\n  bad: yaml: syntax: here")

        loader = HandlerPluginLoader()

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            loader.load_from_contract(contract_file)

        error = exc_info.value
        assert (
            error.model.context.get("loader_error")
            == EnumHandlerLoaderError.INVALID_YAML_SYNTAX.value
        )
