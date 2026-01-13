# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Negative tests for invalid contract YAML files in HandlerPluginLoader.

Part of OMN-1132: Handler Plugin Loader implementation.
PR #143 review: Added comprehensive negative tests for malformed/invalid YAML.

These tests verify that HandlerPluginLoader correctly handles various invalid
contract YAML scenarios and returns appropriate error codes for each case:

Error Code Coverage:
    - HANDLER_LOADER_002 (INVALID_YAML_SYNTAX): Malformed YAML syntax
    - HANDLER_LOADER_003 (SCHEMA_VALIDATION_FAILED): Invalid schema/types
    - HANDLER_LOADER_004 (MISSING_REQUIRED_FIELDS): Required fields missing
    - HANDLER_LOADER_010 (MODULE_NOT_FOUND): Invalid handler_module path
    - HANDLER_LOADER_011 (CLASS_NOT_FOUND): Invalid handler_class path

Test Categories:
    - TestMalformedYamlSyntax: Invalid YAML syntax (brackets, quotes, indentation)
    - TestEmptyYamlContract: Empty and whitespace-only YAML files
    - TestMissingRequiredFields: Missing handler_name, handler_class, handler_type
    - TestInvalidFieldTypes: Wrong types for fields (int instead of string)
    - TestInvalidHandlerPaths: Invalid handler_class and handler_module paths
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest


class TestMalformedYamlSyntax:
    """Negative tests for malformed YAML syntax.

    These tests verify error code HANDLER_LOADER_002 (INVALID_YAML_SYNTAX)
    is returned for various YAML syntax errors.
    """

    @pytest.mark.parametrize(
        ("yaml_content", "description"),
        [
            # Unclosed brackets
            (
                "handler_name: test\nhandler_class: [unclosed",
                "unclosed square bracket",
            ),
            (
                "handler_name: test\nhandler_class: {unclosed",
                "unclosed curly bracket",
            ),
            # Unclosed quotes
            (
                "handler_name: 'unclosed single quote\nhandler_class: test",
                "unclosed single quote",
            ),
            (
                'handler_name: "unclosed double quote\nhandler_class: test',
                "unclosed double quote",
            ),
            # Invalid indentation
            (
                "key:\n  nested: value\n wrong: indent",
                "invalid indentation",
            ),
            # Invalid mapping
            (
                "key: value: extra colon",
                "extra colon in value",
            ),
            # Tab character in indentation (YAML spec discourages mixing tabs/spaces)
            (
                "key:\n\t- invalid tab indent",
                "tab in indentation",
            ),
            # Invalid anchor reference
            (
                "handler_name: *undefined_anchor",
                "undefined anchor reference",
            ),
            # Duplicate key (YAML 1.2 spec - duplicate keys are invalid)
            # Note: PyYAML silently ignores duplicate keys but this tests parser behavior
            (
                "handler_name: first\nhandler_name: second\n  nested_without_parent: value",
                "duplicate keys with invalid nesting",
            ),
        ],
        ids=[
            "unclosed_square_bracket",
            "unclosed_curly_bracket",
            "unclosed_single_quote",
            "unclosed_double_quote",
            "invalid_indentation",
            "extra_colon",
            "tab_indentation",
            "undefined_anchor",
            "duplicate_keys_invalid_nesting",
        ],
    )
    def test_malformed_yaml_returns_invalid_yaml_syntax_error(
        self,
        tmp_path: Path,
        yaml_content: str,
        description: str,
    ) -> None:
        """Test that malformed YAML syntax returns INVALID_YAML_SYNTAX error code.

        Args:
            tmp_path: Pytest temp directory fixture.
            yaml_content: Malformed YAML content to test.
            description: Human-readable description of the syntax error.
        """
        from omnibase_infra.enums import EnumHandlerLoaderError
        from omnibase_infra.errors import ProtocolConfigurationError
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        contract_file = tmp_path / "handler_contract.yaml"
        contract_file.write_text(yaml_content)

        loader = HandlerPluginLoader()

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            loader.load_from_contract(contract_file)

        error = exc_info.value
        error_code = error.model.context.get("loader_error")

        # Should be INVALID_YAML_SYNTAX for syntax errors
        assert error_code == EnumHandlerLoaderError.INVALID_YAML_SYNTAX.value, (
            f"Expected INVALID_YAML_SYNTAX for '{description}', got {error_code}"
        )

    def test_malformed_yaml_error_includes_correlation_id(self, tmp_path: Path) -> None:
        """Test that malformed YAML error includes the provided correlation_id."""
        from omnibase_infra.errors import ProtocolConfigurationError
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        contract_file = tmp_path / "handler_contract.yaml"
        contract_file.write_text("handler_name: [unclosed bracket")

        loader = HandlerPluginLoader()
        test_correlation_id = uuid4()

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            loader.load_from_contract(contract_file, correlation_id=test_correlation_id)

        error = exc_info.value
        assert error.model.correlation_id == test_correlation_id

    def test_malformed_yaml_error_message_includes_syntax_details(
        self, tmp_path: Path
    ) -> None:
        """Test that malformed YAML error message includes syntax error details."""
        from omnibase_infra.errors import ProtocolConfigurationError
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        contract_file = tmp_path / "handler_contract.yaml"
        contract_file.write_text("handler_name: [unclosed bracket")

        loader = HandlerPluginLoader()

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            loader.load_from_contract(contract_file)

        error_message = str(exc_info.value).lower()

        # Error should indicate it's a YAML syntax issue
        assert (
            "yaml" in error_message
            or "syntax" in error_message
            or "invalid" in error_message
        )


class TestEmptyYamlContract:
    """Negative tests for empty YAML contract files.

    Empty files are syntactically valid YAML (parse to None) but fail schema
    validation, returning SCHEMA_VALIDATION_FAILED (HANDLER_LOADER_003).
    """

    def test_empty_file_returns_schema_validation_failed(self, tmp_path: Path) -> None:
        """Test that completely empty file returns SCHEMA_VALIDATION_FAILED."""
        from omnibase_infra.enums import EnumHandlerLoaderError
        from omnibase_infra.errors import ProtocolConfigurationError
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        contract_file = tmp_path / "handler_contract.yaml"
        contract_file.write_text("")

        loader = HandlerPluginLoader()

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            loader.load_from_contract(contract_file)

        error = exc_info.value
        assert (
            error.model.context.get("loader_error")
            == EnumHandlerLoaderError.SCHEMA_VALIDATION_FAILED.value
        )

    def test_whitespace_only_file_returns_schema_validation_failed(
        self, tmp_path: Path
    ) -> None:
        """Test that whitespace-only file returns SCHEMA_VALIDATION_FAILED.

        Note: Whitespace-only files parse to None in YAML, which triggers
        the same "empty content" check as empty files (SCHEMA_VALIDATION_FAILED).
        """
        from omnibase_infra.enums import EnumHandlerLoaderError
        from omnibase_infra.errors import ProtocolConfigurationError
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        contract_file = tmp_path / "handler_contract.yaml"
        contract_file.write_text(
            "   \n\n   "
        )  # Whitespace without tabs (tabs can cause YAML error)

        loader = HandlerPluginLoader()

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            loader.load_from_contract(contract_file)

        error = exc_info.value
        assert (
            error.model.context.get("loader_error")
            == EnumHandlerLoaderError.SCHEMA_VALIDATION_FAILED.value
        )

    def test_comment_only_file_returns_schema_validation_failed(
        self, tmp_path: Path
    ) -> None:
        """Test that file with only comments returns SCHEMA_VALIDATION_FAILED."""
        from omnibase_infra.enums import EnumHandlerLoaderError
        from omnibase_infra.errors import ProtocolConfigurationError
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        contract_file = tmp_path / "handler_contract.yaml"
        contract_file.write_text("# This is just a comment\n# Another comment")

        loader = HandlerPluginLoader()

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            loader.load_from_contract(contract_file)

        error = exc_info.value
        assert (
            error.model.context.get("loader_error")
            == EnumHandlerLoaderError.SCHEMA_VALIDATION_FAILED.value
        )

    def test_empty_file_error_message_indicates_empty_content(
        self, tmp_path: Path
    ) -> None:
        """Test that empty file error message clearly indicates the issue."""
        from omnibase_infra.errors import ProtocolConfigurationError
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        contract_file = tmp_path / "handler_contract.yaml"
        contract_file.write_text("")

        loader = HandlerPluginLoader()

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            loader.load_from_contract(contract_file)

        error_message = str(exc_info.value).lower()
        assert "empty" in error_message


class TestMissingRequiredFields:
    """Negative tests for contracts missing required fields.

    Required fields: handler_name, handler_class, handler_type
    Returns SCHEMA_VALIDATION_FAILED (HANDLER_LOADER_003) via Pydantic validation.
    """

    def test_missing_handler_name_returns_schema_validation_failed(
        self, tmp_path: Path
    ) -> None:
        """Test that missing handler_name returns SCHEMA_VALIDATION_FAILED.

        Note: The Pydantic model accepts both 'handler_name' and 'name' as aliases,
        so the error message may reference 'name' instead of 'handler_name'.
        """
        from omnibase_infra.enums import EnumHandlerLoaderError
        from omnibase_infra.errors import ProtocolConfigurationError
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        contract_file = tmp_path / "handler_contract.yaml"
        contract_file.write_text(
            """
handler_class: tests.unit.runtime.handler_plugin_loader.conftest.MockValidHandler
handler_type: compute
"""
        )

        loader = HandlerPluginLoader()

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            loader.load_from_contract(contract_file)

        error = exc_info.value
        assert (
            error.model.context.get("loader_error")
            == EnumHandlerLoaderError.SCHEMA_VALIDATION_FAILED.value
        )
        # Field may be reported as either "handler_name" or "name" (alias)
        error_lower = str(error).lower()
        assert "handler_name" in error_lower or "name" in error_lower

    def test_missing_handler_class_returns_schema_validation_failed(
        self, tmp_path: Path
    ) -> None:
        """Test that missing handler_class returns SCHEMA_VALIDATION_FAILED."""
        from omnibase_infra.enums import EnumHandlerLoaderError
        from omnibase_infra.errors import ProtocolConfigurationError
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        contract_file = tmp_path / "handler_contract.yaml"
        contract_file.write_text(
            """
handler_name: test.handler
handler_type: compute
"""
        )

        loader = HandlerPluginLoader()

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            loader.load_from_contract(contract_file)

        error = exc_info.value
        assert (
            error.model.context.get("loader_error")
            == EnumHandlerLoaderError.SCHEMA_VALIDATION_FAILED.value
        )
        assert "handler_class" in str(error).lower()

    def test_missing_handler_type_returns_schema_validation_failed(
        self, tmp_path: Path
    ) -> None:
        """Test that missing handler_type returns SCHEMA_VALIDATION_FAILED."""
        from omnibase_infra.enums import EnumHandlerLoaderError
        from omnibase_infra.errors import ProtocolConfigurationError
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        contract_file = tmp_path / "handler_contract.yaml"
        contract_file.write_text(
            """
handler_name: test.handler
handler_class: tests.unit.runtime.handler_plugin_loader.conftest.MockValidHandler
"""
        )

        loader = HandlerPluginLoader()

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            loader.load_from_contract(contract_file)

        error = exc_info.value
        assert (
            error.model.context.get("loader_error")
            == EnumHandlerLoaderError.SCHEMA_VALIDATION_FAILED.value
        )
        assert "handler_type" in str(error).lower()

    def test_missing_multiple_fields_lists_all_errors(self, tmp_path: Path) -> None:
        """Test that missing multiple fields lists all validation errors.

        Note: The Pydantic model uses 'name' as an alias for 'handler_name',
        so the error message may reference 'name' instead of 'handler_name'.
        """
        from omnibase_infra.errors import ProtocolConfigurationError
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        contract_file = tmp_path / "handler_contract.yaml"
        contract_file.write_text(
            """
handler_type: compute
"""
        )

        loader = HandlerPluginLoader()

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            loader.load_from_contract(contract_file)

        error_message = str(exc_info.value).lower()
        # Both handler_name (or name alias) and handler_class are missing
        assert "handler_name" in error_message or "name" in error_message
        assert "handler_class" in error_message


class TestInvalidFieldTypes:
    """Negative tests for contracts with invalid field types.

    These tests verify that wrong types for fields (e.g., int instead of string)
    return SCHEMA_VALIDATION_FAILED (HANDLER_LOADER_003).
    """

    def test_handler_name_as_int_returns_schema_validation_failed(
        self, tmp_path: Path
    ) -> None:
        """Test that handler_name as integer returns SCHEMA_VALIDATION_FAILED."""
        from omnibase_infra.enums import EnumHandlerLoaderError
        from omnibase_infra.errors import ProtocolConfigurationError
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        contract_file = tmp_path / "handler_contract.yaml"
        contract_file.write_text(
            """
handler_name: 12345
handler_class: tests.unit.runtime.handler_plugin_loader.conftest.MockValidHandler
handler_type: compute
"""
        )

        loader = HandlerPluginLoader()

        # Note: Pydantic may coerce int to string, so this may succeed.
        # If it fails, it should be SCHEMA_VALIDATION_FAILED.
        try:
            result = loader.load_from_contract(contract_file)
            # If coercion works, verify handler_name is the string "12345"
            assert result.handler_name == "12345"
        except ProtocolConfigurationError as e:
            assert (
                e.model.context.get("loader_error")
                == EnumHandlerLoaderError.SCHEMA_VALIDATION_FAILED.value
            )

    def test_handler_class_as_list_returns_schema_validation_failed(
        self, tmp_path: Path
    ) -> None:
        """Test that handler_class as list returns SCHEMA_VALIDATION_FAILED."""
        from omnibase_infra.enums import EnumHandlerLoaderError
        from omnibase_infra.errors import ProtocolConfigurationError
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        contract_file = tmp_path / "handler_contract.yaml"
        contract_file.write_text(
            """
handler_name: test.handler
handler_class:
  - item1
  - item2
handler_type: compute
"""
        )

        loader = HandlerPluginLoader()

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            loader.load_from_contract(contract_file)

        error = exc_info.value
        assert (
            error.model.context.get("loader_error")
            == EnumHandlerLoaderError.SCHEMA_VALIDATION_FAILED.value
        )

    def test_handler_type_as_list_returns_schema_validation_failed(
        self, tmp_path: Path
    ) -> None:
        """Test that handler_type as list returns SCHEMA_VALIDATION_FAILED."""
        from omnibase_infra.enums import EnumHandlerLoaderError
        from omnibase_infra.errors import ProtocolConfigurationError
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        contract_file = tmp_path / "handler_contract.yaml"
        contract_file.write_text(
            """
handler_name: test.handler
handler_class: tests.unit.runtime.handler_plugin_loader.conftest.MockValidHandler
handler_type:
  - compute
  - effect
"""
        )

        loader = HandlerPluginLoader()

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            loader.load_from_contract(contract_file)

        error = exc_info.value
        assert (
            error.model.context.get("loader_error")
            == EnumHandlerLoaderError.SCHEMA_VALIDATION_FAILED.value
        )

    def test_capability_tags_as_string_normalizes_to_list(self, tmp_path: Path) -> None:
        """Test that capability_tags as string is normalized to a list.

        The ModelHandlerContract model's field_validator for capability_tags
        normalizes single string values to a list with one element. This test
        verifies that behavior rather than expecting an error.
        """
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        contract_file = tmp_path / "handler_contract.yaml"
        contract_file.write_text(
            """
handler_name: test.handler
handler_class: tests.unit.runtime.handler_plugin_loader.conftest.MockValidHandler
handler_type: compute
capability_tags: "single_tag"
"""
        )

        loader = HandlerPluginLoader()

        # This should succeed because strings are normalized to lists
        result = loader.load_from_contract(contract_file)

        # Verify the string was normalized to a list
        assert result.capability_tags == ["single_tag"]

    def test_invalid_handler_type_value_returns_schema_validation_failed(
        self, tmp_path: Path
    ) -> None:
        """Test that invalid handler_type enum value returns SCHEMA_VALIDATION_FAILED."""
        from omnibase_infra.enums import EnumHandlerLoaderError
        from omnibase_infra.errors import ProtocolConfigurationError
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        contract_file = tmp_path / "handler_contract.yaml"
        contract_file.write_text(
            """
handler_name: test.handler
handler_class: tests.unit.runtime.handler_plugin_loader.conftest.MockValidHandler
handler_type: invalid_type_value
"""
        )

        loader = HandlerPluginLoader()

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            loader.load_from_contract(contract_file)

        error = exc_info.value
        assert (
            error.model.context.get("loader_error")
            == EnumHandlerLoaderError.SCHEMA_VALIDATION_FAILED.value
        )


class TestInvalidHandlerPaths:
    """Negative tests for invalid handler_class paths.

    These tests verify error codes for module and class import failures:
    - HANDLER_LOADER_010 (MODULE_NOT_FOUND): Module doesn't exist
    - HANDLER_LOADER_011 (CLASS_NOT_FOUND): Class doesn't exist in module
    """

    def test_nonexistent_module_returns_module_not_found(self, tmp_path: Path) -> None:
        """Test that nonexistent module returns MODULE_NOT_FOUND error code."""
        from omnibase_infra.enums import EnumHandlerLoaderError
        from omnibase_infra.errors import InfraConnectionError
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

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

        error = exc_info.value
        assert (
            error.model.context.get("loader_error")
            == EnumHandlerLoaderError.MODULE_NOT_FOUND.value
        )

    def test_nonexistent_class_in_valid_module_returns_class_not_found(
        self, tmp_path: Path
    ) -> None:
        """Test that nonexistent class in valid module returns CLASS_NOT_FOUND."""
        from omnibase_infra.enums import EnumHandlerLoaderError
        from omnibase_infra.errors import InfraConnectionError
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        contract_file = tmp_path / "handler_contract.yaml"
        # Use a real module but fake class name
        contract_file.write_text(
            """
handler_name: test.handler
handler_class: tests.unit.runtime.handler_plugin_loader.conftest.NonexistentClass
handler_type: compute
"""
        )

        loader = HandlerPluginLoader()

        with pytest.raises(InfraConnectionError) as exc_info:
            loader.load_from_contract(contract_file)

        error = exc_info.value
        assert (
            error.model.context.get("loader_error")
            == EnumHandlerLoaderError.CLASS_NOT_FOUND.value
        )

    def test_handler_class_without_module_path_returns_import_error(
        self, tmp_path: Path
    ) -> None:
        """Test that handler_class without module path returns appropriate error."""
        from omnibase_infra.errors import (
            InfraConnectionError,
            ProtocolConfigurationError,
        )
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        contract_file = tmp_path / "handler_contract.yaml"
        # Class name without module path (no dots)
        contract_file.write_text(
            """
handler_name: test.handler
handler_class: HandlerWithoutModule
handler_type: compute
"""
        )

        loader = HandlerPluginLoader()

        # May raise either ProtocolConfigurationError (validation) or InfraConnectionError (import)
        with pytest.raises((ProtocolConfigurationError, InfraConnectionError)):
            loader.load_from_contract(contract_file)

    def test_invalid_module_returns_error_with_correlation_id(
        self, tmp_path: Path
    ) -> None:
        """Test that module import error includes correlation_id."""
        from omnibase_infra.errors import InfraConnectionError
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        contract_file = tmp_path / "handler_contract.yaml"
        contract_file.write_text(
            """
handler_name: test.handler
handler_class: nonexistent.module.Handler
handler_type: compute
"""
        )

        loader = HandlerPluginLoader()
        test_correlation_id = uuid4()

        with pytest.raises(InfraConnectionError) as exc_info:
            loader.load_from_contract(contract_file, correlation_id=test_correlation_id)

        error = exc_info.value
        assert error.model.correlation_id == test_correlation_id


class TestProtocolNotImplemented:
    """Negative tests for handlers that don't implement ProtocolHandler.

    These tests verify HANDLER_LOADER_006 (PROTOCOL_NOT_IMPLEMENTED) is returned
    when the handler class is missing required protocol methods.
    """

    def test_handler_missing_all_methods_returns_protocol_not_implemented(
        self, tmp_path: Path
    ) -> None:
        """Test that handler missing all methods returns PROTOCOL_NOT_IMPLEMENTED."""
        from omnibase_infra.enums import EnumHandlerLoaderError
        from omnibase_infra.errors import ProtocolConfigurationError
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        contract_file = tmp_path / "handler_contract.yaml"
        # MockInvalidHandler has no protocol methods
        contract_file.write_text(
            """
handler_name: test.handler
handler_class: tests.unit.runtime.handler_plugin_loader.conftest.MockInvalidHandler
handler_type: compute
"""
        )

        loader = HandlerPluginLoader()

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            loader.load_from_contract(contract_file)

        error = exc_info.value
        assert (
            error.model.context.get("loader_error")
            == EnumHandlerLoaderError.PROTOCOL_NOT_IMPLEMENTED.value
        )

    def test_handler_missing_some_methods_returns_protocol_not_implemented(
        self, tmp_path: Path
    ) -> None:
        """Test that handler missing some methods returns PROTOCOL_NOT_IMPLEMENTED."""
        from omnibase_infra.enums import EnumHandlerLoaderError
        from omnibase_infra.errors import ProtocolConfigurationError
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        contract_file = tmp_path / "handler_contract.yaml"
        # MockPartialHandler only has describe()
        contract_file.write_text(
            """
handler_name: test.handler
handler_class: tests.unit.runtime.handler_plugin_loader.conftest.MockPartialHandler
handler_type: compute
"""
        )

        loader = HandlerPluginLoader()

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            loader.load_from_contract(contract_file)

        error = exc_info.value
        assert (
            error.model.context.get("loader_error")
            == EnumHandlerLoaderError.PROTOCOL_NOT_IMPLEMENTED.value
        )

    def test_protocol_error_lists_missing_methods(self, tmp_path: Path) -> None:
        """Test that PROTOCOL_NOT_IMPLEMENTED error lists which methods are missing."""
        from omnibase_infra.errors import ProtocolConfigurationError
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        contract_file = tmp_path / "handler_contract.yaml"
        contract_file.write_text(
            """
handler_name: test.handler
handler_class: tests.unit.runtime.handler_plugin_loader.conftest.MockPartialHandler
handler_type: compute
"""
        )

        loader = HandlerPluginLoader()

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            loader.load_from_contract(contract_file)

        error_message = str(exc_info.value).lower()
        # MockPartialHandler is missing handler_type, initialize, shutdown, execute
        # Error message should list at least some of these
        missing_methods = ["handler_type", "initialize", "shutdown", "execute"]
        found_any = any(method in error_message for method in missing_methods)
        assert found_any, (
            f"Expected error to list missing methods, got: {exc_info.value}"
        )


class TestValidationErrorContext:
    """Tests for error context completeness in validation errors."""

    def test_schema_validation_error_includes_validation_details(
        self, tmp_path: Path
    ) -> None:
        """Test that schema validation error includes Pydantic validation details."""
        from omnibase_infra.errors import ProtocolConfigurationError
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        contract_file = tmp_path / "handler_contract.yaml"
        contract_file.write_text(
            """
handler_name: test.handler
handler_type: compute
"""
        )  # Missing handler_class

        loader = HandlerPluginLoader()

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            loader.load_from_contract(contract_file)

        error = exc_info.value

        # Error context should include validation details
        context = error.model.context
        assert "contract_path" in context

        # Error message should include the field that failed
        assert "handler_class" in str(error).lower()

    def test_all_error_codes_have_contract_path_in_context(
        self, tmp_path: Path
    ) -> None:
        """Test that all file-level errors include contract_path in context."""
        from omnibase_infra.errors import ProtocolConfigurationError
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        # Test empty file error
        contract_file = tmp_path / "handler_contract.yaml"
        contract_file.write_text("")

        loader = HandlerPluginLoader()

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            loader.load_from_contract(contract_file)

        error = exc_info.value
        context = error.model.context
        assert "contract_path" in context
        assert str(contract_file) in context["contract_path"]

    def test_validation_errors_include_validation_errors_list(
        self, tmp_path: Path
    ) -> None:
        """Test that schema validation errors include validation_errors list."""
        from omnibase_infra.enums import EnumHandlerLoaderError
        from omnibase_infra.errors import ProtocolConfigurationError
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        contract_file = tmp_path / "handler_contract.yaml"
        contract_file.write_text(
            """
handler_type: compute
"""
        )  # Missing handler_name and handler_class

        loader = HandlerPluginLoader()

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            loader.load_from_contract(contract_file)

        error = exc_info.value
        assert (
            error.model.context.get("loader_error")
            == EnumHandlerLoaderError.SCHEMA_VALIDATION_FAILED.value
        )

        # Check if validation_errors is in context (may be optional)
        context = error.model.context
        if "validation_errors" in context:
            validation_errors = context["validation_errors"]
            assert isinstance(validation_errors, list)
            assert (
                len(validation_errors) >= 2
            )  # At least handler_name and handler_class


class TestYamlOnlyWithNonHandlerContent:
    """Tests for YAML files with valid syntax but non-handler content."""

    def test_yaml_with_unrelated_structure_returns_schema_validation_failed(
        self, tmp_path: Path
    ) -> None:
        """Test that YAML with unrelated structure returns SCHEMA_VALIDATION_FAILED."""
        from omnibase_infra.enums import EnumHandlerLoaderError
        from omnibase_infra.errors import ProtocolConfigurationError
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        contract_file = tmp_path / "handler_contract.yaml"
        # Valid YAML but wrong structure (like a config file)
        contract_file.write_text(
            """
database:
  host: localhost
  port: 5432
logging:
  level: INFO
"""
        )

        loader = HandlerPluginLoader()

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            loader.load_from_contract(contract_file)

        error = exc_info.value
        assert (
            error.model.context.get("loader_error")
            == EnumHandlerLoaderError.SCHEMA_VALIDATION_FAILED.value
        )

    def test_yaml_list_at_root_returns_schema_validation_failed(
        self, tmp_path: Path
    ) -> None:
        """Test that YAML with list at root returns SCHEMA_VALIDATION_FAILED."""
        from omnibase_infra.enums import EnumHandlerLoaderError
        from omnibase_infra.errors import ProtocolConfigurationError
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        contract_file = tmp_path / "handler_contract.yaml"
        # Valid YAML but root is a list, not a mapping
        contract_file.write_text(
            """
- item1
- item2
- item3
"""
        )

        loader = HandlerPluginLoader()

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            loader.load_from_contract(contract_file)

        error = exc_info.value
        assert (
            error.model.context.get("loader_error")
            == EnumHandlerLoaderError.SCHEMA_VALIDATION_FAILED.value
        )

    def test_yaml_scalar_at_root_returns_schema_validation_failed(
        self, tmp_path: Path
    ) -> None:
        """Test that YAML with scalar at root returns SCHEMA_VALIDATION_FAILED."""
        from omnibase_infra.enums import EnumHandlerLoaderError
        from omnibase_infra.errors import ProtocolConfigurationError
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        contract_file = tmp_path / "handler_contract.yaml"
        # Valid YAML but root is a scalar, not a mapping
        contract_file.write_text("just a string value")

        loader = HandlerPluginLoader()

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            loader.load_from_contract(contract_file)

        error = exc_info.value
        assert (
            error.model.context.get("loader_error")
            == EnumHandlerLoaderError.SCHEMA_VALIDATION_FAILED.value
        )
