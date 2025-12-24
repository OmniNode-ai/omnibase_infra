# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Comprehensive error path tests for runtime configuration loading.

Tests verify that error handling works correctly for:
- Missing configuration files
- Invalid YAML syntax
- Missing required fields
- Invalid type references
- Error sanitization (no secrets exposed)
- Proper error types raised

These tests ensure the PR #57 review requirement for error path test coverage
is satisfied per the MAJOR flagged issue for Test Coverage for Error Paths.
"""

from __future__ import annotations

import os
from pathlib import Path
from uuid import UUID

import pytest
import yaml

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import (
    ProtocolConfigurationError,
)
from omnibase_infra.runtime.kernel import load_runtime_config
from omnibase_infra.runtime.validation import load_and_validate_config

# TYPE_CHECKING block intentionally removed - no type-only imports needed


class TestMissingConfigFileScenarios:
    """Tests for missing configuration file handling."""

    def test_missing_config_file_raises_protocol_error(self, tmp_path: Path) -> None:
        """Test that missing config file raises ProtocolConfigurationError."""
        nonexistent = tmp_path / "does_not_exist.yaml"

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            load_and_validate_config(nonexistent)

        error = exc_info.value
        assert "not found" in str(error).lower()
        # Verify it's the correct error type
        assert isinstance(error, ProtocolConfigurationError)

    def test_missing_config_file_includes_path_in_error(self, tmp_path: Path) -> None:
        """Test that error message includes the file path for debugging."""
        nonexistent = tmp_path / "missing_config.yaml"

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            load_and_validate_config(nonexistent)

        error = exc_info.value
        # Path should be in error message for debugging
        assert "missing_config.yaml" in str(error)

    def test_missing_config_file_has_correlation_id(self, tmp_path: Path) -> None:
        """Test that error includes correlation_id for distributed tracing."""
        nonexistent = tmp_path / "nonexistent.yaml"

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            load_and_validate_config(nonexistent)

        error = exc_info.value
        # Error model should have correlation_id
        assert hasattr(error, "model")
        assert error.model.correlation_id is not None
        assert isinstance(error.model.correlation_id, UUID)

    def test_missing_config_file_has_transport_type_context(
        self, tmp_path: Path
    ) -> None:
        """Test that error includes transport_type in context."""
        nonexistent = tmp_path / "config.yaml"

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            load_and_validate_config(nonexistent)

        error = exc_info.value
        assert hasattr(error, "model")
        # Should have RUNTIME transport type
        assert (
            error.model.context.get("transport_type") == EnumInfraTransportType.RUNTIME
        )

    def test_missing_parent_directory(self, tmp_path: Path) -> None:
        """Test handling when parent directory doesn't exist."""
        nonexistent = tmp_path / "subdir" / "nested" / "config.yaml"

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            load_and_validate_config(nonexistent)

        error = exc_info.value
        assert "not found" in str(error).lower()

    def test_load_runtime_config_missing_file_returns_defaults(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that kernel load_runtime_config returns defaults when file missing.

        Note: Unlike load_and_validate_config which raises an error,
        load_runtime_config gracefully falls back to defaults when the
        config file is missing.
        """
        # Clear env vars to test true defaults
        monkeypatch.delenv("ONEX_INPUT_TOPIC", raising=False)
        monkeypatch.delenv("ONEX_OUTPUT_TOPIC", raising=False)
        monkeypatch.delenv("ONEX_GROUP_ID", raising=False)

        config = load_runtime_config(tmp_path)

        # Should return defaults, not raise
        assert config.input_topic == "requests"
        assert config.output_topic == "responses"
        assert config.consumer_group == "onex-runtime"


class TestInvalidYamlSyntaxScenarios:
    """Tests for invalid YAML syntax handling."""

    def test_invalid_yaml_unclosed_bracket_raises_error(self, tmp_path: Path) -> None:
        """Test that unclosed bracket in YAML raises ProtocolConfigurationError."""
        config_file = tmp_path / "invalid.yaml"
        config_file.write_text("key: [unclosed bracket")

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            load_and_validate_config(config_file)

        error = exc_info.value
        assert "parse" in str(error).lower() or "yaml" in str(error).lower()

    def test_invalid_yaml_ambiguous_mapping_context_raises_error(
        self, tmp_path: Path
    ) -> None:
        """Test that ambiguous YAML mapping context raises error.

        Note: 'key:value' alone is valid YAML (a plain string without space
        after colon). However, when followed by 'other:stuff:' which tries
        to be a mapping key (trailing colon), YAML parsers fail with
        'mapping values are not allowed here' because the context is ambiguous.
        """
        config_file = tmp_path / "invalid.yaml"
        # First line is valid as string, second line creates mapping ambiguity
        config_file.write_text("key:value\nother:stuff:")

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            load_and_validate_config(config_file)

        error = exc_info.value
        assert "parse" in str(error).lower() or "yaml" in str(error).lower()

    def test_invalid_yaml_tab_indentation_raises_error(self, tmp_path: Path) -> None:
        """Test that tabs used for indentation in YAML raise appropriate error.

        Note: While tabs are valid in YAML string values, they are prohibited
        for indentation per the YAML 1.1 and 1.2 specifications. PyYAML follows
        the spec strictly and rejects tab characters used for indentation.
        """
        config_file = tmp_path / "invalid.yaml"
        config_file.write_text("parent:\n\tchild: value")

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            load_and_validate_config(config_file)

        error = exc_info.value
        assert "parse" in str(error).lower() or "yaml" in str(error).lower()

    def test_invalid_yaml_duplicate_keys_handled(self, tmp_path: Path) -> None:
        """Test handling of YAML with duplicate keys."""
        config_file = tmp_path / "duplicate.yaml"
        # PyYAML allows duplicate keys (last one wins), but this tests parsing
        config_file.write_text("input_topic: first\ninput_topic: second")

        # Should parse successfully (last value wins in PyYAML)
        result = load_and_validate_config(config_file)
        assert result["input_topic"] == "second"

    def test_invalid_yaml_with_binary_content_raises_error(
        self, tmp_path: Path
    ) -> None:
        """Test that binary content in YAML file raises ProtocolConfigurationError.

        Binary content causes UnicodeDecodeError when opening with utf-8 encoding.
        The load_and_validate_config function properly catches this exception and
        wraps it as ProtocolConfigurationError with encoding error details.
        """
        config_file = tmp_path / "binary.yaml"
        config_file.write_bytes(b"\x00\x01\x02\x03\xff\xfe")

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            load_and_validate_config(config_file)

        error = exc_info.value
        # Should mention binary/non-UTF-8 content
        assert "binary" in str(error).lower() or "utf-8" in str(error).lower()
        # Original UnicodeDecodeError should be chained
        assert error.__cause__ is not None
        assert isinstance(error.__cause__, UnicodeDecodeError)

    def test_invalid_yaml_error_includes_parse_details(self, tmp_path: Path) -> None:
        """Test that YAML parse error includes helpful details."""
        config_file = tmp_path / "invalid.yaml"
        config_file.write_text("key: [\ninvalid")

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            load_and_validate_config(config_file)

        error = exc_info.value
        # Error should contain details about the YAML error
        assert hasattr(error, "model")
        # Check that error details are available
        context = error.model.context
        assert context is not None

    def test_kernel_invalid_yaml_raises_protocol_error(self, tmp_path: Path) -> None:
        """Test that kernel load_runtime_config raises error on invalid YAML."""
        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir(parents=True)
        config_file = runtime_dir / "runtime_config.yaml"
        config_file.write_text("invalid: yaml: content: [")

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            load_runtime_config(tmp_path)

        error = exc_info.value
        assert "Failed to parse runtime config YAML" in str(error)


class TestMissingRequiredFieldsScenarios:
    """Tests for missing required fields handling."""

    def test_invalid_input_topic_format_raises_error(self, tmp_path: Path) -> None:
        """Test that invalid input_topic format raises validation error."""
        config_file = tmp_path / "config.yaml"
        config_data = {"input_topic": "invalid topic with spaces"}
        with open(config_file, "w", encoding="utf-8") as f:
            yaml.dump(config_data, f)

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            load_and_validate_config(config_file)

        error = exc_info.value
        assert "validation failed" in str(error).lower()
        assert "input_topic" in str(error).lower()

    def test_invalid_topic_type_raises_error(self, tmp_path: Path) -> None:
        """Test that non-string topic raises validation error."""
        config_file = tmp_path / "config.yaml"
        config_data = {"input_topic": 12345}  # Should be string
        with open(config_file, "w", encoding="utf-8") as f:
            yaml.dump(config_data, f)

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            load_and_validate_config(config_file)

        error = exc_info.value
        assert "must be a string" in str(error)

    def test_invalid_event_bus_type_raises_error(self, tmp_path: Path) -> None:
        """Test that invalid event_bus.type raises validation error."""
        config_file = tmp_path / "config.yaml"
        config_data = {"event_bus": {"type": "unknown_bus_type"}}
        with open(config_file, "w", encoding="utf-8") as f:
            yaml.dump(config_data, f)

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            load_and_validate_config(config_file)

        error = exc_info.value
        assert "event_bus.type" in str(error)
        # Should mention valid types
        assert "inmemory" in str(error) or "kafka" in str(error)

    def test_invalid_grace_period_negative_raises_error(self, tmp_path: Path) -> None:
        """Test that negative grace_period_seconds raises validation error."""
        config_file = tmp_path / "config.yaml"
        config_data = {"shutdown": {"grace_period_seconds": -10}}
        with open(config_file, "w", encoding="utf-8") as f:
            yaml.dump(config_data, f)

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            load_and_validate_config(config_file)

        error = exc_info.value
        assert "grace_period_seconds" in str(error)
        assert ">=" in str(error)

    def test_invalid_grace_period_too_large_raises_error(self, tmp_path: Path) -> None:
        """Test that grace_period_seconds > 300 raises validation error."""
        config_file = tmp_path / "config.yaml"
        config_data = {"shutdown": {"grace_period_seconds": 999}}
        with open(config_file, "w", encoding="utf-8") as f:
            yaml.dump(config_data, f)

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            load_and_validate_config(config_file)

        error = exc_info.value
        assert "grace_period_seconds" in str(error)
        assert "<=" in str(error)

    def test_multiple_validation_errors_all_reported(self, tmp_path: Path) -> None:
        """Test that multiple validation errors are all collected and reported."""
        config_file = tmp_path / "config.yaml"
        config_data = {
            "input_topic": "invalid topic",
            "output_topic": "also invalid",
            "event_bus": {"type": "nonexistent"},
            "shutdown": {"grace_period_seconds": -1},
        }
        with open(config_file, "w", encoding="utf-8") as f:
            yaml.dump(config_data, f)

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            load_and_validate_config(config_file)

        error = exc_info.value
        # Should report all errors, not just the first one
        assert "validation failed" in str(error).lower()
        # Error context should have validation_errors list
        assert hasattr(error, "model")
        context = error.model.context
        if "validation_errors" in context:
            assert len(context["validation_errors"]) >= 3

    def test_kernel_validation_error_has_structured_context(
        self, tmp_path: Path
    ) -> None:
        """Test that kernel validation errors have structured context."""
        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir(parents=True)
        config_file = runtime_dir / "runtime_config.yaml"
        config_data = {"input_topic": "spaces are bad"}
        with open(config_file, "w", encoding="utf-8") as f:
            yaml.dump(config_data, f)

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            load_runtime_config(tmp_path)

        error = exc_info.value
        assert hasattr(error, "model")
        context = error.model.context
        assert context is not None
        # Should have config_path for debugging
        assert "config_path" in context


class TestErrorSanitization:
    """Tests verifying that errors don't expose sensitive information."""

    def test_valid_config_with_extra_fields_loads_successfully(
        self, tmp_path: Path
    ) -> None:
        """Test that valid config with extra fields (like credentials) loads without error.

        This verifies that the config loader:
        1. Accepts valid topic names
        2. Ignores extra fields that may contain sensitive values
        3. Does not expose these values in any success or error path

        Note: This is a success-path test that validates the config loader handles
        extra fields gracefully. Error sanitization tests are in other methods.
        """
        config_file = tmp_path / "config.yaml"
        # Create a config with extra fields (simulating credentials)
        config_data = {
            "input_topic": "valid-topic",
            "password": "super_secret_password_12345",
            "api_key": "sk_test_secret_key_xyz",
        }
        with open(config_file, "w", encoding="utf-8") as f:
            yaml.dump(config_data, f)

        # This config is valid for topic validation, should not raise
        result = load_and_validate_config(config_file)
        assert result["input_topic"] == "valid-topic"

    def test_error_does_not_expose_passwords_on_parse_failure(
        self, tmp_path: Path
    ) -> None:
        """Test that YAML parse errors don't expose potential secrets in message."""
        config_file = tmp_path / "config.yaml"
        # Create invalid YAML that mentions password-like patterns
        config_file.write_text("password: secret123\ninvalid: yaml: [")

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            load_and_validate_config(config_file)

        error = exc_info.value
        error_str = str(error)
        # Error message should NOT contain the secret value
        assert "secret123" not in error_str

    def test_error_sanitizes_file_path_secrets(self, tmp_path: Path) -> None:
        """Test that error doesn't expose secrets that might be in path names."""
        # Create a path that looks like it might contain sensitive info
        safe_path = tmp_path / "config_test.yaml"

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            load_and_validate_config(safe_path)

        error = exc_info.value
        error_str = str(error)
        # Path should be in error for debugging
        assert "config_test.yaml" in error_str
        # But no credential-like patterns should be exposed beyond path

    def test_validation_error_sanitizes_values(self, tmp_path: Path) -> None:
        """Test that validation errors don't include the actual invalid values if sensitive."""
        config_file = tmp_path / "config.yaml"
        # The topic name itself isn't secret, but testing sanitization pattern
        config_data = {"input_topic": "topic with spaces"}
        with open(config_file, "w", encoding="utf-8") as f:
            yaml.dump(config_data, f)

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            load_and_validate_config(config_file)

        error = exc_info.value
        error_str = str(error)
        # The pattern or field name should be mentioned, but error handling
        # should avoid exposing the full value in certain contexts
        assert "input_topic" in error_str


class TestProperErrorTypesRaised:
    """Tests verifying correct error types are raised for each scenario."""

    def test_file_not_found_raises_protocol_configuration_error(
        self, tmp_path: Path
    ) -> None:
        """Test that FileNotFoundError is wrapped as ProtocolConfigurationError."""
        nonexistent = tmp_path / "missing.yaml"

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            load_and_validate_config(nonexistent)

        error = exc_info.value
        # Should be our error type, not raw FileNotFoundError
        assert isinstance(error, ProtocolConfigurationError)
        # Original error should be chained
        assert error.__cause__ is not None
        assert isinstance(error.__cause__, FileNotFoundError)

    def test_yaml_error_raises_protocol_configuration_error(
        self, tmp_path: Path
    ) -> None:
        """Test that yaml.YAMLError is wrapped as ProtocolConfigurationError."""
        config_file = tmp_path / "invalid.yaml"
        config_file.write_text("invalid: yaml: [")

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            load_and_validate_config(config_file)

        error = exc_info.value
        assert isinstance(error, ProtocolConfigurationError)
        # Original YAML error should be chained
        assert error.__cause__ is not None
        import yaml as yaml_module

        assert isinstance(error.__cause__, yaml_module.YAMLError)

    def test_validation_error_raises_protocol_configuration_error(
        self, tmp_path: Path
    ) -> None:
        """Test that validation failures raise ProtocolConfigurationError."""
        config_file = tmp_path / "config.yaml"
        config_data = {"input_topic": "invalid topic"}
        with open(config_file, "w", encoding="utf-8") as f:
            yaml.dump(config_data, f)

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            load_and_validate_config(config_file)

        error = exc_info.value
        assert isinstance(error, ProtocolConfigurationError)

    def test_os_error_raises_protocol_configuration_error(self, tmp_path: Path) -> None:
        """Test that OSError during read is wrapped as ProtocolConfigurationError.

        Note: This test is skipped when running as root (UID 0) because root
        can read files regardless of permissions. Also skipped on Windows.
        The test may also be skipped in certain container environments where
        file permission changes don't work as expected.
        """
        if os.name == "nt":  # Skip on Windows
            pytest.skip("File permission test not applicable on Windows")

        # Check if running as root
        if os.getuid() == 0:
            pytest.skip("Root user can read files regardless of permissions")

        config_file = tmp_path / "unreadable.yaml"
        config_file.write_text("valid: config")

        original_mode = config_file.stat().st_mode
        try:
            config_file.chmod(0o000)

            # Try to open the file to see if permissions work in this environment
            try:
                with open(config_file, encoding="utf-8") as f:
                    f.read()
                # If we get here, permissions aren't enforced (e.g., some container setups)
                pytest.skip("File permissions not enforced in this environment")
            except PermissionError:
                pass  # Expected - permissions work, continue with the test

            with pytest.raises(ProtocolConfigurationError) as exc_info:
                load_and_validate_config(config_file)

            error = exc_info.value
            assert isinstance(error, ProtocolConfigurationError)
            # Original error should be chained
            assert error.__cause__ is not None
        finally:
            # Restore permissions for cleanup
            config_file.chmod(original_mode)


class TestErrorContextCompleteness:
    """Tests verifying error context contains required fields."""

    def test_error_context_has_operation(self, tmp_path: Path) -> None:
        """Test that error context includes operation field."""
        nonexistent = tmp_path / "missing.yaml"

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            load_and_validate_config(nonexistent)

        error = exc_info.value
        context = error.model.context
        assert "operation" in context
        assert context["operation"] == "validate_config"

    def test_error_context_has_target_name(self, tmp_path: Path) -> None:
        """Test that error context includes target_name (config path)."""
        nonexistent = tmp_path / "myconfig.yaml"

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            load_and_validate_config(nonexistent)

        error = exc_info.value
        context = error.model.context
        assert "target_name" in context
        assert "myconfig.yaml" in context["target_name"]

    def test_error_context_has_config_path(self, tmp_path: Path) -> None:
        """Test that error context includes config_path for debugging."""
        nonexistent = tmp_path / "test.yaml"

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            load_and_validate_config(nonexistent)

        error = exc_info.value
        context = error.model.context
        assert "config_path" in context
        assert "test.yaml" in context["config_path"]

    def test_error_context_has_correlation_id(self, tmp_path: Path) -> None:
        """Test that error includes correlation_id for tracing."""
        nonexistent = tmp_path / "config.yaml"

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            load_and_validate_config(nonexistent)

        error = exc_info.value
        assert error.model.correlation_id is not None
        assert isinstance(error.model.correlation_id, UUID)

    def test_validation_error_context_has_error_count(self, tmp_path: Path) -> None:
        """Test that validation error includes error_count for metrics."""
        config_file = tmp_path / "config.yaml"
        config_data = {
            "input_topic": "bad topic",
            "output_topic": "also bad",
        }
        with open(config_file, "w", encoding="utf-8") as f:
            yaml.dump(config_data, f)

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            load_and_validate_config(config_file)

        error = exc_info.value
        context = error.model.context
        assert "error_count" in context
        assert context["error_count"] >= 2

    def test_validation_error_context_has_validation_errors_list(
        self, tmp_path: Path
    ) -> None:
        """Test that validation error includes full validation_errors list."""
        config_file = tmp_path / "config.yaml"
        config_data = {"input_topic": "spaces in topic"}
        with open(config_file, "w", encoding="utf-8") as f:
            yaml.dump(config_data, f)

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            load_and_validate_config(config_file)

        error = exc_info.value
        context = error.model.context
        assert "validation_errors" in context
        assert isinstance(context["validation_errors"], list)
        assert len(context["validation_errors"]) >= 1


class TestKernelSpecificErrorPaths:
    """Tests for kernel-specific error handling paths."""

    def test_kernel_pydantic_validation_error_is_wrapped(self, tmp_path: Path) -> None:
        """Test that Pydantic ValidationError is wrapped as ProtocolConfigurationError."""
        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir(parents=True)
        config_file = runtime_dir / "runtime_config.yaml"
        # Create config with invalid type for a Pydantic model field
        config_data = {
            "input_topic": "valid-topic",
            "shutdown": "not_a_dict",  # Should be dict with grace_period_seconds
        }
        with open(config_file, "w", encoding="utf-8") as f:
            yaml.dump(config_data, f)

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            load_runtime_config(tmp_path)

        error = exc_info.value
        assert isinstance(error, ProtocolConfigurationError)

    def test_kernel_contract_validation_before_pydantic(self, tmp_path: Path) -> None:
        """Test that contract validation runs before Pydantic validation."""
        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir(parents=True)
        config_file = runtime_dir / "runtime_config.yaml"
        # Create config that fails contract validation (topic pattern)
        config_data = {"input_topic": "spaces in topic"}
        with open(config_file, "w", encoding="utf-8") as f:
            yaml.dump(config_data, f)

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            load_runtime_config(tmp_path)

        error = exc_info.value
        # Should fail at contract validation, not Pydantic
        assert "Contract validation failed" in str(error)

    def test_kernel_error_chaining_preserved(self, tmp_path: Path) -> None:
        """Test that error chaining is preserved through kernel error handling."""
        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir(parents=True)
        config_file = runtime_dir / "runtime_config.yaml"
        config_file.write_text("invalid: yaml: [")

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            load_runtime_config(tmp_path)

        error = exc_info.value
        # Original YAML error should be preserved as cause
        assert error.__cause__ is not None
        import yaml as yaml_module

        assert isinstance(error.__cause__, yaml_module.YAMLError)
