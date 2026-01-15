# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Tests for handler routing loader utility.

This module tests the shared handler routing loader that loads handler routing
configuration from contract.yaml files. The loader is part of the contract-driven
orchestrator pattern introduced in OMN-1316.

Test Categories:
    - TestConvertClassToHandlerKey: Tests for the class-to-handler-key conversion
    - TestLoadHandlerRoutingSubcontractHappyPath: Tests for successful loading
    - TestLoadHandlerRoutingSubcontractErrors: Tests for error handling
    - TestLoadHandlerRoutingSubcontractEdgeCases: Tests for edge cases

Part of OMN-1316: Extract handler routing loader to shared utility.

Running Tests:
    # Run all handler routing loader tests:
    pytest tests/unit/runtime/contract_loaders/test_handler_routing_loader.py -v

    # Run specific test class:
    pytest tests/unit/runtime/contract_loaders/test_handler_routing_loader.py::TestConvertClassToHandlerKey -v
"""

from __future__ import annotations

from pathlib import Path

import pytest

from .conftest import (
    CONTRACT_WITH_MISSING_EVENT_MODEL_NAME_YAML,
    CONTRACT_WITH_MISSING_HANDLER_NAME_YAML,
)

# =============================================================================
# TestConvertClassToHandlerKey
# =============================================================================


class TestConvertClassToHandlerKey:
    """Tests for convert_class_to_handler_key() function.

    This function converts CamelCase handler class names to kebab-case
    handler keys as used in ServiceHandlerRegistry.
    """

    def test_standard_camel_case_conversion(self) -> None:
        """Test standard CamelCase to kebab-case conversion."""
        from omnibase_infra.runtime.contract_loaders import convert_class_to_handler_key

        assert (
            convert_class_to_handler_key("HandlerNodeIntrospected")
            == "handler-node-introspected"
        )

    def test_runtime_tick_handler_conversion(self) -> None:
        """Test HandlerRuntimeTick conversion."""
        from omnibase_infra.runtime.contract_loaders import convert_class_to_handler_key

        assert (
            convert_class_to_handler_key("HandlerRuntimeTick") == "handler-runtime-tick"
        )

    def test_registration_acked_handler_conversion(self) -> None:
        """Test HandlerNodeRegistrationAcked conversion."""
        from omnibase_infra.runtime.contract_loaders import convert_class_to_handler_key

        assert (
            convert_class_to_handler_key("HandlerNodeRegistrationAcked")
            == "handler-node-registration-acked"
        )

    def test_heartbeat_handler_conversion(self) -> None:
        """Test HandlerNodeHeartbeat conversion."""
        from omnibase_infra.runtime.contract_loaders import convert_class_to_handler_key

        assert (
            convert_class_to_handler_key("HandlerNodeHeartbeat")
            == "handler-node-heartbeat"
        )

    def test_simple_single_word_class(self) -> None:
        """Test single word class name."""
        from omnibase_infra.runtime.contract_loaders import convert_class_to_handler_key

        assert convert_class_to_handler_key("Handler") == "handler"

    def test_two_word_class(self) -> None:
        """Test two word class name."""
        from omnibase_infra.runtime.contract_loaders import convert_class_to_handler_key

        assert convert_class_to_handler_key("MyHandler") == "my-handler"

    def test_acronym_handling(self) -> None:
        """Test handling of uppercase acronyms.

        The function inserts hyphens before uppercase letters that follow
        lowercase letters, and before uppercase letters that follow other
        uppercase+lowercase sequences.
        """
        from omnibase_infra.runtime.contract_loaders import convert_class_to_handler_key

        # MyHTTPHandler -> my-http-handler
        assert convert_class_to_handler_key("MyHTTPHandler") == "my-http-handler"

    def test_consecutive_uppercase(self) -> None:
        """Test handling of consecutive uppercase letters."""
        from omnibase_infra.runtime.contract_loaders import convert_class_to_handler_key

        # HTTPHandler -> http-handler (consecutive uppercase at start)
        assert convert_class_to_handler_key("HTTPHandler") == "http-handler"

    def test_all_lowercase(self) -> None:
        """Test already lowercase string."""
        from omnibase_infra.runtime.contract_loaders import convert_class_to_handler_key

        assert convert_class_to_handler_key("handler") == "handler"

    def test_with_numbers(self) -> None:
        """Test handler name with numbers."""
        from omnibase_infra.runtime.contract_loaders import convert_class_to_handler_key

        assert convert_class_to_handler_key("Handler2Event") == "handler2-event"

    def test_empty_string(self) -> None:
        """Test empty string input."""
        from omnibase_infra.runtime.contract_loaders import convert_class_to_handler_key

        assert convert_class_to_handler_key("") == ""

    def test_underscore_handling(self) -> None:
        """Test that underscores in class names are handled consistently.

        Underscores are atypical in Python class names but may occur.
        The regex-based conversion preserves underscores since it only
        operates on letter case boundaries (CamelCase -> kebab-case),
        not on underscore characters.
        """
        from omnibase_infra.runtime.contract_loaders import convert_class_to_handler_key

        result = convert_class_to_handler_key("My_Handler")

        # Verify underscore is preserved (not stripped or converted to hyphen)
        assert "_" in result, f"Underscore should be preserved in '{result}'"

        # Verify result is lowercase (all outputs should be lowercase)
        assert result == result.lower(), f"Result should be lowercase: '{result}'"

        # Verify exact output: underscore preserved, hyphen added at _H boundary
        # "My_Handler" -> "My_-Handler" (regex 1) -> "my_-handler" (lowercase)
        assert result == "my_-handler", f"Expected 'my_-handler', got '{result}'"


# =============================================================================
# TestLoadHandlerRoutingSubcontractHappyPath
# =============================================================================


class TestLoadHandlerRoutingSubcontractHappyPath:
    """Tests for successful handler routing subcontract loading.

    These tests verify that valid contract.yaml files are correctly
    parsed and converted to ModelRoutingSubcontract instances.
    """

    def test_load_valid_contract(self, valid_contract_path: Path) -> None:
        """Test loading a valid contract with handler_routing section."""
        from omnibase_infra.models.routing import ModelRoutingSubcontract
        from omnibase_infra.runtime.contract_loaders import (
            load_handler_routing_subcontract,
        )

        result = load_handler_routing_subcontract(valid_contract_path)

        # Verify return type
        assert isinstance(result, ModelRoutingSubcontract)

        # Verify routing strategy
        assert result.routing_strategy == "payload_type_match"

        # Verify handlers are loaded
        assert len(result.handlers) == 2

    def test_load_minimal_contract(self, minimal_contract_path: Path) -> None:
        """Test loading a minimal valid contract."""
        from omnibase_infra.runtime.contract_loaders import (
            load_handler_routing_subcontract,
        )

        result = load_handler_routing_subcontract(minimal_contract_path)

        assert result.routing_strategy == "payload_type_match"
        assert len(result.handlers) == 1
        assert result.handlers[0].routing_key == "TestEventModel"
        assert result.handlers[0].handler_key == "test-handler"

    def test_version_defaults_to_1_0_0(self, valid_contract_path: Path) -> None:
        """Test that version defaults to 1.0.0 if not specified."""
        from omnibase_infra.runtime.contract_loaders import (
            load_handler_routing_subcontract,
        )

        result = load_handler_routing_subcontract(valid_contract_path)

        assert result.version.major == 1
        assert result.version.minor == 0
        assert result.version.patch == 0

    def test_default_handler_is_none(self, valid_contract_path: Path) -> None:
        """Test that default_handler is None if not specified."""
        from omnibase_infra.runtime.contract_loaders import (
            load_handler_routing_subcontract,
        )

        result = load_handler_routing_subcontract(valid_contract_path)

        assert result.default_handler is None

    def test_handler_key_conversion(self, valid_contract_path: Path) -> None:
        """Test that handler class names are converted to handler keys."""
        from omnibase_infra.runtime.contract_loaders import (
            load_handler_routing_subcontract,
        )

        result = load_handler_routing_subcontract(valid_contract_path)

        # Find the handler for ModelNodeIntrospectionEvent
        introspection_entry = next(
            (
                e
                for e in result.handlers
                if e.routing_key == "ModelNodeIntrospectionEvent"
            ),
            None,
        )

        assert introspection_entry is not None
        assert introspection_entry.handler_key == "handler-node-introspected"

    def test_routing_key_matches_event_model_name(
        self, valid_contract_path: Path
    ) -> None:
        """Test that routing_key matches the event_model.name from contract."""
        from omnibase_infra.runtime.contract_loaders import (
            load_handler_routing_subcontract,
        )

        result = load_handler_routing_subcontract(valid_contract_path)

        expected_routing_keys = {
            "ModelNodeIntrospectionEvent",
            "ModelRuntimeTick",
        }

        actual_routing_keys = {entry.routing_key for entry in result.handlers}

        assert expected_routing_keys == actual_routing_keys

    def test_empty_handlers_list_returns_empty_subcontract(
        self, contract_with_empty_handlers_path: Path
    ) -> None:
        """Test loading contract with empty handlers list."""
        from omnibase_infra.runtime.contract_loaders import (
            load_handler_routing_subcontract,
        )

        result = load_handler_routing_subcontract(contract_with_empty_handlers_path)

        assert result.routing_strategy == "payload_type_match"
        assert len(result.handlers) == 0


# =============================================================================
# TestLoadHandlerRoutingSubcontractErrors
# =============================================================================


class TestLoadHandlerRoutingSubcontractErrors:
    """Tests for error handling in handler routing subcontract loading.

    These tests verify that appropriate ProtocolConfigurationError
    exceptions are raised for various error conditions.
    """

    def test_missing_file_raises_error(self, nonexistent_contract_path: Path) -> None:
        """Test that missing contract file raises ProtocolConfigurationError."""
        from omnibase_infra.errors import ProtocolConfigurationError
        from omnibase_infra.runtime.contract_loaders import (
            load_handler_routing_subcontract,
        )

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            load_handler_routing_subcontract(nonexistent_contract_path)

        # Verify error message mentions "not found"
        assert "not found" in str(exc_info.value).lower()

    def test_invalid_yaml_raises_error(self, invalid_yaml_path: Path) -> None:
        """Test that invalid YAML syntax raises ProtocolConfigurationError."""
        from omnibase_infra.errors import ProtocolConfigurationError
        from omnibase_infra.runtime.contract_loaders import (
            load_handler_routing_subcontract,
        )

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            load_handler_routing_subcontract(invalid_yaml_path)

        # Verify error message mentions YAML error
        error_msg = str(exc_info.value).lower()
        assert "yaml" in error_msg or "syntax" in error_msg

    def test_empty_file_raises_error(self, empty_contract_path: Path) -> None:
        """Test that empty contract file raises ProtocolConfigurationError."""
        from omnibase_infra.errors import ProtocolConfigurationError
        from omnibase_infra.runtime.contract_loaders import (
            load_handler_routing_subcontract,
        )

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            load_handler_routing_subcontract(empty_contract_path)

        # Verify error message mentions "empty"
        assert "empty" in str(exc_info.value).lower()

    def test_whitespace_only_file_raises_error(
        self, whitespace_only_contract_path: Path
    ) -> None:
        """Test that file with only whitespace raises ProtocolConfigurationError."""
        from omnibase_infra.errors import ProtocolConfigurationError
        from omnibase_infra.runtime.contract_loaders import (
            load_handler_routing_subcontract,
        )

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            load_handler_routing_subcontract(whitespace_only_contract_path)

        # Verify error message mentions "empty"
        assert "empty" in str(exc_info.value).lower()

    def test_missing_handler_routing_section_raises_error(
        self, contract_without_routing_path: Path
    ) -> None:
        """Test that missing handler_routing section raises ProtocolConfigurationError."""
        from omnibase_infra.errors import ProtocolConfigurationError
        from omnibase_infra.runtime.contract_loaders import (
            load_handler_routing_subcontract,
        )

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            load_handler_routing_subcontract(contract_without_routing_path)

        # Verify error message mentions handler_routing
        assert "handler_routing" in str(exc_info.value).lower()

    def test_error_context_includes_operation(
        self, nonexistent_contract_path: Path
    ) -> None:
        """Test that error context includes operation name for debugging."""
        from omnibase_infra.errors import ProtocolConfigurationError
        from omnibase_infra.runtime.contract_loaders import (
            load_handler_routing_subcontract,
        )

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            load_handler_routing_subcontract(nonexistent_contract_path)

        # Verify error has context with operation
        error = exc_info.value
        assert error.model.context is not None

    def test_error_context_includes_target_name(
        self, nonexistent_contract_path: Path
    ) -> None:
        """Test that error context includes target path for debugging."""
        from omnibase_infra.errors import ProtocolConfigurationError
        from omnibase_infra.runtime.contract_loaders import (
            load_handler_routing_subcontract,
        )

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            load_handler_routing_subcontract(nonexistent_contract_path)

        # Verify error message mentions the file path
        error_msg = str(exc_info.value)
        assert str(nonexistent_contract_path) in error_msg


# =============================================================================
# TestLoadHandlerRoutingSubcontractEdgeCases
# =============================================================================


class TestLoadHandlerRoutingSubcontractEdgeCases:
    """Tests for edge cases in handler routing subcontract loading.

    These tests verify correct behavior for unusual but valid inputs,
    partial data, and boundary conditions.
    """

    def test_incomplete_handler_entries_skipped_with_warning(
        self,
        contract_with_incomplete_handler_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test that handler entries missing required fields are skipped.

        The loader should skip entries missing event_model.name or handler.name
        and log a warning, rather than failing entirely.

        Note: The fixture has 2 entries:
        - First entry has both event_model.name and handler.name (valid)
        - Second entry has only modules, no names (invalid, skipped)
        """
        import logging

        from omnibase_infra.runtime.contract_loaders import (
            load_handler_routing_subcontract,
        )

        with caplog.at_level(logging.WARNING):
            result = load_handler_routing_subcontract(
                contract_with_incomplete_handler_path
            )

        # First entry is valid, second is incomplete and skipped
        assert len(result.handlers) == 1
        assert result.handlers[0].routing_key == "TestEventModel"
        assert result.handlers[0].handler_key == "test-handler"

        # Should have logged warnings about skipped entries
        assert any("skipping" in record.message.lower() for record in caplog.records)

    def test_missing_event_model_name_skipped(self, tmp_path: Path) -> None:
        """Test that entries missing event_model.name are skipped."""
        from omnibase_infra.runtime.contract_loaders import (
            load_handler_routing_subcontract,
        )

        contract_file = tmp_path / "contract.yaml"
        contract_file.write_text(CONTRACT_WITH_MISSING_EVENT_MODEL_NAME_YAML)

        result = load_handler_routing_subcontract(contract_file)

        # Entry should be skipped
        assert len(result.handlers) == 0

    def test_missing_handler_name_skipped(self, tmp_path: Path) -> None:
        """Test that entries missing handler.name are skipped."""
        from omnibase_infra.runtime.contract_loaders import (
            load_handler_routing_subcontract,
        )

        contract_file = tmp_path / "contract.yaml"
        contract_file.write_text(CONTRACT_WITH_MISSING_HANDLER_NAME_YAML)

        result = load_handler_routing_subcontract(contract_file)

        # Entry should be skipped
        assert len(result.handlers) == 0

    def test_routing_strategy_defaults_if_not_specified(self, tmp_path: Path) -> None:
        """Test that routing_strategy defaults to 'payload_type_match'."""
        from omnibase_infra.runtime.contract_loaders import (
            load_handler_routing_subcontract,
        )

        # Create contract without routing_strategy
        contract_content = """
name: "test"
version: "1.0.0"
handler_routing:
  handlers:
    - event_model:
        name: "TestEvent"
        module: "test.models"
      handler:
        name: "TestHandler"
        module: "test.handlers"
"""
        contract_file = tmp_path / "contract.yaml"
        contract_file.write_text(contract_content)

        result = load_handler_routing_subcontract(contract_file)

        assert result.routing_strategy == "payload_type_match"

    def test_handlers_section_missing_defaults_to_empty_list(
        self, tmp_path: Path
    ) -> None:
        """Test that missing handlers section defaults to empty list."""
        from omnibase_infra.runtime.contract_loaders import (
            load_handler_routing_subcontract,
        )

        # Create contract with handler_routing but no handlers key
        contract_content = """
name: "test"
version: "1.0.0"
handler_routing:
  routing_strategy: "payload_type_match"
"""
        contract_file = tmp_path / "contract.yaml"
        contract_file.write_text(contract_content)

        result = load_handler_routing_subcontract(contract_file)

        assert len(result.handlers) == 0

    def test_absolute_path_works(self, valid_contract_path: Path) -> None:
        """Test that absolute paths work correctly."""
        from omnibase_infra.runtime.contract_loaders import (
            load_handler_routing_subcontract,
        )

        # Ensure we're using an absolute path
        abs_path = valid_contract_path.resolve()
        assert abs_path.is_absolute()

        result = load_handler_routing_subcontract(abs_path)

        assert result is not None
        assert len(result.handlers) == 2

    def test_relative_path_works(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that relative paths work correctly."""
        from omnibase_infra.runtime.contract_loaders import (
            load_handler_routing_subcontract,
        )

        # Create a contract file
        contract_file = tmp_path / "contract.yaml"
        contract_file.write_text("""
name: "test"
version: "1.0.0"
handler_routing:
  routing_strategy: "payload_type_match"
  handlers:
    - event_model:
        name: "TestEvent"
        module: "test.models"
      handler:
        name: "TestHandler"
        module: "test.handlers"
""")

        # Change to the tmp_path directory
        monkeypatch.chdir(tmp_path)

        # Use relative path
        relative_path = Path("contract.yaml")
        result = load_handler_routing_subcontract(relative_path)

        assert result is not None
        assert len(result.handlers) == 1


# =============================================================================
# TestLoadHandlerRoutingSubcontractIntegration
# =============================================================================


class TestLoadHandlerRoutingSubcontractIntegration:
    """Integration tests verifying the loader works with real orchestrator contracts.

    These tests verify that the loader correctly parses the actual
    contract.yaml from the node_registration_orchestrator.
    """

    def test_load_real_orchestrator_contract(self) -> None:
        """Test loading the real NodeRegistrationOrchestrator contract."""
        from omnibase_infra.nodes.node_registration_orchestrator.node import (
            _create_handler_routing_subcontract,
        )

        # The thin wrapper function should work correctly
        result = _create_handler_routing_subcontract()

        # Verify expected properties
        assert result.routing_strategy == "payload_type_match"
        assert len(result.handlers) >= 4  # At least 4 handlers defined

        # Verify expected handlers exist
        routing_keys = {entry.routing_key for entry in result.handlers}
        assert "ModelNodeIntrospectionEvent" in routing_keys
        assert "ModelRuntimeTick" in routing_keys
        assert "ModelNodeRegistrationAcked" in routing_keys
        assert "ModelNodeHeartbeatEvent" in routing_keys

    def test_real_contract_handler_keys_are_valid(self) -> None:
        """Test that real contract handler keys follow naming convention."""
        from omnibase_infra.nodes.node_registration_orchestrator.node import (
            _create_handler_routing_subcontract,
        )

        result = _create_handler_routing_subcontract()

        for entry in result.handlers:
            # All handler keys should start with "handler-"
            assert entry.handler_key.startswith("handler-"), (
                f"Handler key should start with 'handler-': {entry.handler_key}"
            )
            # All handler keys should be lowercase kebab-case
            assert entry.handler_key == entry.handler_key.lower(), (
                f"Handler key should be lowercase: {entry.handler_key}"
            )
            # No underscores (should use hyphens)
            assert "_" not in entry.handler_key, (
                f"Handler key should use hyphens, not underscores: {entry.handler_key}"
            )


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "TestConvertClassToHandlerKey",
    "TestLoadHandlerRoutingSubcontractEdgeCases",
    "TestLoadHandlerRoutingSubcontractErrors",
    "TestLoadHandlerRoutingSubcontractHappyPath",
    "TestLoadHandlerRoutingSubcontractIntegration",
]
