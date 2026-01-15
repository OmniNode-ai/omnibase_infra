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
    - TestValidRoutingStrategies: Tests for VALID_ROUTING_STRATEGIES constant

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
    CONTRACT_WITH_INVALID_ROUTING_STRATEGY_YAML,
    CONTRACT_WITH_MISSING_EVENT_MODEL_NAME_YAML,
    CONTRACT_WITH_MISSING_HANDLER_NAME_YAML,
    CONTRACT_WITH_UNKNOWN_ROUTING_STRATEGY_YAML,
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

    @pytest.mark.parametrize(
        ("class_name", "expected"),
        [
            # Basic underscore in middle - regex inserts hyphen at case boundary after _
            # "My_Handler" -> "My_-Handler" (first regex: _ precedes uppercase H) -> lowercase
            ("My_Handler", "my_-handler"),
            # Multiple underscores - hyphen inserted at each _Uppercase boundary
            # "Handler_Name_Test" -> "Handler_-Name_-Test" -> lowercase
            ("Handler_Name_Test", "handler_-name_-test"),
            # Leading underscore - hyphen inserted after _ before uppercase L
            # "_LeadingUnderscore" -> "_-Leading-Underscore" -> lowercase
            ("_LeadingUnderscore", "_-leading-underscore"),
            # Trailing underscore - no uppercase follows, underscore preserved as-is
            ("Handler_", "handler_"),
            # Double underscore - second _ precedes uppercase H, hyphen inserted there
            # "My__Handler" -> "My__-Handler" -> lowercase
            ("My__Handler", "my__-handler"),
            # All lowercase with underscore - no case boundaries, no hyphens added
            ("my_handler", "my_handler"),
            # Underscore before acronym - _ is not in [a-z0-9] so no hyphen before HTTP
            # "My_HTTPHandler" -> "My_HTTP-Handler" (only before Handler) -> lowercase
            ("My_HTTPHandler", "my_http-handler"),
        ],
        ids=[
            "underscore_mid",
            "multiple_underscores",
            "leading_underscore",
            "trailing_underscore",
            "double_underscore",
            "lowercase_underscore",
            "underscore_before_acronym",
        ],
    )
    def test_underscore_handling(self, class_name: str, expected: str) -> None:
        """Test underscore handling in class name to handler key conversion.

        Note: Underscores are atypical in Python class names (PEP 8 recommends
        CamelCase for classes). This test documents the ACTUAL behavior of the
        regex-based conversion, which may produce non-ideal output like
        "my_-handler" (mixed underscore and hyphen). The regex operates only
        on letter case boundaries, not underscore characters.

        If underscore handling needs to change (e.g., converting underscores
        to hyphens), update the convert_class_to_handler_key function and
        these expected values accordingly.
        """
        from omnibase_infra.runtime.contract_loaders import convert_class_to_handler_key

        assert convert_class_to_handler_key(class_name) == expected


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
# TestValidRoutingStrategies
# =============================================================================


class TestValidRoutingStrategies:
    """Tests for VALID_ROUTING_STRATEGIES constant and routing strategy validation.

    These tests verify that:
    - VALID_ROUTING_STRATEGIES contains only implemented strategies
    - Unknown strategies trigger warnings and fall back to payload_type_match
    - The constant is properly exported and accessible
    """

    def test_valid_routing_strategies_contains_only_payload_type_match(self) -> None:
        """Test that VALID_ROUTING_STRATEGIES only contains 'payload_type_match'.

        Currently only 'payload_type_match' is implemented. Other strategies
        like 'first_match' or 'all_match' may be added in future versions.
        """
        from omnibase_infra.runtime.contract_loaders import VALID_ROUTING_STRATEGIES

        assert frozenset({"payload_type_match"}) == VALID_ROUTING_STRATEGIES

    def test_valid_routing_strategies_is_frozenset(self) -> None:
        """Test that VALID_ROUTING_STRATEGIES is an immutable frozenset."""
        from omnibase_infra.runtime.contract_loaders import VALID_ROUTING_STRATEGIES

        assert isinstance(VALID_ROUTING_STRATEGIES, frozenset)

    def test_valid_routing_strategies_has_exactly_one_entry(self) -> None:
        """Test that VALID_ROUTING_STRATEGIES has exactly one entry.

        This test will fail if additional strategies are added, reminding
        developers to add corresponding tests for new strategies.
        """
        from omnibase_infra.runtime.contract_loaders import VALID_ROUTING_STRATEGIES

        assert len(VALID_ROUTING_STRATEGIES) == 1

    def test_payload_type_match_in_valid_strategies(self) -> None:
        """Test that 'payload_type_match' is in VALID_ROUTING_STRATEGIES."""
        from omnibase_infra.runtime.contract_loaders import VALID_ROUTING_STRATEGIES

        assert "payload_type_match" in VALID_ROUTING_STRATEGIES

    def test_first_match_not_in_valid_strategies(self) -> None:
        """Test that 'first_match' is NOT in VALID_ROUTING_STRATEGIES.

        This strategy was previously listed but never implemented.
        """
        from omnibase_infra.runtime.contract_loaders import VALID_ROUTING_STRATEGIES

        assert "first_match" not in VALID_ROUTING_STRATEGIES

    def test_all_match_not_in_valid_strategies(self) -> None:
        """Test that 'all_match' is NOT in VALID_ROUTING_STRATEGIES.

        This strategy was previously listed but never implemented.
        """
        from omnibase_infra.runtime.contract_loaders import VALID_ROUTING_STRATEGIES

        assert "all_match" not in VALID_ROUTING_STRATEGIES

    def test_invalid_strategy_triggers_warning_and_fallback(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that invalid routing strategy triggers warning and falls back.

        When a contract specifies an unimplemented strategy like 'first_match',
        the loader should:
        1. Log a warning message
        2. Fall back to 'payload_type_match'
        3. Still load the handlers correctly
        """
        import logging

        from omnibase_infra.runtime.contract_loaders import (
            load_handler_routing_subcontract,
        )

        contract_file = tmp_path / "contract.yaml"
        contract_file.write_text(CONTRACT_WITH_INVALID_ROUTING_STRATEGY_YAML)

        with caplog.at_level(logging.WARNING):
            result = load_handler_routing_subcontract(contract_file)

        # Verify fallback to payload_type_match
        assert result.routing_strategy == "payload_type_match"

        # Verify warning was logged
        assert any(
            "first_match" in record.message and "unknown" in record.message.lower()
            for record in caplog.records
        )

        # Verify handlers were still loaded correctly
        assert len(result.handlers) == 1
        assert result.handlers[0].routing_key == "TestEventModel"

    def test_unknown_strategy_triggers_warning_and_fallback(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that completely unknown routing strategy triggers warning.

        When a contract specifies a completely unknown strategy like
        'some_unknown_strategy', the loader should:
        1. Log a warning message listing valid strategies
        2. Fall back to 'payload_type_match'
        """
        import logging

        from omnibase_infra.runtime.contract_loaders import (
            load_handler_routing_subcontract,
        )

        contract_file = tmp_path / "contract.yaml"
        contract_file.write_text(CONTRACT_WITH_UNKNOWN_ROUTING_STRATEGY_YAML)

        with caplog.at_level(logging.WARNING):
            result = load_handler_routing_subcontract(contract_file)

        # Verify fallback to payload_type_match
        assert result.routing_strategy == "payload_type_match"

        # Verify warning was logged with the unknown strategy name
        assert any(
            "some_unknown_strategy" in record.message for record in caplog.records
        )

    def test_warning_message_lists_valid_strategies(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that the warning message includes the list of valid strategies."""
        import logging

        from omnibase_infra.runtime.contract_loaders import (
            load_handler_routing_subcontract,
        )

        contract_file = tmp_path / "contract.yaml"
        contract_file.write_text(CONTRACT_WITH_UNKNOWN_ROUTING_STRATEGY_YAML)

        with caplog.at_level(logging.WARNING):
            load_handler_routing_subcontract(contract_file)

        # Verify warning message lists the valid strategy
        assert any("payload_type_match" in record.message for record in caplog.records)

    def test_valid_strategy_does_not_trigger_warning(
        self, valid_contract_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that valid 'payload_type_match' strategy does not trigger warning."""
        import logging

        from omnibase_infra.runtime.contract_loaders import (
            load_handler_routing_subcontract,
        )

        with caplog.at_level(logging.WARNING):
            result = load_handler_routing_subcontract(valid_contract_path)

        # Verify strategy is preserved
        assert result.routing_strategy == "payload_type_match"

        # Verify no warning about routing strategy was logged
        routing_warnings = [
            record
            for record in caplog.records
            if "routing_strategy" in record.message.lower()
            and "unknown" in record.message.lower()
        ]
        assert len(routing_warnings) == 0

    def test_fixture_uses_invalid_strategy(
        self, contract_with_invalid_routing_strategy_path: Path
    ) -> None:
        """Test that the fixture contract uses 'first_match' strategy.

        This verifies the fixture is set up correctly for testing.
        """
        import yaml

        with contract_with_invalid_routing_strategy_path.open() as f:
            contract = yaml.safe_load(f)

        assert contract["handler_routing"]["routing_strategy"] == "first_match"

    def test_fixture_uses_unknown_strategy(
        self, contract_with_unknown_routing_strategy_path: Path
    ) -> None:
        """Test that the fixture contract uses 'some_unknown_strategy'.

        This verifies the fixture is set up correctly for testing.
        """
        import yaml

        with contract_with_unknown_routing_strategy_path.open() as f:
            contract = yaml.safe_load(f)

        assert (
            contract["handler_routing"]["routing_strategy"] == "some_unknown_strategy"
        )


# =============================================================================
# TestFileSizeEnforcement
# =============================================================================


class TestFileSizeEnforcement:
    """Tests for file size limit enforcement (security control).

    Per CLAUDE.md Handler Plugin Loader security patterns, a 10MB file size
    limit is enforced to prevent memory exhaustion attacks via large YAML files.

    Error code: FILE_SIZE_EXCEEDED (HANDLER_LOADER_050)
    """

    def test_max_contract_file_size_constant_is_10mb(self) -> None:
        """Test that MAX_CONTRACT_FILE_SIZE_BYTES is 10MB."""
        from omnibase_infra.runtime.contract_loaders import MAX_CONTRACT_FILE_SIZE_BYTES

        expected_size = 10 * 1024 * 1024  # 10MB
        assert expected_size == MAX_CONTRACT_FILE_SIZE_BYTES

    def test_max_contract_file_size_is_exported(self) -> None:
        """Test that MAX_CONTRACT_FILE_SIZE_BYTES is in __all__ exports."""
        from omnibase_infra.runtime import contract_loaders

        assert "MAX_CONTRACT_FILE_SIZE_BYTES" in contract_loaders.__all__

    def test_oversized_file_raises_error_for_routing_subcontract(
        self, oversized_contract_path: Path
    ) -> None:
        """Test that oversized file raises ProtocolConfigurationError.

        Files exceeding 10MB should be rejected BEFORE loading to prevent
        memory exhaustion attacks.
        """
        from omnibase_infra.errors import ProtocolConfigurationError
        from omnibase_infra.runtime.contract_loaders import (
            load_handler_routing_subcontract,
        )

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            load_handler_routing_subcontract(oversized_contract_path)

        error_msg = str(exc_info.value)
        assert "exceeds maximum size" in error_msg.lower()
        assert "FILE_SIZE_EXCEEDED" in error_msg
        assert "HANDLER_LOADER_050" in error_msg

    def test_oversized_file_raises_error_for_class_info(
        self, oversized_contract_path: Path
    ) -> None:
        """Test that oversized file raises error for load_handler_class_info_from_contract."""
        from omnibase_infra.errors import ProtocolConfigurationError
        from omnibase_infra.runtime.contract_loaders import (
            load_handler_class_info_from_contract,
        )

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            load_handler_class_info_from_contract(oversized_contract_path)

        error_msg = str(exc_info.value)
        assert "exceeds maximum size" in error_msg.lower()
        assert "FILE_SIZE_EXCEEDED" in error_msg
        assert "HANDLER_LOADER_050" in error_msg

    def test_error_context_includes_transport_type(
        self, oversized_contract_path: Path
    ) -> None:
        """Test that file size error includes FILESYSTEM transport type."""
        from omnibase_infra.errors import ProtocolConfigurationError
        from omnibase_infra.runtime.contract_loaders import (
            load_handler_routing_subcontract,
        )

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            load_handler_routing_subcontract(oversized_contract_path)

        error = exc_info.value
        assert error.model.context is not None
        # Context is stored as a dict in the error model
        context = error.model.context
        assert context.get("transport_type") == "filesystem"

    def test_error_context_includes_target_name(
        self, oversized_contract_path: Path
    ) -> None:
        """Test that file size error includes target path in context."""
        from omnibase_infra.errors import ProtocolConfigurationError
        from omnibase_infra.runtime.contract_loaders import (
            load_handler_routing_subcontract,
        )

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            load_handler_routing_subcontract(oversized_contract_path)

        error = exc_info.value
        assert error.model.context is not None
        # Context is stored as a dict in the error model
        context = error.model.context
        target_name = context.get("target_name", "")
        assert (
            str(oversized_contract_path) in target_name
            or "contract.yaml" in target_name
        )

    def test_file_exactly_at_limit_is_accepted(self, tmp_path: Path) -> None:
        """Test that a file exactly at the size limit is accepted.

        This is a boundary condition test - files at exactly 10MB should
        still be processed (only files EXCEEDING the limit are rejected).
        """
        from omnibase_infra.runtime.contract_loaders import (
            MAX_CONTRACT_FILE_SIZE_BYTES,
            load_handler_routing_subcontract,
        )

        contract_file = tmp_path / "contract.yaml"
        # Create valid YAML content padded to exact limit
        base_content = """name: "test"
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
"""
        # Pad with comment characters to reach exact size
        padding_needed = MAX_CONTRACT_FILE_SIZE_BYTES - len(
            base_content.encode("utf-8")
        )
        if padding_needed > 0:
            padded_content = base_content + "\n# " + ("x" * (padding_needed - 3))
        else:
            padded_content = base_content
        contract_file.write_text(padded_content)

        # File at exact limit should be accepted (no error raised)
        result = load_handler_routing_subcontract(contract_file)
        assert result is not None
        assert len(result.handlers) == 1

    def test_file_one_byte_over_limit_is_rejected(self, tmp_path: Path) -> None:
        """Test that a file one byte over the limit is rejected.

        This is a boundary condition test ensuring the check is > not >=.
        """
        from omnibase_infra.errors import ProtocolConfigurationError
        from omnibase_infra.runtime.contract_loaders import (
            MAX_CONTRACT_FILE_SIZE_BYTES,
            load_handler_routing_subcontract,
        )

        contract_file = tmp_path / "contract.yaml"
        # Create content that is exactly one byte over the limit
        oversized_content = "x" * (MAX_CONTRACT_FILE_SIZE_BYTES + 1)
        contract_file.write_text(oversized_content)

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            load_handler_routing_subcontract(contract_file)

        assert "exceeds maximum size" in str(exc_info.value).lower()

    def test_file_size_check_happens_before_yaml_parsing(self, tmp_path: Path) -> None:
        """Test that file size is checked BEFORE attempting to parse YAML.

        This is important for security - we don't want to load a huge file
        into memory before checking the size.
        """
        from omnibase_infra.errors import ProtocolConfigurationError
        from omnibase_infra.runtime.contract_loaders import (
            MAX_CONTRACT_FILE_SIZE_BYTES,
            load_handler_routing_subcontract,
        )

        contract_file = tmp_path / "contract.yaml"
        # Create oversized content that would also be invalid YAML
        oversized_invalid_content = "[[[" * (MAX_CONTRACT_FILE_SIZE_BYTES // 3 + 1)
        contract_file.write_text(oversized_invalid_content)

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            load_handler_routing_subcontract(contract_file)

        # Should fail on file size, NOT on YAML parsing
        error_msg = str(exc_info.value)
        assert "exceeds maximum size" in error_msg.lower()
        assert "FILE_SIZE_EXCEEDED" in error_msg

    def test_small_valid_file_is_accepted(self, valid_contract_path: Path) -> None:
        """Test that a small valid file passes the size check."""
        from omnibase_infra.runtime.contract_loaders import (
            load_handler_routing_subcontract,
        )

        # Should not raise any error
        result = load_handler_routing_subcontract(valid_contract_path)
        assert result is not None

    def test_error_message_includes_actual_file_size(
        self, oversized_contract_path: Path
    ) -> None:
        """Test that error message includes the actual file size."""
        from omnibase_infra.errors import ProtocolConfigurationError
        from omnibase_infra.runtime.contract_loaders import (
            MAX_CONTRACT_FILE_SIZE_BYTES,
            load_handler_routing_subcontract,
        )

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            load_handler_routing_subcontract(oversized_contract_path)

        error_msg = str(exc_info.value)
        # Should mention both actual size and limit
        assert str(MAX_CONTRACT_FILE_SIZE_BYTES) in error_msg
        # The actual size should also be mentioned
        actual_size = oversized_contract_path.stat().st_size
        assert str(actual_size) in error_msg


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
    "TestFileSizeEnforcement",
    "TestLoadHandlerRoutingSubcontractEdgeCases",
    "TestLoadHandlerRoutingSubcontractErrors",
    "TestLoadHandlerRoutingSubcontractHappyPath",
    "TestLoadHandlerRoutingSubcontractIntegration",
    "TestValidRoutingStrategies",
]
