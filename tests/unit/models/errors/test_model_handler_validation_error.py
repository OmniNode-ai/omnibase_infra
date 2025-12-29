# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for ModelHandlerValidationError model.

Tests the canonical handler validation error model per OMN-1091 TDD requirements.

TDD Requirements:
    RED: Test that contract parse error produces structured error with rule_id
    RED: Test that security violation produces structured error with remediation_hint

This test module follows the RED phase of TDD - all tests should fail initially
until ModelHandlerValidationError is implemented.

.. versionadded:: 0.6.1
    Created as part of OMN-1091 structured validation and error reporting.
"""

from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from omnibase_infra.enums import (
    EnumHandlerErrorType,
    EnumHandlerSourceType,
    EnumHandlerType,
)

# These imports should work once model is created:
from omnibase_infra.models.errors import ModelHandlerValidationError
from omnibase_infra.models.handlers import ModelHandlerIdentifier


class TestModelHandlerValidationError:
    """Tests for ModelHandlerValidationError model.

    This test suite validates the structured error model for handler validation
    failures. Each test verifies specific aspects of error construction, formatting,
    and metadata handling.
    """

    # TDD RED: Contract parse error with rule_id
    def test_contract_parse_error_has_rule_id(self) -> None:
        """Contract parse errors must have a rule_id for traceability.

        Validates that contract parsing errors include structured rule_id
        for precise error identification and remediation tracking.

        Test: OMN-1091 TDD Requirement 1
        """
        handler_id = ModelHandlerIdentifier.from_handler_id("test-handler")

        error = ModelHandlerValidationError(
            error_type=EnumHandlerErrorType.CONTRACT_PARSE_ERROR,
            rule_id="CONTRACT-001",
            handler_identity=handler_id,
            source_type=EnumHandlerSourceType.CONTRACT,
            message="Failed to parse contract YAML",
            remediation_hint="Check YAML syntax at line 15",
            file_path="nodes/test/contract.yaml",
        )

        assert error.rule_id == "CONTRACT-001"
        assert error.error_type == EnumHandlerErrorType.CONTRACT_PARSE_ERROR

    # TDD RED: Security violation with remediation_hint
    def test_security_violation_has_remediation_hint(self) -> None:
        """Security violations must include remediation hints.

        Validates that security validation errors include actionable remediation
        hints to guide developers in fixing security issues.

        Test: OMN-1091 TDD Requirement 2
        """
        handler_id = ModelHandlerIdentifier.from_handler_id("http-handler")

        error = ModelHandlerValidationError(
            error_type=EnumHandlerErrorType.SECURITY_VALIDATION_ERROR,
            rule_id="SECURITY-001",
            handler_identity=handler_id,
            source_type=EnumHandlerSourceType.STATIC_ANALYSIS,
            message="Handler exposes sensitive method",
            remediation_hint="Prefix method with underscore: _internal_method",
        )

        assert (
            error.remediation_hint == "Prefix method with underscore: _internal_method"
        )
        assert error.error_type == EnumHandlerErrorType.SECURITY_VALIDATION_ERROR

    def test_format_for_ci_output(self) -> None:
        """Error should format for CI/GitHub Actions annotations.

        Validates that errors can be formatted for CI systems with
        file paths, line numbers, and rule IDs for automated tooling.
        """
        handler_id = ModelHandlerIdentifier.from_handler_id("test-handler")

        error = ModelHandlerValidationError(
            error_type=EnumHandlerErrorType.CONTRACT_VALIDATION_ERROR,
            rule_id="CONTRACT-002",
            handler_identity=handler_id,
            source_type=EnumHandlerSourceType.CONTRACT,
            message="Missing required field: input_model",
            remediation_hint="Add input_model field to contract.yaml",
            file_path="nodes/test/contract.yaml",
            line_number=10,
        )

        ci_output = error.format_for_ci()
        assert "nodes/test/contract.yaml" in ci_output
        assert "10" in ci_output
        assert "CONTRACT-002" in ci_output

    def test_format_for_logging_output(self) -> None:
        """Error should format for structured logging.

        Validates that errors can be formatted for structured logging
        systems with all relevant metadata preserved.
        """
        handler_id = ModelHandlerIdentifier.from_handler_id("db-handler")

        error = ModelHandlerValidationError(
            error_type=EnumHandlerErrorType.CONFIGURATION_ERROR,
            rule_id="CONFIG-001",
            handler_identity=handler_id,
            source_type=EnumHandlerSourceType.CONFIGURATION,
            message="Invalid pool size",
            remediation_hint="Set pool_size between 1 and 100",
        )

        log_output = error.format_for_logging()
        assert "CONFIG-001" in log_output
        assert "db-handler" in log_output
        assert "Invalid pool size" in log_output

    def test_is_blocking_for_security_errors(self) -> None:
        """Security errors should be blocking.

        Validates that security validation errors are classified as
        blocking errors that prevent handler registration.
        """
        handler_id = ModelHandlerIdentifier.from_handler_id("test-handler")

        error = ModelHandlerValidationError(
            error_type=EnumHandlerErrorType.SECURITY_VALIDATION_ERROR,
            rule_id="SECURITY-002",
            handler_identity=handler_id,
            source_type=EnumHandlerSourceType.STATIC_ANALYSIS,
            message="Potential credential exposure",
            remediation_hint="Remove credentials from config",
        )

        assert error.is_blocking() is True

    def test_error_chaining_with_caused_by(self) -> None:
        """Errors should support chaining via caused_by field.

        Validates that errors can be chained to preserve root cause
        information through multiple validation layers.
        """
        handler_id = ModelHandlerIdentifier.from_handler_id("test-handler")

        root_error = ModelHandlerValidationError(
            error_type=EnumHandlerErrorType.CONTRACT_PARSE_ERROR,
            rule_id="CONTRACT-001",
            handler_identity=handler_id,
            source_type=EnumHandlerSourceType.CONTRACT,
            message="YAML parse failed",
            remediation_hint="Fix YAML syntax",
            file_path="contract.yaml",
            line_number=5,
        )

        wrapper_error = ModelHandlerValidationError(
            error_type=EnumHandlerErrorType.CONTRACT_VALIDATION_ERROR,
            rule_id="CONTRACT-010",
            handler_identity=handler_id,
            source_type=EnumHandlerSourceType.CONTRACT,
            message="Contract validation failed",
            remediation_hint="See caused_by for details",
            caused_by=root_error,
        )

        assert wrapper_error.caused_by is not None
        assert wrapper_error.caused_by.rule_id == "CONTRACT-001"

    def test_to_structured_dict(self) -> None:
        """Error should serialize to structured dict for JSON.

        Validates that errors can be serialized to dictionaries
        for JSON output in APIs and structured logs.
        """
        handler_id = ModelHandlerIdentifier.from_handler_id("test-handler")

        error = ModelHandlerValidationError(
            error_type=EnumHandlerErrorType.ARCHITECTURE_VALIDATION_ERROR,
            rule_id="ARCH-001",
            handler_identity=handler_id,
            source_type=EnumHandlerSourceType.STATIC_ANALYSIS,
            message="Invalid handler pattern",
            remediation_hint="Use approved patterns",
            details={"pattern": "singleton", "expected": "factory"},
        )

        result = error.to_structured_dict()
        assert isinstance(result, dict)
        assert result["rule_id"] == "ARCH-001"
        assert result["error_type"] == "architecture_validation_error"

    def test_factory_from_contract_error(self) -> None:
        """Factory method should create contract errors easily.

        Validates that convenience factory method exists for creating
        contract validation errors with correct defaults.
        """
        handler_id = ModelHandlerIdentifier.from_handler_id("test-handler")

        error = ModelHandlerValidationError.from_contract_error(
            rule_id="CONTRACT-003",
            message="Missing node_type field",
            file_path="nodes/test/contract.yaml",
            remediation_hint="Add node_type: EFFECT to contract",
            handler_identity=handler_id,
        )

        assert error.error_type == EnumHandlerErrorType.CONTRACT_VALIDATION_ERROR
        assert error.source_type == EnumHandlerSourceType.CONTRACT
        assert error.rule_id == "CONTRACT-003"

    def test_factory_from_security_violation(self) -> None:
        """Factory method should create security errors easily.

        Validates that convenience factory method exists for creating
        security validation errors with correct defaults.
        """
        handler_id = ModelHandlerIdentifier.from_handler_id("http-handler")

        error = ModelHandlerValidationError.from_security_violation(
            rule_id="SECURITY-003",
            message="API key exposed in logs",
            remediation_hint="Use secret manager for API keys",
            handler_identity=handler_id,
        )

        assert error.error_type == EnumHandlerErrorType.SECURITY_VALIDATION_ERROR
        assert error.source_type == EnumHandlerSourceType.STATIC_ANALYSIS
        assert error.remediation_hint == "Use secret manager for API keys"

    def test_correlation_id_tracking(self) -> None:
        """Error should support correlation ID for distributed tracing.

        Validates that errors can track correlation IDs for distributed
        tracing and request correlation across services.
        """
        handler_id = ModelHandlerIdentifier.from_handler_id("test-handler")
        correlation_id = uuid4()

        error = ModelHandlerValidationError(
            error_type=EnumHandlerErrorType.REGISTRATION_ERROR,
            rule_id="REG-001",
            handler_identity=handler_id,
            source_type=EnumHandlerSourceType.REGISTRATION,
            message="Handler already registered",
            remediation_hint="Check for duplicate registrations",
            correlation_id=correlation_id,
        )

        assert error.correlation_id == correlation_id

    def test_model_is_frozen(self) -> None:
        """Model should be immutable (frozen).

        Validates that ModelHandlerValidationError instances are immutable
        to prevent accidental modification in error contexts.
        """
        handler_id = ModelHandlerIdentifier.from_handler_id("test-handler")

        error = ModelHandlerValidationError(
            error_type=EnumHandlerErrorType.TYPE_MISMATCH_ERROR,
            rule_id="TYPE-001",
            handler_identity=handler_id,
            source_type=EnumHandlerSourceType.DESCRIPTOR,
            message="Expected EFFECT, got COMPUTE",
            remediation_hint="Change handler type to EFFECT",
        )

        with pytest.raises(ValidationError):
            error.message = "new message"  # type: ignore[misc]


# Additional edge case tests


class TestModelHandlerValidationErrorEdgeCases:
    """Edge case tests for ModelHandlerValidationError.

    Tests boundary conditions and edge cases in error handling.
    """

    def test_optional_fields_can_be_none(self) -> None:
        """Optional fields should accept None values.

        Validates that optional fields like file_path, line_number, etc.
        can be omitted or set to None without validation errors.
        """
        handler_id = ModelHandlerIdentifier.from_handler_id("test-handler")

        error = ModelHandlerValidationError(
            error_type=EnumHandlerErrorType.CONFIGURATION_ERROR,
            rule_id="CONFIG-002",
            handler_identity=handler_id,
            source_type=EnumHandlerSourceType.CONFIGURATION,
            message="Configuration error",
            remediation_hint="Fix configuration",
        )

        assert error.file_path is None
        assert error.line_number is None
        assert error.correlation_id is None

    def test_details_field_accepts_arbitrary_dict(self) -> None:
        """Details field should accept arbitrary dict data.

        Validates that the details field can store arbitrary structured
        data for debugging and error context.
        """
        handler_id = ModelHandlerIdentifier.from_handler_id("test-handler")

        complex_details = {
            "expected_signature": "async def handle(event: ModelEvent) -> str",
            "actual_signature": "def handle(event) -> None",
            "violations": ["missing_async", "wrong_return_type"],
            "metadata": {"line": 42, "column": 10},
        }

        error = ModelHandlerValidationError(
            error_type=EnumHandlerErrorType.DESCRIPTOR_VALIDATION_ERROR,
            rule_id="DESC-001",
            handler_identity=handler_id,
            source_type=EnumHandlerSourceType.DESCRIPTOR,
            message="Handler signature mismatch",
            remediation_hint="Update handler signature",
            details=complex_details,
        )

        assert error.details is not None
        assert error.details["violations"] == ["missing_async", "wrong_return_type"]

    def test_error_with_minimal_fields(self) -> None:
        """Error can be created with only required fields.

        Validates that errors can be created with minimal information
        when full context is not available.
        """
        handler_id = ModelHandlerIdentifier.from_handler_id("minimal-handler")

        error = ModelHandlerValidationError(
            error_type=EnumHandlerErrorType.REGISTRATION_ERROR,
            rule_id="REG-002",
            handler_identity=handler_id,
            source_type=EnumHandlerSourceType.REGISTRATION,
            message="Registration failed",
            remediation_hint="Check handler registration process",
        )

        assert error.error_type == EnumHandlerErrorType.REGISTRATION_ERROR
        assert error.rule_id == "REG-002"
        assert error.message == "Registration failed"
        assert error.file_path is None  # Optional field
        assert error.line_number is None  # Optional field

    def test_format_for_ci_without_file_info(self) -> None:
        """CI format should handle missing file path gracefully.

        Validates that CI formatting works even when file path
        and line number are not available.
        """
        handler_id = ModelHandlerIdentifier.from_handler_id("test-handler")

        error = ModelHandlerValidationError(
            error_type=EnumHandlerErrorType.REGISTRATION_ERROR,
            rule_id="REG-003",
            handler_identity=handler_id,
            source_type=EnumHandlerSourceType.RUNTIME,
            message="Runtime error occurred",
            remediation_hint="Check handler implementation",
        )

        ci_output = error.format_for_ci()
        assert "REG-003" in ci_output
        assert "Runtime error occurred" in ci_output


class TestModelHandlerValidationErrorIntegration:
    """Integration tests for ModelHandlerValidationError.

    Tests integration with related models and systems.
    """

    def test_integration_with_handler_identifier(self) -> None:
        """Error should integrate with ModelHandlerIdentifier.

        Validates that ModelHandlerValidationError correctly uses
        ModelHandlerIdentifier for handler context.
        """
        handler_id = ModelHandlerIdentifier.from_node(
            node_path="nodes/http/node.py",
            handler_type=EnumHandlerType.EFFECT,
            handler_name="HTTP Effect Handler",
        )

        error = ModelHandlerValidationError(
            error_type=EnumHandlerErrorType.EXECUTION_SHAPE_VIOLATION,
            rule_id="EXEC-001",
            handler_identity=handler_id,
            source_type=EnumHandlerSourceType.RUNTIME,
            message="EFFECT handler cannot publish directly",
            remediation_hint="Return events instead of publishing",
        )

        assert error.handler_identity.handler_type == EnumHandlerType.EFFECT
        assert error.handler_identity.node_path == "nodes/http/node.py"

    def test_error_collection_multiple_errors(self) -> None:
        """Multiple errors can be collected and processed.

        Validates that multiple validation errors can be collected
        and processed together for batch error reporting.
        """
        handler_id = ModelHandlerIdentifier.from_handler_id("test-handler")

        errors = [
            ModelHandlerValidationError(
                error_type=EnumHandlerErrorType.CONTRACT_VALIDATION_ERROR,
                rule_id="CONTRACT-001",
                handler_identity=handler_id,
                source_type=EnumHandlerSourceType.CONTRACT,
                message="Missing input_model",
                remediation_hint="Add input_model to contract",
            ),
            ModelHandlerValidationError(
                error_type=EnumHandlerErrorType.CONTRACT_VALIDATION_ERROR,
                rule_id="CONTRACT-002",
                handler_identity=handler_id,
                source_type=EnumHandlerSourceType.CONTRACT,
                message="Invalid node_type",
                remediation_hint="Use EFFECT, COMPUTE, REDUCER, or ORCHESTRATOR",
            ),
        ]

        assert len(errors) == 2
        assert all(
            e.error_type == EnumHandlerErrorType.CONTRACT_VALIDATION_ERROR
            for e in errors
        )


# Run tests with: pytest tests/unit/models/errors/test_model_handler_validation_error.py -v
