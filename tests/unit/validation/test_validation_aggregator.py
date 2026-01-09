# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for ValidationAggregator.

This module provides comprehensive unit tests for the validation error
aggregation and reporting functionality used during application startup.

Tests cover:
    - Aggregator initialization
    - Error collection (single and batch)
    - Error counting and classification
    - Error grouping by type and source
    - Console output formatting
    - CI annotation formatting
    - Summary formatting
    - Exception raising for blocking errors
    - Clearing aggregated errors

Related:
    - OMN-1091: Structured Validation & Error Reporting for Handlers
"""

from __future__ import annotations

import pytest

from omnibase_infra.enums import (
    EnumHandlerErrorType,
    EnumHandlerSourceType,
)
from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.models.errors import ModelHandlerValidationError
from omnibase_infra.models.handlers import ModelHandlerIdentifier
from omnibase_infra.validation import ValidationAggregator

# =============================================================================
# Test Configuration
# =============================================================================

pytestmark = [pytest.mark.unit]


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def handler_identity() -> ModelHandlerIdentifier:
    """Create a test handler identifier."""
    return ModelHandlerIdentifier.from_handler_id("test-handler")


@pytest.fixture
def contract_error(
    handler_identity: ModelHandlerIdentifier,
) -> ModelHandlerValidationError:
    """Create a blocking contract validation error."""
    return ModelHandlerValidationError.from_contract_error(
        rule_id="CONTRACT-001",
        message="Invalid YAML syntax in contract.yaml",
        file_path="nodes/registration/contract.yaml",
        line_number=5,
        remediation_hint="Check YAML indentation and syntax",
        handler_identity=handler_identity,
        severity="error",
    )


@pytest.fixture
def security_warning(
    handler_identity: ModelHandlerIdentifier,
) -> ModelHandlerValidationError:
    """Create a non-blocking security validation warning."""
    return ModelHandlerValidationError.from_security_violation(
        rule_id="SECURITY-002",
        message="Handler exposes potentially sensitive method names",
        remediation_hint="Prefix internal methods with underscore",
        handler_identity=handler_identity,
        file_path="nodes/auth/handlers/handler_authenticate.py",
        line_number=42,
        severity="warning",
    )


@pytest.fixture
def architecture_error(
    handler_identity: ModelHandlerIdentifier,
) -> ModelHandlerValidationError:
    """Create a blocking architecture validation error."""
    return ModelHandlerValidationError.from_architecture_error(
        rule_id="ARCH-001",
        message="COMPUTE handler performs I/O operation",
        remediation_hint="Move I/O logic to EFFECT handler",
        handler_identity=handler_identity,
        file_path="nodes/compute/node.py",
        line_number=85,
        severity="error",
    )


@pytest.fixture
def descriptor_warning(
    handler_identity: ModelHandlerIdentifier,
) -> ModelHandlerValidationError:
    """Create a non-blocking descriptor validation warning."""
    return ModelHandlerValidationError.from_descriptor_error(
        rule_id="DESCRIPTOR-001",
        message="Handler method signature could be improved",
        remediation_hint="Consider using type hints for better IDE support",
        handler_identity=handler_identity,
        severity="warning",
    )


# =============================================================================
# Test: Initialization
# =============================================================================


def test_initialization() -> None:
    """Test ValidationAggregator initialization."""
    aggregator = ValidationAggregator()

    assert not aggregator.has_errors
    assert not aggregator.has_blocking_errors
    assert aggregator.error_count == 0
    assert aggregator.blocking_error_count == 0


# =============================================================================
# Test: Adding Errors
# =============================================================================


def test_add_single_error(contract_error: ModelHandlerValidationError) -> None:
    """Test adding a single error to the aggregator."""
    aggregator = ValidationAggregator()
    aggregator.add_error(contract_error)

    assert aggregator.has_errors
    assert aggregator.has_blocking_errors
    assert aggregator.error_count == 1
    assert aggregator.blocking_error_count == 1


def test_add_multiple_errors_individually(
    contract_error: ModelHandlerValidationError,
    security_warning: ModelHandlerValidationError,
) -> None:
    """Test adding multiple errors individually."""
    aggregator = ValidationAggregator()
    aggregator.add_error(contract_error)
    aggregator.add_error(security_warning)

    assert aggregator.has_errors
    assert aggregator.has_blocking_errors
    assert aggregator.error_count == 2
    assert aggregator.blocking_error_count == 1


def test_add_multiple_errors_batch(
    contract_error: ModelHandlerValidationError,
    security_warning: ModelHandlerValidationError,
    architecture_error: ModelHandlerValidationError,
) -> None:
    """Test adding multiple errors in a batch."""
    aggregator = ValidationAggregator()
    errors = [contract_error, security_warning, architecture_error]
    aggregator.add_errors(errors)

    assert aggregator.has_errors
    assert aggregator.has_blocking_errors
    assert aggregator.error_count == 3
    assert aggregator.blocking_error_count == 2


def test_add_empty_sequence() -> None:
    """Test adding an empty sequence of errors."""
    aggregator = ValidationAggregator()
    aggregator.add_errors([])

    assert not aggregator.has_errors
    assert aggregator.error_count == 0


# =============================================================================
# Test: Error Classification
# =============================================================================


def test_has_blocking_errors_with_only_warnings(
    security_warning: ModelHandlerValidationError,
    descriptor_warning: ModelHandlerValidationError,
) -> None:
    """Test has_blocking_errors returns False when only warnings exist."""
    aggregator = ValidationAggregator()
    aggregator.add_errors([security_warning, descriptor_warning])

    assert aggregator.has_errors
    assert not aggregator.has_blocking_errors
    assert aggregator.error_count == 2
    assert aggregator.blocking_error_count == 0


def test_has_blocking_errors_with_mixed_severity(
    contract_error: ModelHandlerValidationError,
    security_warning: ModelHandlerValidationError,
) -> None:
    """Test has_blocking_errors returns True when any blocking error exists."""
    aggregator = ValidationAggregator()
    aggregator.add_errors([contract_error, security_warning])

    assert aggregator.has_errors
    assert aggregator.has_blocking_errors
    assert aggregator.error_count == 2
    assert aggregator.blocking_error_count == 1


# =============================================================================
# Test: Error Grouping
# =============================================================================


def test_get_errors_by_type(
    contract_error: ModelHandlerValidationError,
    security_warning: ModelHandlerValidationError,
    architecture_error: ModelHandlerValidationError,
) -> None:
    """Test grouping errors by error type."""
    aggregator = ValidationAggregator()
    aggregator.add_errors([contract_error, security_warning, architecture_error])

    by_type = aggregator.get_errors_by_type()

    assert len(by_type) == 3
    assert EnumHandlerErrorType.CONTRACT_PARSE_ERROR in by_type
    assert EnumHandlerErrorType.SECURITY_VALIDATION_ERROR in by_type
    assert EnumHandlerErrorType.ARCHITECTURE_VALIDATION_ERROR in by_type
    assert len(by_type[EnumHandlerErrorType.CONTRACT_PARSE_ERROR]) == 1
    assert len(by_type[EnumHandlerErrorType.SECURITY_VALIDATION_ERROR]) == 1
    assert len(by_type[EnumHandlerErrorType.ARCHITECTURE_VALIDATION_ERROR]) == 1


def test_get_errors_by_type_multiple_same_type(
    handler_identity: ModelHandlerIdentifier,
) -> None:
    """Test grouping multiple errors of the same type."""
    aggregator = ValidationAggregator()

    error1 = ModelHandlerValidationError.from_contract_error(
        rule_id="CONTRACT-001",
        message="Missing required field",
        file_path="contract1.yaml",
        remediation_hint="Fix error 1",
        handler_identity=handler_identity,
    )
    error2 = ModelHandlerValidationError.from_contract_error(
        rule_id="CONTRACT-002",
        message="Invalid field value",
        file_path="contract2.yaml",
        remediation_hint="Fix error 2",
        handler_identity=handler_identity,
    )

    aggregator.add_errors([error1, error2])

    by_type = aggregator.get_errors_by_type()

    assert len(by_type) == 1
    # from_contract_error chooses CONTRACT_VALIDATION_ERROR when message doesn't contain "parse" or "yaml"
    assert EnumHandlerErrorType.CONTRACT_VALIDATION_ERROR in by_type
    assert len(by_type[EnumHandlerErrorType.CONTRACT_VALIDATION_ERROR]) == 2


def test_get_errors_by_source(
    contract_error: ModelHandlerValidationError,
    security_warning: ModelHandlerValidationError,
    architecture_error: ModelHandlerValidationError,
) -> None:
    """Test grouping errors by source type."""
    aggregator = ValidationAggregator()
    aggregator.add_errors([contract_error, security_warning, architecture_error])

    by_source = aggregator.get_errors_by_source()

    assert len(by_source) == 2
    assert EnumHandlerSourceType.CONTRACT in by_source
    assert EnumHandlerSourceType.STATIC_ANALYSIS in by_source
    assert len(by_source[EnumHandlerSourceType.CONTRACT]) == 1
    assert len(by_source[EnumHandlerSourceType.STATIC_ANALYSIS]) == 2


def test_get_errors_by_source_empty() -> None:
    """Test grouping errors by source with no errors."""
    aggregator = ValidationAggregator()

    by_source = aggregator.get_errors_by_source()

    assert len(by_source) == 0


# =============================================================================
# Test: Console Formatting
# =============================================================================


def test_format_for_console_no_errors() -> None:
    """Test console formatting with no errors."""
    aggregator = ValidationAggregator()

    output = aggregator.format_for_console()

    assert output == "No validation errors found"


def test_format_for_console_single_error(
    contract_error: ModelHandlerValidationError,
) -> None:
    """Test console formatting with a single error."""
    aggregator = ValidationAggregator()
    aggregator.add_error(contract_error)

    output = aggregator.format_for_console()

    assert "HANDLER VALIDATION ERRORS" in output
    assert "(1 total: 1 blocking, 0 warnings)" in output
    assert "[CONTRACT]" in output
    assert "[CONTRACT-001]" in output
    assert "Invalid YAML syntax" in output
    assert "Location: nodes/registration/contract.yaml:5" in output
    assert "Remediation: Check YAML indentation and syntax" in output
    assert "BLOCKED:" in output


def test_format_for_console_multiple_errors(
    contract_error: ModelHandlerValidationError,
    security_warning: ModelHandlerValidationError,
    architecture_error: ModelHandlerValidationError,
) -> None:
    """Test console formatting with multiple errors."""
    aggregator = ValidationAggregator()
    aggregator.add_errors([contract_error, security_warning, architecture_error])

    output = aggregator.format_for_console()

    assert "HANDLER VALIDATION ERRORS" in output
    assert "(3 total: 2 blocking, 1 warnings)" in output
    assert "[CONTRACT]" in output
    assert "[STATIC_ANALYSIS]" in output
    assert "[CONTRACT-001]" in output
    assert "[SECURITY-002]" in output
    assert "[ARCH-001]" in output
    assert "BLOCKED:" in output


def test_format_for_console_only_warnings(
    security_warning: ModelHandlerValidationError,
    descriptor_warning: ModelHandlerValidationError,
) -> None:
    """Test console formatting with only non-blocking warnings."""
    aggregator = ValidationAggregator()
    aggregator.add_errors([security_warning, descriptor_warning])

    output = aggregator.format_for_console()

    assert "HANDLER VALIDATION ERRORS" in output
    assert "(2 total: 0 blocking, 2 warnings)" in output
    assert "WARNING" in output
    assert "PROCEEDING WITH WARNINGS:" in output
    assert "BLOCKED:" not in output


def test_format_for_console_grouping_by_source(
    contract_error: ModelHandlerValidationError,
    security_warning: ModelHandlerValidationError,
) -> None:
    """Test that console output groups errors by source type."""
    aggregator = ValidationAggregator()
    aggregator.add_errors([contract_error, security_warning])

    output = aggregator.format_for_console()

    # Find positions of source type headers
    contract_pos = output.find("[CONTRACT]")
    static_analysis_pos = output.find("[STATIC_ANALYSIS]")

    # CONTRACT should appear before STATIC_ANALYSIS (alphabetical)
    assert contract_pos > 0
    assert static_analysis_pos > 0
    assert contract_pos < static_analysis_pos


# =============================================================================
# Test: CI Formatting
# =============================================================================


def test_format_for_ci_no_errors() -> None:
    """Test CI formatting with no errors."""
    aggregator = ValidationAggregator()

    output = aggregator.format_for_ci()

    assert output == ""


def test_format_for_ci_single_error(
    contract_error: ModelHandlerValidationError,
) -> None:
    """Test CI formatting with a single error."""
    aggregator = ValidationAggregator()
    aggregator.add_error(contract_error)

    output = aggregator.format_for_ci()

    assert "::error" in output
    assert "file=nodes/registration/contract.yaml" in output
    assert "line=5" in output
    assert "[CONTRACT-001]" in output
    assert "Invalid YAML syntax" in output
    assert "Remediation:" in output


def test_format_for_ci_multiple_errors(
    contract_error: ModelHandlerValidationError,
    security_warning: ModelHandlerValidationError,
) -> None:
    """Test CI formatting with multiple errors."""
    aggregator = ValidationAggregator()
    aggregator.add_errors([contract_error, security_warning])

    output = aggregator.format_for_ci()
    lines = output.split("\n")

    assert len(lines) == 2
    assert "::error" in lines[0]
    assert "::warning" in lines[1]


def test_format_for_ci_warning_annotation(
    security_warning: ModelHandlerValidationError,
) -> None:
    """Test CI formatting uses warning annotation for non-blocking errors."""
    aggregator = ValidationAggregator()
    aggregator.add_error(security_warning)

    output = aggregator.format_for_ci()

    assert "::warning" in output
    assert "::error" not in output


# =============================================================================
# Test: Summary Formatting
# =============================================================================


def test_format_summary_no_errors() -> None:
    """Test summary formatting with no errors."""
    aggregator = ValidationAggregator()

    summary = aggregator.format_summary()

    assert summary == "Handler Validation: PASSED (0 errors)"


def test_format_summary_single_error(
    contract_error: ModelHandlerValidationError,
) -> None:
    """Test summary formatting with a single error."""
    aggregator = ValidationAggregator()
    aggregator.add_error(contract_error)

    summary = aggregator.format_summary()

    assert summary == "Handler Validation: FAILED (1 total: 1 blocking, 0 warnings)"


def test_format_summary_multiple_errors(
    contract_error: ModelHandlerValidationError,
    security_warning: ModelHandlerValidationError,
    architecture_error: ModelHandlerValidationError,
) -> None:
    """Test summary formatting with multiple errors."""
    aggregator = ValidationAggregator()
    aggregator.add_errors([contract_error, security_warning, architecture_error])

    summary = aggregator.format_summary()

    assert summary == "Handler Validation: FAILED (3 total: 2 blocking, 1 warning)"


def test_format_summary_only_warnings(
    security_warning: ModelHandlerValidationError,
    descriptor_warning: ModelHandlerValidationError,
) -> None:
    """Test summary formatting with only warnings."""
    aggregator = ValidationAggregator()
    aggregator.add_errors([security_warning, descriptor_warning])

    summary = aggregator.format_summary()

    assert (
        summary
        == "Handler Validation: PASSED WITH WARNINGS (2 total: 0 blocking, 2 warnings)"
    )


# =============================================================================
# Test: Exception Raising
# =============================================================================


def test_raise_if_blocking_with_blocking_errors(
    contract_error: ModelHandlerValidationError,
) -> None:
    """Test raise_if_blocking raises exception when blocking errors exist."""
    aggregator = ValidationAggregator()
    aggregator.add_error(contract_error)

    with pytest.raises(ProtocolConfigurationError) as exc_info:
        aggregator.raise_if_blocking()

    assert "Handler validation failed" in str(exc_info.value)
    assert "1 blocking errors" in str(exc_info.value)


def test_raise_if_blocking_with_only_warnings(
    security_warning: ModelHandlerValidationError,
) -> None:
    """Test raise_if_blocking does not raise when only warnings exist."""
    aggregator = ValidationAggregator()
    aggregator.add_error(security_warning)

    # Should not raise
    aggregator.raise_if_blocking()


def test_raise_if_blocking_with_no_errors() -> None:
    """Test raise_if_blocking does not raise when no errors exist."""
    aggregator = ValidationAggregator()

    # Should not raise
    aggregator.raise_if_blocking()


def test_raise_if_blocking_error_message(
    contract_error: ModelHandlerValidationError,
    architecture_error: ModelHandlerValidationError,
) -> None:
    """Test raise_if_blocking includes formatted error output in exception."""
    aggregator = ValidationAggregator()
    aggregator.add_errors([contract_error, architecture_error])

    with pytest.raises(ProtocolConfigurationError) as exc_info:
        aggregator.raise_if_blocking()

    error_message = str(exc_info.value)
    assert "2 blocking errors" in error_message
    assert "CONTRACT-001" in error_message
    assert "ARCH-001" in error_message


# =============================================================================
# Test: Clearing Errors
# =============================================================================


def test_clear_empty_aggregator() -> None:
    """Test clearing an empty aggregator."""
    aggregator = ValidationAggregator()
    aggregator.clear()

    assert not aggregator.has_errors
    assert aggregator.error_count == 0


def test_clear_aggregator_with_errors(
    contract_error: ModelHandlerValidationError,
    security_warning: ModelHandlerValidationError,
) -> None:
    """Test clearing aggregator removes all errors."""
    aggregator = ValidationAggregator()
    aggregator.add_errors([contract_error, security_warning])

    assert aggregator.has_errors
    assert aggregator.error_count == 2

    aggregator.clear()

    assert not aggregator.has_errors
    assert not aggregator.has_blocking_errors
    assert aggregator.error_count == 0
    assert aggregator.blocking_error_count == 0


def test_clear_and_reuse_aggregator(
    contract_error: ModelHandlerValidationError,
) -> None:
    """Test that aggregator can be reused after clearing."""
    aggregator = ValidationAggregator()
    aggregator.add_error(contract_error)

    assert aggregator.error_count == 1

    aggregator.clear()

    assert aggregator.error_count == 0

    # Reuse aggregator
    aggregator.add_error(contract_error)

    assert aggregator.error_count == 1


# =============================================================================
# Test: Edge Cases
# =============================================================================


def test_error_without_file_path(handler_identity: ModelHandlerIdentifier) -> None:
    """Test formatting error without file path."""
    error = ModelHandlerValidationError(
        error_type=EnumHandlerErrorType.REGISTRATION_ERROR,
        rule_id="REG-001",
        handler_identity=handler_identity,
        source_type=EnumHandlerSourceType.REGISTRATION,
        message="Registration failed",
        remediation_hint="Check handler configuration",
        severity="error",
    )

    aggregator = ValidationAggregator()
    aggregator.add_error(error)

    output = aggregator.format_for_console()

    assert "Location:" not in output
    assert "[REG-001]" in output


def test_error_without_line_number(handler_identity: ModelHandlerIdentifier) -> None:
    """Test formatting error without line number."""
    error = ModelHandlerValidationError.from_contract_error(
        rule_id="CONTRACT-001",
        message="Missing required field",
        file_path="contract.yaml",
        remediation_hint="Add required field",
        handler_identity=handler_identity,
        line_number=None,
    )

    aggregator = ValidationAggregator()
    aggregator.add_error(error)

    output = aggregator.format_for_console()

    assert "Location: contract.yaml" in output
    assert "contract.yaml:" not in output  # No colon after file path


def test_multiple_errors_same_rule_id(handler_identity: ModelHandlerIdentifier) -> None:
    """Test handling multiple errors with the same rule ID."""
    error1 = ModelHandlerValidationError.from_contract_error(
        rule_id="CONTRACT-001",
        message="Error in file 1",
        file_path="contract1.yaml",
        remediation_hint="Fix file 1",
        handler_identity=handler_identity,
    )
    error2 = ModelHandlerValidationError.from_contract_error(
        rule_id="CONTRACT-001",
        message="Error in file 2",
        file_path="contract2.yaml",
        remediation_hint="Fix file 2",
        handler_identity=handler_identity,
    )

    aggregator = ValidationAggregator()
    aggregator.add_errors([error1, error2])

    assert aggregator.error_count == 2

    output = aggregator.format_for_console()
    assert output.count("[CONTRACT-001]") == 2


# =============================================================================
# Test: Integration Scenarios
# =============================================================================


def test_typical_startup_validation_flow(
    contract_error: ModelHandlerValidationError,
    security_warning: ModelHandlerValidationError,
    architecture_error: ModelHandlerValidationError,
) -> None:
    """Test a typical startup validation flow."""
    aggregator = ValidationAggregator()

    # Simulate contract validation
    aggregator.add_error(contract_error)

    # Simulate security validation
    aggregator.add_error(security_warning)

    # Simulate architecture validation
    aggregator.add_error(architecture_error)

    # Check status
    assert aggregator.has_errors
    assert aggregator.has_blocking_errors
    assert aggregator.error_count == 3
    assert aggregator.blocking_error_count == 2

    # Format for logging
    console_output = aggregator.format_for_console()
    assert "HANDLER VALIDATION ERRORS" in console_output
    assert "(3 total: 2 blocking, 1 warnings)" in console_output

    # Format for CI
    ci_output = aggregator.format_for_ci()
    assert len(ci_output.split("\n")) == 3

    # Should raise due to blocking errors
    with pytest.raises(ProtocolConfigurationError):
        aggregator.raise_if_blocking()


def test_startup_validation_with_only_warnings(
    security_warning: ModelHandlerValidationError,
    descriptor_warning: ModelHandlerValidationError,
) -> None:
    """Test startup validation flow with only non-blocking warnings."""
    aggregator = ValidationAggregator()

    aggregator.add_errors([security_warning, descriptor_warning])

    # Check status
    assert aggregator.has_errors
    assert not aggregator.has_blocking_errors
    assert aggregator.error_count == 2
    assert aggregator.blocking_error_count == 0

    # Format for logging
    console_output = aggregator.format_for_console()
    assert "PROCEEDING WITH WARNINGS:" in console_output
    assert "BLOCKED:" not in console_output

    # Should not raise
    aggregator.raise_if_blocking()
