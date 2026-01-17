# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for ChainPropagationValidator.

This module provides unit tests for the chain propagation validator,
testing individual validation methods and edge cases.

Tests cover:
    - Validator instantiation
    - Correlation propagation validation
    - Causation chain validation
    - Combined chain validation
    - Workflow chain validation
    - Edge cases (empty chains, single message, etc.)

Related:
    - OMN-951: Enforce Correlation and Causation Chain Validation
"""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest

from omnibase_core.models.core.model_envelope_metadata import ModelEnvelopeMetadata
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.enums import EnumChainViolationType, EnumValidationSeverity
from omnibase_infra.models.validation.model_chain_violation import ModelChainViolation
from omnibase_infra.validation.validator_chain_propagation import (
    ChainPropagationValidator,
)

# =============================================================================
# Test Configuration
# =============================================================================

pytestmark = [pytest.mark.unit]


# =============================================================================
# Payload Classes for Testing
# =============================================================================


class PayloadTypeA:
    """Simple payload class A for testing."""

    def __init__(self, value: str) -> None:
        self.value = value


class PayloadTypeB:
    """Simple payload class B for testing."""

    def __init__(self, value: str) -> None:
        self.value = value


# =============================================================================
# Helper Functions
# =============================================================================


def create_envelope(
    correlation_id: UUID | None = None,
    causation_id: UUID | None = None,
    payload_class: type = PayloadTypeA,
) -> ModelEventEnvelope[PayloadTypeA | PayloadTypeB]:
    """Create a test envelope with optional correlation and causation IDs.

    Args:
        correlation_id: Optional correlation ID for the envelope.
        causation_id: Optional causation ID to store in metadata tags.
        payload_class: The payload class to use.

    Returns:
        A configured ModelEventEnvelope instance.
    """
    metadata = ModelEnvelopeMetadata()
    if causation_id is not None:
        metadata = ModelEnvelopeMetadata(
            tags={"causation_id": str(causation_id)},
        )

    return ModelEventEnvelope(
        payload=payload_class(value="test"),
        correlation_id=correlation_id,
        metadata=metadata,
    )


# =============================================================================
# Validator Creation Tests
# =============================================================================


class TestValidatorCreation:
    """Tests for ChainPropagationValidator instantiation."""

    def test_validator_creation(self) -> None:
        """ChainPropagationValidator should instantiate without errors."""
        validator = ChainPropagationValidator()

        assert validator is not None
        assert isinstance(validator, ChainPropagationValidator)

    def test_validator_is_stateless(self) -> None:
        """ChainPropagationValidator should be stateless (no instance state)."""
        validator = ChainPropagationValidator()

        # Validator should not have any mutable state
        # It only has methods, no instance attributes that change
        assert not hasattr(validator, "_state")
        assert not hasattr(validator, "_cache")


# =============================================================================
# Correlation Propagation Tests
# =============================================================================


class TestValidateCorrelation:
    """Tests for validate_correlation_propagation method."""

    def test_validate_correlation_success_matching_ids(self) -> None:
        """Matching correlation_ids should return no violations."""
        validator = ChainPropagationValidator()
        correlation_id = uuid4()

        parent = create_envelope(correlation_id=correlation_id)
        child = create_envelope(
            correlation_id=correlation_id,
            causation_id=parent.envelope_id,
        )

        violations = validator.validate_correlation_propagation(parent, child)

        assert len(violations) == 0

    def test_validate_correlation_success_no_parent_correlation(self) -> None:
        """When parent has no correlation_id, validation passes."""
        validator = ChainPropagationValidator()

        parent = create_envelope(correlation_id=None)
        child = create_envelope(
            correlation_id=uuid4(),  # Child can have any correlation_id
            causation_id=parent.envelope_id,
        )

        violations = validator.validate_correlation_propagation(parent, child)

        assert len(violations) == 0

    def test_validate_correlation_failure_mismatch(self) -> None:
        """Different correlation_ids should return a violation."""
        validator = ChainPropagationValidator()

        parent = create_envelope(correlation_id=uuid4())
        child = create_envelope(
            correlation_id=uuid4(),  # Different!
            causation_id=parent.envelope_id,
        )

        violations = validator.validate_correlation_propagation(parent, child)

        assert len(violations) == 1
        assert (
            violations[0].violation_type == EnumChainViolationType.CORRELATION_MISMATCH
        )
        assert violations[0].severity == "error"

    def test_validate_correlation_failure_child_missing(self) -> None:
        """Missing correlation_id in child when parent has one is a violation."""
        validator = ChainPropagationValidator()

        parent = create_envelope(correlation_id=uuid4())
        child = create_envelope(
            correlation_id=None,  # Missing!
            causation_id=parent.envelope_id,
        )

        violations = validator.validate_correlation_propagation(parent, child)

        assert len(violations) == 1
        assert (
            violations[0].violation_type == EnumChainViolationType.CORRELATION_MISMATCH
        )
        assert violations[0].actual_value is None


# =============================================================================
# Causation Chain Tests
# =============================================================================


class TestValidateCausation:
    """Tests for validate_causation_chain method."""

    def test_validate_causation_success_correct_reference(self) -> None:
        """Child's causation_id matching parent's envelope_id should pass."""
        validator = ChainPropagationValidator()
        correlation_id = uuid4()

        parent = create_envelope(correlation_id=correlation_id)
        child = create_envelope(
            correlation_id=correlation_id,
            causation_id=parent.envelope_id,  # Correct reference
        )

        violations = validator.validate_causation_chain(parent, child)

        assert len(violations) == 0

    def test_validate_causation_failure_wrong_reference(self) -> None:
        """Wrong causation_id should return a violation."""
        validator = ChainPropagationValidator()
        correlation_id = uuid4()

        parent = create_envelope(correlation_id=correlation_id)
        child = create_envelope(
            correlation_id=correlation_id,
            causation_id=uuid4(),  # Wrong reference!
        )

        violations = validator.validate_causation_chain(parent, child)

        assert len(violations) == 1
        assert (
            violations[0].violation_type
            == EnumChainViolationType.CAUSATION_CHAIN_BROKEN
        )
        assert violations[0].expected_value == parent.envelope_id
        assert violations[0].severity == "error"

    def test_validate_causation_failure_missing_causation(self) -> None:
        """Missing causation_id in child should return a violation."""
        validator = ChainPropagationValidator()
        correlation_id = uuid4()

        parent = create_envelope(correlation_id=correlation_id)
        child = create_envelope(
            correlation_id=correlation_id,
            causation_id=None,  # Missing!
        )

        violations = validator.validate_causation_chain(parent, child)

        assert len(violations) == 1
        assert (
            violations[0].violation_type
            == EnumChainViolationType.CAUSATION_CHAIN_BROKEN
        )
        assert violations[0].actual_value is None


# =============================================================================
# Combined Validation Tests
# =============================================================================


class TestValidateChain:
    """Tests for validate_chain method (combined validation)."""

    def test_validate_chain_combines_both(self) -> None:
        """validate_chain should run both correlation and causation validation."""
        validator = ChainPropagationValidator()
        correlation_id = uuid4()

        parent = create_envelope(correlation_id=correlation_id)

        # Child with both violations: wrong correlation + missing causation
        bad_child = create_envelope(
            correlation_id=uuid4(),  # Wrong
            causation_id=None,  # Missing
        )

        violations = validator.validate_chain(parent, bad_child)

        assert len(violations) == 2

        violation_types = {v.violation_type for v in violations}
        assert EnumChainViolationType.CORRELATION_MISMATCH in violation_types
        assert EnumChainViolationType.CAUSATION_CHAIN_BROKEN in violation_types

    def test_validate_chain_empty_for_valid(self) -> None:
        """validate_chain should return empty list for valid chain."""
        validator = ChainPropagationValidator()
        correlation_id = uuid4()

        parent = create_envelope(correlation_id=correlation_id)
        valid_child = create_envelope(
            correlation_id=correlation_id,
            causation_id=parent.envelope_id,
        )

        violations = validator.validate_chain(parent, valid_child)

        assert len(violations) == 0


# =============================================================================
# Workflow Chain Tests
# =============================================================================


class TestValidateWorkflowChain:
    """Tests for validate_workflow_chain method."""

    def test_validate_workflow_chain_empty_list(self) -> None:
        """Empty envelope list should return no violations."""
        validator = ChainPropagationValidator()

        violations = validator.validate_workflow_chain([])

        assert len(violations) == 0

    def test_validate_workflow_chain_single_message(self) -> None:
        """Single message (no chain) should return no violations."""
        validator = ChainPropagationValidator()

        single = create_envelope(correlation_id=uuid4())

        violations = validator.validate_workflow_chain([single])

        assert len(violations) == 0

    def test_validate_workflow_chain_all_valid(self) -> None:
        """Valid workflow chain should return no violations."""
        validator = ChainPropagationValidator()
        correlation_id = uuid4()

        # Create valid chain: msg1 -> msg2 -> msg3
        msg1 = create_envelope(correlation_id=correlation_id)
        msg2 = create_envelope(
            correlation_id=correlation_id,
            causation_id=msg1.envelope_id,
        )
        msg3 = create_envelope(
            correlation_id=correlation_id,
            causation_id=msg2.envelope_id,
        )

        violations = validator.validate_workflow_chain([msg1, msg2, msg3])

        assert len(violations) == 0

    def test_validate_workflow_chain_detects_issues(self) -> None:
        """Workflow chain with issues should detect violations."""
        validator = ChainPropagationValidator()
        correlation_id = uuid4()

        # Create chain with correlation drift in msg3
        msg1 = create_envelope(correlation_id=correlation_id)
        msg2 = create_envelope(
            correlation_id=correlation_id,
            causation_id=msg1.envelope_id,
        )
        msg3 = create_envelope(
            correlation_id=uuid4(),  # Different correlation!
            causation_id=msg2.envelope_id,
        )

        violations = validator.validate_workflow_chain([msg1, msg2, msg3])

        assert len(violations) >= 1

        # Should detect correlation mismatch
        correlation_violations = [
            v
            for v in violations
            if v.violation_type == EnumChainViolationType.CORRELATION_MISMATCH
        ]
        assert len(correlation_violations) >= 1

    def test_validate_workflow_chain_detects_missing_causation(self) -> None:
        """Workflow chain with missing causation_id should detect violation."""
        validator = ChainPropagationValidator()
        correlation_id = uuid4()

        # Create chain where msg2 is missing causation_id
        msg1 = create_envelope(correlation_id=correlation_id)
        msg2 = create_envelope(
            correlation_id=correlation_id,
            causation_id=None,  # Missing causation!
        )

        violations = validator.validate_workflow_chain([msg1, msg2])

        assert len(violations) >= 1

        # Should detect causation chain broken
        causation_violations = [
            v
            for v in violations
            if v.violation_type == EnumChainViolationType.CAUSATION_CHAIN_BROKEN
        ]
        assert len(causation_violations) >= 1

    def test_validate_workflow_chain_detects_external_ancestor(self) -> None:
        """Workflow should detect causation_id referencing external message."""
        validator = ChainPropagationValidator()
        correlation_id = uuid4()

        # Create chain where msg2 references an external message
        msg1 = create_envelope(correlation_id=correlation_id)
        msg2 = create_envelope(
            correlation_id=correlation_id,
            causation_id=uuid4(),  # References message not in chain!
        )

        violations = validator.validate_workflow_chain([msg1, msg2])

        assert len(violations) >= 1

        # Should detect ancestor skipped (external reference)
        ancestor_violations = [
            v
            for v in violations
            if v.violation_type == EnumChainViolationType.CAUSATION_ANCESTOR_SKIPPED
        ]
        assert len(ancestor_violations) >= 1


# =============================================================================
# Violation Model Tests
# =============================================================================


class TestViolationModel:
    """Tests for ModelChainViolation behavior."""

    def test_violation_is_blocking_error(self) -> None:
        """Violation with severity='error' should be blocking."""
        violation = ModelChainViolation(
            violation_type=EnumChainViolationType.CORRELATION_MISMATCH,
            expected_value=uuid4(),
            actual_value=uuid4(),
            message_id=uuid4(),
            violation_message="Test violation",
            severity=EnumValidationSeverity.ERROR,
        )

        assert violation.is_blocking() is True

    def test_violation_is_not_blocking_warning(self) -> None:
        """Violation with severity='warning' should not be blocking."""
        violation = ModelChainViolation(
            violation_type=EnumChainViolationType.CAUSATION_CHAIN_BROKEN,
            expected_value=uuid4(),
            actual_value=uuid4(),
            message_id=uuid4(),
            violation_message="Test warning",
            severity=EnumValidationSeverity.WARNING,
        )

        assert violation.is_blocking() is False

    def test_violation_format_for_logging(self) -> None:
        """Violation should format correctly for logging."""
        msg_id = uuid4()
        expected = uuid4()
        actual = uuid4()

        violation = ModelChainViolation(
            violation_type=EnumChainViolationType.CORRELATION_MISMATCH,
            expected_value=expected,
            actual_value=actual,
            message_id=msg_id,
            violation_message="Correlation mismatch detected",
            severity=EnumValidationSeverity.ERROR,
        )

        log_output = violation.format_for_logging()

        assert "[error]" in log_output
        assert "CORRELATION_MISMATCH" in log_output.upper()
        assert f"message={msg_id}" in log_output

    def test_violation_to_structured_dict(self) -> None:
        """Violation should convert to structured dict for logging systems."""
        msg_id = uuid4()
        expected = uuid4()
        actual = uuid4()

        violation = ModelChainViolation(
            violation_type=EnumChainViolationType.CAUSATION_CHAIN_BROKEN,
            expected_value=expected,
            actual_value=actual,
            message_id=msg_id,
            violation_message="Chain broken",
            severity=EnumValidationSeverity.ERROR,
        )

        structured = violation.to_structured_dict()

        assert structured["violation_type"] == "causation_chain_broken"
        assert structured["severity"] == "error"
        assert structured["message_id"] == str(msg_id)
        assert structured["expected_value"] == str(expected)
        assert structured["actual_value"] == str(actual)


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_both_correlation_ids_none(self) -> None:
        """Both parent and child having None correlation_id should pass."""
        validator = ChainPropagationValidator()

        parent = create_envelope(correlation_id=None)
        child = create_envelope(
            correlation_id=None,
            causation_id=parent.envelope_id,
        )

        violations = validator.validate_correlation_propagation(parent, child)

        assert len(violations) == 0

    def test_causation_id_in_headers(self) -> None:
        """Validator should find causation_id in metadata headers."""
        validator = ChainPropagationValidator()
        correlation_id = uuid4()

        parent = create_envelope(correlation_id=correlation_id)

        # Create child with causation_id in headers instead of tags
        child_metadata = ModelEnvelopeMetadata(
            headers={"x-causation-id": str(parent.envelope_id)},
        )
        child = ModelEventEnvelope(
            payload=PayloadTypeB(value="test"),
            correlation_id=correlation_id,
            metadata=child_metadata,
        )

        violations = validator.validate_causation_chain(parent, child)

        assert len(violations) == 0

    def test_thread_safety_stateless(self) -> None:
        """Validator should be safe to use from multiple threads.

        Since the validator is stateless, multiple concurrent calls
        with different data should not interfere with each other.
        """
        import concurrent.futures

        validator = ChainPropagationValidator()

        def validate_pair(pair_id: int) -> list[ModelChainViolation]:
            """Create and validate a unique pair of envelopes."""
            correlation_id = uuid4()
            parent = create_envelope(correlation_id=correlation_id)
            child = create_envelope(
                correlation_id=correlation_id,
                causation_id=parent.envelope_id,
            )
            return validator.validate_chain(parent, child)

        # Run multiple validations concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(validate_pair, i) for i in range(100)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        # All should pass (empty violation lists)
        for violations in results:
            assert len(violations) == 0


# =============================================================================
# Linear Chain Validation Tests
# =============================================================================


class TestValidateLinearWorkflowChain:
    """Tests for validate_linear_workflow_chain method.

    This method enforces strict linear chain validation where each message
    must reference its immediate predecessor (no ancestor skipping).
    """

    def test_linear_chain_valid(self) -> None:
        """Valid linear chain (each msg references direct parent) passes."""
        validator = ChainPropagationValidator()
        correlation_id = uuid4()

        # Create valid linear chain: msg1 -> msg2 -> msg3
        msg1 = create_envelope(correlation_id=correlation_id)
        msg2 = create_envelope(
            correlation_id=correlation_id,
            causation_id=msg1.envelope_id,
        )
        msg3 = create_envelope(
            correlation_id=correlation_id,
            causation_id=msg2.envelope_id,
        )

        violations = validator.validate_linear_workflow_chain([msg1, msg2, msg3])

        assert len(violations) == 0

    def test_linear_chain_detects_ancestor_skip(self) -> None:
        """Ancestor skip in linear chain should be detected as violation.

        When msg3 references msg1 instead of msg2, this is an ancestor skip
        that passes validate_workflow_chain() but fails linear validation.
        """
        validator = ChainPropagationValidator()
        correlation_id = uuid4()

        # Create chain where msg3 skips msg2 and references msg1
        msg1 = create_envelope(correlation_id=correlation_id)
        msg2 = create_envelope(
            correlation_id=correlation_id,
            causation_id=msg1.envelope_id,
        )
        # msg3 references msg1 (skipping msg2) - this is the violation
        msg3 = create_envelope(
            correlation_id=correlation_id,
            causation_id=msg1.envelope_id,  # Should be msg2.envelope_id
        )

        # Linear validation should detect this as a causation chain break
        violations = validator.validate_linear_workflow_chain([msg1, msg2, msg3])

        assert len(violations) >= 1

        # Should detect causation chain broken (expected msg2, got msg1)
        causation_violations = [
            v
            for v in violations
            if v.violation_type == EnumChainViolationType.CAUSATION_CHAIN_BROKEN
        ]
        assert len(causation_violations) == 1

        violation = causation_violations[0]
        assert violation.expected_value == msg2.envelope_id
        assert violation.actual_value == msg1.envelope_id

    def test_linear_chain_single_message_no_violations(self) -> None:
        """Single message in chain should return no violations."""
        validator = ChainPropagationValidator()

        single = create_envelope(correlation_id=uuid4())

        violations = validator.validate_linear_workflow_chain([single])

        assert len(violations) == 0

    def test_linear_chain_empty_list_no_violations(self) -> None:
        """Empty envelope list should return no violations."""
        validator = ChainPropagationValidator()

        violations = validator.validate_linear_workflow_chain([])

        assert len(violations) == 0

    def test_linear_chain_two_messages_valid(self) -> None:
        """Two-message linear chain should pass if properly linked."""
        validator = ChainPropagationValidator()
        correlation_id = uuid4()

        msg1 = create_envelope(correlation_id=correlation_id)
        msg2 = create_envelope(
            correlation_id=correlation_id,
            causation_id=msg1.envelope_id,
        )

        violations = validator.validate_linear_workflow_chain([msg1, msg2])

        assert len(violations) == 0

    def test_linear_chain_detects_correlation_mismatch(self) -> None:
        """Linear chain should also detect correlation mismatches."""
        validator = ChainPropagationValidator()
        correlation_id = uuid4()

        msg1 = create_envelope(correlation_id=correlation_id)
        # msg2 has different correlation (still references msg1 correctly)
        msg2 = create_envelope(
            correlation_id=uuid4(),  # Different correlation!
            causation_id=msg1.envelope_id,
        )

        violations = validator.validate_linear_workflow_chain([msg1, msg2])

        assert len(violations) >= 1

        correlation_violations = [
            v
            for v in violations
            if v.violation_type == EnumChainViolationType.CORRELATION_MISMATCH
        ]
        assert len(correlation_violations) == 1

    def test_linear_chain_detects_missing_causation(self) -> None:
        """Linear chain should detect missing causation_id."""
        validator = ChainPropagationValidator()
        correlation_id = uuid4()

        msg1 = create_envelope(correlation_id=correlation_id)
        msg2 = create_envelope(
            correlation_id=correlation_id,
            causation_id=None,  # Missing causation!
        )

        violations = validator.validate_linear_workflow_chain([msg1, msg2])

        assert len(violations) >= 1

        causation_violations = [
            v
            for v in violations
            if v.violation_type == EnumChainViolationType.CAUSATION_CHAIN_BROKEN
        ]
        assert len(causation_violations) == 1
        assert causation_violations[0].actual_value is None
