# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for NodeArchitectureValidatorCompute.

This module provides comprehensive tests for the architecture validator compute node,
including validation of rules, violation detection, fail-fast behavior, and custom
__bool__ behavior on result models.

The tests use mock rules implementing ProtocolArchitectureRule to avoid dependencies
on real validators (owned by OMN-1099). This approach enables TDD for the validator
infrastructure while real rules are developed separately.

Test Categories:
    - TestNodeArchitectureValidatorCompute: Core validator behavior tests
    - TestMockRuleProtocolCompliance: Verify mock rule implements protocol correctly
    - TestValidationResultBoolBehavior: Custom __bool__ behavior tests
    - TestViolationSeverityBehavior: Severity-based blocking behavior tests
    - TestValidatorEdgeCases: Edge cases and boundary conditions

Related:
    - OMN-1138: Architecture Validator for omnibase_infra
    - OMN-1099: Validators implementing ProtocolArchitectureRule (future)

.. versionadded:: 0.8.0
    Created as part of OMN-1138 Architecture Validator implementation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from omnibase_infra.errors import RuntimeHostError
from omnibase_infra.nodes.architecture_validator import (
    SUPPORTED_RULE_IDS,
    EnumValidationSeverity,
    ModelArchitectureValidationRequest,
    ModelArchitectureValidationResult,
    ModelArchitectureViolation,
    ModelRuleCheckResult,
    NodeArchitectureValidatorCompute,
    ProtocolArchitectureRule,
)

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer


# =============================================================================
# Mock Rule Implementation
# =============================================================================


class MockRule:
    """Mock implementation of ProtocolArchitectureRule for testing.

    Provides configurable behavior to simulate various rule outcomes without
    requiring real validator implementations from OMN-1099.

    This mock enables testing of:
    - Rule filtering by rule_id
    - Passing and failing checks
    - Severity levels (ERROR, WARNING, INFO)
    - Custom violation messages and details

    Example:
        >>> rule = MockRule(
        ...     rule_id="TEST_RULE",
        ...     name="Test Rule",
        ...     severity=EnumValidationSeverity.ERROR,
        ...     should_pass=False,
        ...     message="Test violation message",
        ... )
        >>> result = rule.check(some_target)
        >>> result.passed
        False
        >>> result.message
        'Test violation message'

    .. versionadded:: 0.8.0
    """

    def __init__(
        self,
        rule_id: str,
        name: str,
        severity: EnumValidationSeverity,
        should_pass: bool = True,
        message: str | None = None,
        details: dict[str, object] | None = None,
    ) -> None:
        """Initialize mock rule with configurable behavior.

        Args:
            rule_id: Unique identifier for this rule.
            name: Human-readable name for display.
            severity: Severity level if rule is violated.
            should_pass: If True, check() returns passed=True; else False.
            message: Custom message for check results.
            details: Additional details to include in check results.
        """
        self._rule_id = rule_id
        self._name = name
        self._severity = severity
        self._should_pass = should_pass
        self._message = message
        self._details = details
        self._check_count = 0

    @property
    def rule_id(self) -> str:
        """Return the unique identifier for this rule."""
        return self._rule_id

    @property
    def name(self) -> str:
        """Return the human-readable name for this rule."""
        return self._name

    @property
    def description(self) -> str:
        """Return a detailed description of what this rule checks."""
        return f"Mock rule: {self._name} (for testing)"

    @property
    def severity(self) -> EnumValidationSeverity:
        """Return the severity level for violations of this rule."""
        return self._severity

    @property
    def check_count(self) -> int:
        """Return the number of times check() has been called."""
        return self._check_count

    def check(self, target: object) -> ModelRuleCheckResult:
        """Check the target against this rule.

        Args:
            target: The node, handler, or other object to validate.

        Returns:
            ModelRuleCheckResult with configurable passed/message/details.
        """
        self._check_count += 1
        return ModelRuleCheckResult(
            passed=self._should_pass,
            rule_id=self._rule_id,
            message=self._message,
            details=self._details,
        )


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_container() -> MagicMock:
    """Create a minimal mock ONEX container for validator tests.

    Returns:
        MagicMock configured with minimal container.config attribute.
    """
    container = MagicMock()
    container.config = MagicMock()
    return container


@pytest.fixture
def passing_rule() -> MockRule:
    """Create a mock rule that always passes.

    Returns:
        MockRule configured to return passed=True on all checks.
        Uses NO_HANDLER_PUBLISHING as a valid supported rule_id.
    """
    return MockRule(
        rule_id="NO_HANDLER_PUBLISHING",
        name="No Handler Publishing Rule",
        severity=EnumValidationSeverity.ERROR,
        should_pass=True,
    )


@pytest.fixture
def failing_rule() -> MockRule:
    """Create a mock rule that always fails with ERROR severity.

    Returns:
        MockRule configured to return passed=False with ERROR severity.
        Uses PURE_REDUCERS as a valid supported rule_id.
    """
    return MockRule(
        rule_id="PURE_REDUCERS",
        name="Pure Reducers Rule",
        severity=EnumValidationSeverity.ERROR,
        should_pass=False,
        message="This rule always fails",
    )


@pytest.fixture
def warning_rule() -> MockRule:
    """Create a mock rule that fails with WARNING severity.

    Returns:
        MockRule configured to return passed=False with WARNING severity.
        Uses NO_FSM_IN_ORCHESTRATORS as a valid supported rule_id.
    """
    return MockRule(
        rule_id="NO_FSM_IN_ORCHESTRATORS",
        name="No FSM in Orchestrators Rule",
        severity=EnumValidationSeverity.WARNING,
        should_pass=False,
        message="This is a warning",
    )


@pytest.fixture
def info_rule() -> MockRule:
    """Create a mock rule that fails with INFO severity.

    Returns:
        MockRule configured to return passed=False with INFO severity.
        Uses NO_WORKFLOW_IN_REDUCERS as a valid supported rule_id.
    """
    return MockRule(
        rule_id="NO_WORKFLOW_IN_REDUCERS",
        name="No Workflow in Reducers Rule",
        severity=EnumValidationSeverity.INFO,
        should_pass=False,
        message="This is informational",
    )


@pytest.fixture
def sample_node() -> object:
    """Create a sample node object for testing.

    Returns:
        Simple object to use as validation target.
    """

    class SampleNode:
        """Sample node class for testing."""

        __name__ = "SampleNode"

    return SampleNode()


@pytest.fixture
def sample_handler() -> object:
    """Create a sample handler object for testing.

    Returns:
        Simple object to use as validation target.
    """

    class SampleHandler:
        """Sample handler class for testing."""

        __name__ = "SampleHandler"

    return SampleHandler()


# =============================================================================
# Tests for NodeArchitectureValidatorCompute
# =============================================================================


@pytest.mark.unit
class TestNodeArchitectureValidatorCompute:
    """Test cases for the architecture validator compute node.

    Tests cover core validation logic including rule execution, violation
    detection, fail-fast behavior, and rule filtering.
    """

    def test_empty_request_returns_valid_result(
        self,
        mock_container: MagicMock,
        passing_rule: MockRule,
    ) -> None:
        """Test that empty request (no nodes/handlers) passes validation.

        An empty request should return a valid result with zero violations
        since there is nothing to validate.
        """
        validator = NodeArchitectureValidatorCompute(
            container=mock_container,
            rules=(passing_rule,),
        )
        request = ModelArchitectureValidationRequest(
            nodes=(),
            handlers=(),
        )

        result = validator.compute(request)

        assert result.valid is True
        assert result.violation_count == 0
        assert result.nodes_checked == 0
        assert result.handlers_checked == 0
        assert passing_rule.check_count == 0  # No targets to check

    def test_no_rules_returns_valid_result(
        self,
        mock_container: MagicMock,
        sample_node: object,
    ) -> None:
        """Test that validation passes when no rules are registered.

        Even with nodes/handlers to check, validation should pass if
        there are no rules to enforce.
        """
        validator = NodeArchitectureValidatorCompute(
            container=mock_container,
            rules=(),  # No rules
        )
        request = ModelArchitectureValidationRequest(
            nodes=(sample_node,),
            handlers=(),
        )

        result = validator.compute(request)

        assert result.valid is True
        assert result.violation_count == 0
        assert result.nodes_checked == 1
        assert result.rules_checked == ()

    def test_all_rules_pass_returns_valid_result(
        self,
        mock_container: MagicMock,
        passing_rule: MockRule,
        sample_node: object,
        sample_handler: object,
    ) -> None:
        """Test that validation passes when all rules pass.

        When all rules return passed=True, the result should be valid
        with no violations.
        """
        another_passing_rule = MockRule(
            rule_id="PURE_REDUCERS",
            name="Pure Reducers Rule",
            severity=EnumValidationSeverity.WARNING,
            should_pass=True,
        )
        validator = NodeArchitectureValidatorCompute(
            container=mock_container,
            rules=(passing_rule, another_passing_rule),
        )
        request = ModelArchitectureValidationRequest(
            nodes=(sample_node,),
            handlers=(sample_handler,),
        )

        result = validator.compute(request)

        assert result.valid is True
        assert result.violation_count == 0
        assert result.nodes_checked == 1
        assert result.handlers_checked == 1
        assert set(result.rules_checked) == {"NO_HANDLER_PUBLISHING", "PURE_REDUCERS"}

    def test_single_violation_detected(
        self,
        mock_container: MagicMock,
        failing_rule: MockRule,
        sample_node: object,
    ) -> None:
        """Test that a single violation is correctly captured.

        When a rule fails, the violation should be captured with all
        relevant details (rule_id, rule_name, severity, target info, message).
        """
        validator = NodeArchitectureValidatorCompute(
            container=mock_container,
            rules=(failing_rule,),
        )
        request = ModelArchitectureValidationRequest(
            nodes=(sample_node,),
            handlers=(),
        )

        result = validator.compute(request)

        assert result.valid is False
        assert result.violation_count == 1
        violation = result.violations[0]
        assert violation.rule_id == "PURE_REDUCERS"
        assert violation.rule_name == "Pure Reducers Rule"
        assert violation.severity == EnumValidationSeverity.ERROR
        assert violation.message == "This rule always fails"

    def test_multiple_violations_aggregated(
        self,
        mock_container: MagicMock,
        sample_node: object,
        sample_handler: object,
    ) -> None:
        """Test that multiple violations are aggregated in result.

        When multiple rules fail or a rule fails on multiple targets,
        all violations should be collected in the result.
        """
        fail_rule_1 = MockRule(
            rule_id="NO_HANDLER_PUBLISHING",
            name="No Handler Publishing Rule",
            severity=EnumValidationSeverity.ERROR,
            should_pass=False,
            message="First failure",
        )
        fail_rule_2 = MockRule(
            rule_id="PURE_REDUCERS",
            name="Pure Reducers Rule",
            severity=EnumValidationSeverity.WARNING,
            should_pass=False,
            message="Second failure",
        )
        validator = NodeArchitectureValidatorCompute(
            container=mock_container,
            rules=(fail_rule_1, fail_rule_2),
        )
        request = ModelArchitectureValidationRequest(
            nodes=(sample_node,),
            handlers=(sample_handler,),
        )

        result = validator.compute(request)

        assert result.valid is False
        # 2 rules x (1 node + 1 handler) = 4 violations
        assert result.violation_count == 4
        rule_ids = [v.rule_id for v in result.violations]
        assert rule_ids.count("NO_HANDLER_PUBLISHING") == 2
        assert rule_ids.count("PURE_REDUCERS") == 2

    def test_fail_fast_stops_on_first_violation(
        self,
        mock_container: MagicMock,
        sample_node: object,
        sample_handler: object,
    ) -> None:
        """Test that fail_fast=True stops after first violation.

        When fail_fast is enabled, the validator should return immediately
        after detecting the first violation, without checking remaining
        rules or targets.
        """
        fail_rule_1 = MockRule(
            rule_id="NO_HANDLER_PUBLISHING",
            name="No Handler Publishing Rule",
            severity=EnumValidationSeverity.ERROR,
            should_pass=False,
            message="First failure",
        )
        fail_rule_2 = MockRule(
            rule_id="PURE_REDUCERS",
            name="Pure Reducers Rule",
            severity=EnumValidationSeverity.ERROR,
            should_pass=False,
            message="Second failure",
        )
        validator = NodeArchitectureValidatorCompute(
            container=mock_container,
            rules=(fail_rule_1, fail_rule_2),
        )
        request = ModelArchitectureValidationRequest(
            nodes=(sample_node,),
            handlers=(sample_handler,),
            fail_fast=True,
        )

        result = validator.compute(request)

        assert result.valid is False
        # Should stop after first violation
        assert result.violation_count == 1
        assert result.violations[0].rule_id == "NO_HANDLER_PUBLISHING"
        # Second rule should not be checked (first rule failed first)
        assert fail_rule_1.check_count == 1
        # fail_rule_2 may or may not be checked depending on order,
        # but total violations should be 1

    def test_rule_id_filter_only_checks_specified_rules(
        self,
        mock_container: MagicMock,
        sample_node: object,
    ) -> None:
        """Test that rule_ids filter limits which rules are checked.

        When rule_ids is specified in the request, only rules with matching
        IDs should be executed.
        """
        rule_a = MockRule(
            rule_id="NO_HANDLER_PUBLISHING",
            name="No Handler Publishing Rule",
            severity=EnumValidationSeverity.ERROR,
            should_pass=False,
            message="Rule A failed",
        )
        rule_b = MockRule(
            rule_id="PURE_REDUCERS",
            name="Pure Reducers Rule",
            severity=EnumValidationSeverity.ERROR,
            should_pass=False,
            message="Rule B failed",
        )
        rule_c = MockRule(
            rule_id="NO_FSM_IN_ORCHESTRATORS",
            name="No FSM in Orchestrators Rule",
            severity=EnumValidationSeverity.ERROR,
            should_pass=False,
            message="Rule C failed",
        )
        validator = NodeArchitectureValidatorCompute(
            container=mock_container,
            rules=(rule_a, rule_b, rule_c),
        )
        request = ModelArchitectureValidationRequest(
            nodes=(sample_node,),
            handlers=(),
            rule_ids=(
                "NO_HANDLER_PUBLISHING",
                "NO_FSM_IN_ORCHESTRATORS",
            ),  # Only check A and C
        )

        result = validator.compute(request)

        assert result.valid is False
        assert result.violation_count == 2
        assert set(result.rules_checked) == {
            "NO_HANDLER_PUBLISHING",
            "NO_FSM_IN_ORCHESTRATORS",
        }
        # Rule B should not be checked
        assert rule_a.check_count == 1
        assert rule_b.check_count == 0
        assert rule_c.check_count == 1

    def test_correlation_id_propagated(
        self,
        mock_container: MagicMock,
        passing_rule: MockRule,
        sample_node: object,
    ) -> None:
        """Test that correlation_id is passed through to result.

        The correlation_id from the request should be preserved in the
        result for distributed tracing purposes.
        """
        correlation_id = str(uuid4())
        validator = NodeArchitectureValidatorCompute(
            container=mock_container,
            rules=(passing_rule,),
        )
        request = ModelArchitectureValidationRequest(
            nodes=(sample_node,),
            handlers=(),
            correlation_id=correlation_id,
        )

        result = validator.compute(request)

        assert result.correlation_id == correlation_id

    def test_nodes_and_handlers_counted_correctly(
        self,
        mock_container: MagicMock,
        passing_rule: MockRule,
    ) -> None:
        """Test that nodes_checked and handlers_checked counts are accurate.

        The result should accurately reflect the number of nodes and handlers
        that were validated.
        """
        # Create multiple sample nodes and handlers
        nodes = tuple(
            type(f"Node{i}", (), {"__name__": f"Node{i}"})() for i in range(3)
        )
        handlers = tuple(
            type(f"Handler{i}", (), {"__name__": f"Handler{i}"})() for i in range(5)
        )

        validator = NodeArchitectureValidatorCompute(
            container=mock_container,
            rules=(passing_rule,),
        )
        request = ModelArchitectureValidationRequest(
            nodes=nodes,
            handlers=handlers,
        )

        result = validator.compute(request)

        assert result.nodes_checked == 3
        assert result.handlers_checked == 5

    def test_rules_checked_list_populated(
        self,
        mock_container: MagicMock,
        sample_node: object,
    ) -> None:
        """Test that rules_checked contains IDs of all checked rules.

        The result should list all rule IDs that were evaluated during
        validation, regardless of whether they passed or failed.
        """
        rule_1 = MockRule(
            rule_id="NO_HANDLER_PUBLISHING",
            name="No Handler Publishing Rule",
            severity=EnumValidationSeverity.ERROR,
            should_pass=True,
        )
        rule_2 = MockRule(
            rule_id="PURE_REDUCERS",
            name="Pure Reducers Rule",
            severity=EnumValidationSeverity.ERROR,
            should_pass=False,
        )
        rule_3 = MockRule(
            rule_id="NO_FSM_IN_ORCHESTRATORS",
            name="No FSM in Orchestrators Rule",
            severity=EnumValidationSeverity.WARNING,
            should_pass=True,
        )
        validator = NodeArchitectureValidatorCompute(
            container=mock_container,
            rules=(rule_1, rule_2, rule_3),
        )
        request = ModelArchitectureValidationRequest(
            nodes=(sample_node,),
            handlers=(),
        )

        result = validator.compute(request)

        assert set(result.rules_checked) == {
            "NO_HANDLER_PUBLISHING",
            "PURE_REDUCERS",
            "NO_FSM_IN_ORCHESTRATORS",
        }
        assert result.rules_checked_count == 3

    def test_violation_details_captured(
        self,
        mock_container: MagicMock,
        sample_node: object,
    ) -> None:
        """Test that violation details (target_type, target_name, etc.) are correct.

        When a violation is created, it should capture the target's type name
        and target name (class name or string representation).
        """
        rule_with_details = MockRule(
            rule_id="NO_DIRECT_HANDLER_DISPATCH",
            name="No Direct Handler Dispatch Rule",
            severity=EnumValidationSeverity.ERROR,
            should_pass=False,
            message="Detailed failure",
            details={"extra_info": "test_value", "count": 42},
        )
        validator = NodeArchitectureValidatorCompute(
            container=mock_container,
            rules=(rule_with_details,),
        )
        request = ModelArchitectureValidationRequest(
            nodes=(sample_node,),
            handlers=(),
        )

        result = validator.compute(request)

        assert result.violation_count == 1
        violation = result.violations[0]
        assert violation.target_type == "SampleNode"
        assert "SampleNode" in violation.target_name
        assert violation.message == "Detailed failure"
        assert violation.details == {"extra_info": "test_value", "count": 42}


# =============================================================================
# Tests for Validation Result Boolean Behavior
# =============================================================================


@pytest.mark.unit
class TestValidationResultBoolBehavior:
    """Tests for custom __bool__ behavior in ModelArchitectureValidationResult.

    The result model overrides __bool__ to return True only when validation
    passed (no violations). This enables idiomatic usage like:

        if result:
            # Validation passed
        else:
            # Validation failed

    Warning:
        This differs from standard Pydantic behavior where bool(model)
        always returns True for any valid model instance.
    """

    def test_result_bool_true_when_valid(
        self,
        mock_container: MagicMock,
        passing_rule: MockRule,
        sample_node: object,
    ) -> None:
        """Test that bool(result) is True when no violations."""
        validator = NodeArchitectureValidatorCompute(
            container=mock_container,
            rules=(passing_rule,),
        )
        request = ModelArchitectureValidationRequest(
            nodes=(sample_node,),
            handlers=(),
        )

        result = validator.compute(request)

        assert bool(result) is True
        assert result  # Idiomatic usage
        assert result.valid is True

    def test_result_bool_false_when_violations(
        self,
        mock_container: MagicMock,
        failing_rule: MockRule,
        sample_node: object,
    ) -> None:
        """Test that bool(result) is False when violations exist."""
        validator = NodeArchitectureValidatorCompute(
            container=mock_container,
            rules=(failing_rule,),
        )
        request = ModelArchitectureValidationRequest(
            nodes=(sample_node,),
            handlers=(),
        )

        result = validator.compute(request)

        assert bool(result) is False
        if result:
            pytest.fail("Expected bool(result) to be False")
        assert result.valid is False

    def test_result_bool_matches_valid_property(
        self,
        mock_container: MagicMock,
        passing_rule: MockRule,
        failing_rule: MockRule,
        sample_node: object,
    ) -> None:
        """Verify bool(result) == result.valid in all cases."""
        validator_pass = NodeArchitectureValidatorCompute(
            container=mock_container,
            rules=(passing_rule,),
        )
        validator_fail = NodeArchitectureValidatorCompute(
            container=mock_container,
            rules=(failing_rule,),
        )
        request = ModelArchitectureValidationRequest(
            nodes=(sample_node,),
            handlers=(),
        )

        result_pass = validator_pass.compute(request)
        result_fail = validator_fail.compute(request)

        assert bool(result_pass) == result_pass.valid
        assert bool(result_fail) == result_fail.valid

    def test_result_bool_differs_from_none_check(
        self,
        mock_container: MagicMock,
        failing_rule: MockRule,
        sample_node: object,
    ) -> None:
        """Verify that bool(result) differs from `result is not None`.

        This documents the potentially surprising behavior where a valid
        model instance returns False for bool().
        """
        validator = NodeArchitectureValidatorCompute(
            container=mock_container,
            rules=(failing_rule,),
        )
        request = ModelArchitectureValidationRequest(
            nodes=(sample_node,),
            handlers=(),
        )

        result = validator.compute(request)

        # Model exists (is not None)
        assert result is not None

        # But bool(result) is False because violations exist
        assert bool(result) is False

    def test_passed_factory_creates_valid_result(self) -> None:
        """Test that ModelArchitectureValidationResult.passed() creates valid result."""
        result = ModelArchitectureValidationResult.passed(
            rules_checked=("RULE_1", "RULE_2"),
            nodes_checked=5,
            handlers_checked=10,
            correlation_id="test-correlation-id",
        )

        assert result.valid is True
        assert bool(result) is True
        assert result.violation_count == 0
        assert result.rules_checked == ("RULE_1", "RULE_2")
        assert result.nodes_checked == 5
        assert result.handlers_checked == 10
        assert result.correlation_id == "test-correlation-id"


# =============================================================================
# Tests for Violation Severity Behavior
# =============================================================================


@pytest.mark.unit
class TestViolationSeverityBehavior:
    """Tests for severity-based blocking behavior.

    ERROR severity violations should block startup, while WARNING and INFO
    severity violations should allow startup to proceed.
    """

    def test_error_severity_blocks_startup(
        self,
        mock_container: MagicMock,
        failing_rule: MockRule,
        sample_node: object,
    ) -> None:
        """Test that ERROR severity violations block startup."""
        validator = NodeArchitectureValidatorCompute(
            container=mock_container,
            rules=(failing_rule,),
        )
        request = ModelArchitectureValidationRequest(
            nodes=(sample_node,),
            handlers=(),
        )

        result = validator.compute(request)

        assert result.valid is False
        violation = result.violations[0]
        assert violation.severity == EnumValidationSeverity.ERROR
        assert violation.blocks_startup() is True

    def test_warning_severity_does_not_block_startup(
        self,
        mock_container: MagicMock,
        warning_rule: MockRule,
        sample_node: object,
    ) -> None:
        """Test that WARNING severity violations don't block startup."""
        validator = NodeArchitectureValidatorCompute(
            container=mock_container,
            rules=(warning_rule,),
        )
        request = ModelArchitectureValidationRequest(
            nodes=(sample_node,),
            handlers=(),
        )

        result = validator.compute(request)

        assert result.valid is False  # Still a violation
        violation = result.violations[0]
        assert violation.severity == EnumValidationSeverity.WARNING
        assert violation.blocks_startup() is False

    def test_info_severity_does_not_block_startup(
        self,
        mock_container: MagicMock,
        info_rule: MockRule,
        sample_node: object,
    ) -> None:
        """Test that INFO severity violations don't block startup."""
        validator = NodeArchitectureValidatorCompute(
            container=mock_container,
            rules=(info_rule,),
        )
        request = ModelArchitectureValidationRequest(
            nodes=(sample_node,),
            handlers=(),
        )

        result = validator.compute(request)

        assert result.valid is False  # Still a violation
        violation = result.violations[0]
        assert violation.severity == EnumValidationSeverity.INFO
        assert violation.blocks_startup() is False

    def test_mixed_severity_violations(
        self,
        mock_container: MagicMock,
        sample_node: object,
    ) -> None:
        """Test result with mixed severity violations."""
        error_rule = MockRule(
            rule_id="NO_HANDLER_PUBLISHING",
            name="No Handler Publishing Rule",
            severity=EnumValidationSeverity.ERROR,
            should_pass=False,
            message="Error violation",
        )
        warning_rule = MockRule(
            rule_id="PURE_REDUCERS",
            name="Pure Reducers Rule",
            severity=EnumValidationSeverity.WARNING,
            should_pass=False,
            message="Warning violation",
        )
        validator = NodeArchitectureValidatorCompute(
            container=mock_container,
            rules=(error_rule, warning_rule),
        )
        request = ModelArchitectureValidationRequest(
            nodes=(sample_node,),
            handlers=(),
        )

        result = validator.compute(request)

        assert result.valid is False
        assert result.violation_count == 2

        # Check that we have both severities
        severities = [v.severity for v in result.violations]
        assert EnumValidationSeverity.ERROR in severities
        assert EnumValidationSeverity.WARNING in severities

        # Only ERROR violations block startup
        blocking_violations = [v for v in result.violations if v.blocks_startup()]
        assert len(blocking_violations) == 1
        assert blocking_violations[0].severity == EnumValidationSeverity.ERROR


# =============================================================================
# Tests for Mock Rule Protocol Compliance
# =============================================================================


@pytest.mark.unit
class TestMockRuleProtocolCompliance:
    """Verify MockRule properly implements ProtocolArchitectureRule.

    These tests ensure our mock is a valid stand-in for real rule implementations.
    """

    def test_mock_rule_has_required_properties(self) -> None:
        """Test that MockRule has all required protocol properties."""
        rule = MockRule(
            rule_id="TEST_RULE",
            name="Test Rule",
            severity=EnumValidationSeverity.ERROR,
        )

        # Verify all required properties exist
        assert hasattr(rule, "rule_id")
        assert hasattr(rule, "name")
        assert hasattr(rule, "description")
        assert hasattr(rule, "severity")

        # Verify property values
        assert rule.rule_id == "TEST_RULE"
        assert rule.name == "Test Rule"
        assert "Mock rule" in rule.description
        assert rule.severity == EnumValidationSeverity.ERROR

    def test_mock_rule_check_returns_result(self) -> None:
        """Test that MockRule.check() returns ModelRuleCheckResult."""
        rule = MockRule(
            rule_id="TEST_RULE",
            name="Test Rule",
            severity=EnumValidationSeverity.WARNING,
            should_pass=True,
            message="Check passed",
        )

        result = rule.check(object())

        assert isinstance(result, ModelRuleCheckResult)
        assert result.passed is True
        assert result.rule_id == "TEST_RULE"
        assert result.message == "Check passed"

    def test_mock_rule_check_returns_failure(self) -> None:
        """Test that MockRule.check() can return failure results."""
        rule = MockRule(
            rule_id="FAIL_RULE",
            name="Fail Rule",
            severity=EnumValidationSeverity.ERROR,
            should_pass=False,
            message="Check failed",
            details={"reason": "test_reason"},
        )

        result = rule.check(object())

        assert isinstance(result, ModelRuleCheckResult)
        assert result.passed is False
        assert result.rule_id == "FAIL_RULE"
        assert result.message == "Check failed"
        assert result.details == {"reason": "test_reason"}
        assert result.is_violation() is True

    def test_mock_rule_is_runtime_checkable(self) -> None:
        """Test that MockRule passes runtime_checkable protocol check.

        ProtocolArchitectureRule is marked @runtime_checkable, so isinstance()
        should work. However, per ONEX conventions we prefer duck typing.
        """
        rule = MockRule(
            rule_id="TEST_RULE",
            name="Test Rule",
            severity=EnumValidationSeverity.ERROR,
        )

        # Runtime checkable protocol check
        assert isinstance(rule, ProtocolArchitectureRule)

        # Duck typing verification (preferred per ONEX conventions)
        assert hasattr(rule, "rule_id")
        assert hasattr(rule, "name")
        assert hasattr(rule, "description")
        assert hasattr(rule, "severity")
        assert hasattr(rule, "check") and callable(rule.check)

    def test_mock_rule_tracks_check_count(self) -> None:
        """Test that MockRule tracks how many times check() is called."""
        rule = MockRule(
            rule_id="TRACK_RULE",
            name="Track Rule",
            severity=EnumValidationSeverity.INFO,
        )

        assert rule.check_count == 0

        rule.check(object())
        assert rule.check_count == 1

        rule.check(object())
        rule.check(object())
        assert rule.check_count == 3


# =============================================================================
# Tests for Edge Cases
# =============================================================================


@pytest.mark.unit
class TestValidatorEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_rule_uses_description_when_no_message(
        self,
        mock_container: MagicMock,
        sample_node: object,
    ) -> None:
        """Test that rule description is used when check result has no message."""
        rule_no_message = MockRule(
            rule_id="NO_LOCAL_ONLY_PATHS",
            name="No Local Only Paths Rule",
            severity=EnumValidationSeverity.ERROR,
            should_pass=False,
            message=None,  # No message
        )
        validator = NodeArchitectureValidatorCompute(
            container=mock_container,
            rules=(rule_no_message,),
        )
        request = ModelArchitectureValidationRequest(
            nodes=(sample_node,),
            handlers=(),
        )

        result = validator.compute(request)

        violation = result.violations[0]
        # Should use rule description when message is None
        assert violation.message == rule_no_message.description

    def test_target_without_name_attribute(
        self,
        mock_container: MagicMock,
        failing_rule: MockRule,
    ) -> None:
        """Test handling of targets without __name__ attribute.

        The validator should fall back to str(target) when __name__ is not available.
        """
        # Create object without __name__
        target = object()

        validator = NodeArchitectureValidatorCompute(
            container=mock_container,
            rules=(failing_rule,),
        )
        request = ModelArchitectureValidationRequest(
            nodes=(target,),
            handlers=(),
        )

        result = validator.compute(request)

        violation = result.violations[0]
        # Should use str(target) fallback
        assert "object" in violation.target_name.lower()

    def test_rule_id_filter_with_nonexistent_rule(
        self,
        mock_container: MagicMock,
        passing_rule: MockRule,
        sample_node: object,
    ) -> None:
        """Test that filtering by non-existent rule ID results in no rules checked."""
        validator = NodeArchitectureValidatorCompute(
            container=mock_container,
            rules=(passing_rule,),
        )
        request = ModelArchitectureValidationRequest(
            nodes=(sample_node,),
            handlers=(),
            rule_ids=("NONEXISTENT_RULE",),
        )

        result = validator.compute(request)

        assert result.valid is True
        assert result.rules_checked == ()
        assert passing_rule.check_count == 0

    def test_validation_result_str_representation(
        self,
        mock_container: MagicMock,
        passing_rule: MockRule,
        failing_rule: MockRule,
        sample_node: object,
    ) -> None:
        """Test __str__ representation of validation results."""
        validator_pass = NodeArchitectureValidatorCompute(
            container=mock_container,
            rules=(passing_rule,),
        )
        validator_fail = NodeArchitectureValidatorCompute(
            container=mock_container,
            rules=(failing_rule,),
        )
        request = ModelArchitectureValidationRequest(
            nodes=(sample_node,),
            handlers=(),
        )

        result_pass = validator_pass.compute(request)
        result_fail = validator_fail.compute(request)

        pass_str = str(result_pass)
        fail_str = str(result_fail)

        assert "PASSED" in pass_str
        assert "FAILED" in fail_str
        assert "violations=0" in pass_str
        assert "violations=1" in fail_str

    def test_violation_format_for_logging(
        self,
        mock_container: MagicMock,
        failing_rule: MockRule,
        sample_node: object,
    ) -> None:
        """Test that violations can be formatted for logging."""
        validator = NodeArchitectureValidatorCompute(
            container=mock_container,
            rules=(failing_rule,),
        )
        request = ModelArchitectureValidationRequest(
            nodes=(sample_node,),
            handlers=(),
        )

        result = validator.compute(request)
        violation = result.violations[0]

        log_str = violation.format_for_logging()

        assert "[ERROR]" in log_str
        assert "PURE_REDUCERS" in log_str
        assert "Pure Reducers Rule" in log_str
        assert "This rule always fails" in log_str

    def test_violation_to_structured_dict(
        self,
        mock_container: MagicMock,
        failing_rule: MockRule,
        sample_node: object,
    ) -> None:
        """Test that violations can be converted to structured dict."""
        validator = NodeArchitectureValidatorCompute(
            container=mock_container,
            rules=(failing_rule,),
        )
        request = ModelArchitectureValidationRequest(
            nodes=(sample_node,),
            handlers=(),
        )

        result = validator.compute(request)
        violation = result.violations[0]

        structured = violation.to_structured_dict()

        assert structured["rule_id"] == "PURE_REDUCERS"
        assert structured["rule_name"] == "Pure Reducers Rule"
        assert structured["severity"] == "error"
        assert structured["message"] == "This rule always fails"

    def test_empty_rule_ids_filter_different_from_none(
        self,
        mock_container: MagicMock,
        passing_rule: MockRule,
        sample_node: object,
    ) -> None:
        """Test that empty rule_ids tuple vs None have different behavior.

        - rule_ids=None means check all rules
        - rule_ids=() means check no rules (empty filter)
        """
        validator = NodeArchitectureValidatorCompute(
            container=mock_container,
            rules=(passing_rule,),
        )

        # None = check all rules
        request_none = ModelArchitectureValidationRequest(
            nodes=(sample_node,),
            handlers=(),
            rule_ids=None,
        )
        result_none = validator.compute(request_none)

        # Empty tuple = check no rules
        request_empty = ModelArchitectureValidationRequest(
            nodes=(sample_node,),
            handlers=(),
            rule_ids=(),
        )
        result_empty = validator.compute(request_empty)

        assert result_none.rules_checked == ("NO_HANDLER_PUBLISHING",)
        assert result_empty.rules_checked == ()

    def test_model_rule_check_result_helper_methods(self) -> None:
        """Test ModelRuleCheckResult helper methods."""
        result = ModelRuleCheckResult(
            passed=False,
            rule_id="TEST_RULE",
        )

        # Test is_violation()
        assert result.is_violation() is True

        # Test with_message()
        updated = result.with_message("Updated message")
        assert updated.message == "Updated message"
        assert result.message is None  # Original unchanged (immutable)

        # Test with_details()
        with_details = result.with_details({"key": "value"})
        assert with_details.details == {"key": "value"}
        assert result.details is None  # Original unchanged


# =============================================================================
# Tests for Thread Safety and Immutability
# =============================================================================


@pytest.mark.unit
class TestImmutabilityAndThreadSafety:
    """Tests for immutability and thread safety of validation results."""

    def test_validation_result_is_frozen(self) -> None:
        """Verify ModelArchitectureValidationResult is frozen."""
        result = ModelArchitectureValidationResult.passed(
            rules_checked=("RULE_1",),
            nodes_checked=5,
        )

        config = ModelArchitectureValidationResult.model_config
        assert config.get("frozen") is True

    def test_validation_request_is_frozen(self) -> None:
        """Verify ModelArchitectureValidationRequest is frozen."""
        request = ModelArchitectureValidationRequest(
            nodes=(),
            handlers=(),
        )

        config = ModelArchitectureValidationRequest.model_config
        assert config.get("frozen") is True

    def test_violation_is_frozen(self) -> None:
        """Verify ModelArchitectureViolation is frozen."""
        violation = ModelArchitectureViolation(
            rule_id="TEST",
            rule_name="Test",
            severity=EnumValidationSeverity.ERROR,
            target_type="Node",
            target_name="TestNode",
            message="Test message",
        )

        config = ModelArchitectureViolation.model_config
        assert config.get("frozen") is True

    def test_rule_check_result_is_frozen(self) -> None:
        """Verify ModelRuleCheckResult is frozen."""
        result = ModelRuleCheckResult(
            passed=True,
            rule_id="TEST",
        )

        config = ModelRuleCheckResult.model_config
        assert config.get("frozen") is True


# =============================================================================
# =============================================================================
# Tests for Rule ID Validation Against Contract
# =============================================================================


@pytest.mark.unit
class TestRuleIdValidation:
    """Tests for rule_id validation against contract supported_rules.

    The validator validates rule_ids against the contract's SUPPORTED_RULE_IDS
    during construction (__init__). This ensures that only rules defined in
    the contract can be used, preventing configuration errors and version
    mismatches.

    Validation Behavior:
        - Valid/supported rule_ids: Accepted, validator constructed successfully
        - Invalid/unsupported rule_ids: RuntimeHostError raised during __init__
        - Error message: Includes the invalid rule_id and list of supported rules
        - Mixed valid/invalid: Fails on first invalid rule_id encountered

    Supported rule IDs (from contract_architecture_validator.yaml):
        - NO_HANDLER_PUBLISHING
        - PURE_REDUCERS
        - NO_FSM_IN_ORCHESTRATORS
        - NO_WORKFLOW_IN_REDUCERS
        - NO_DIRECT_HANDLER_DISPATCH
        - NO_LOCAL_ONLY_PATHS

    Related:
        - OMN-1138: Architecture Validator implementation
        - contract_architecture_validator.yaml: Source of supported_rules

    .. versionadded:: 0.8.0
        Created as part of OMN-1138 rule_id validation coverage.
    """

    def test_supported_rule_id_passes_construction(
        self,
        mock_container: MagicMock,
        sample_node: object,
    ) -> None:
        """Valid/supported rule IDs should be accepted during construction.

        When all rule_ids are in the contract's supported_rules list,
        the validator should be constructed successfully without error.
        """
        rule_a = MockRule(
            rule_id="NO_HANDLER_PUBLISHING",
            name="No Handler Publishing",
            severity=EnumValidationSeverity.ERROR,
            should_pass=True,
        )
        rule_b = MockRule(
            rule_id="PURE_REDUCERS",
            name="Pure Reducers",
            severity=EnumValidationSeverity.ERROR,
            should_pass=True,
        )

        # Should not raise any error
        validator = NodeArchitectureValidatorCompute(
            container=mock_container,
            rules=(rule_a, rule_b),
        )

        request = ModelArchitectureValidationRequest(
            nodes=(sample_node,),
            handlers=(),
        )

        result = validator.compute(request)

        assert result.valid is True
        assert set(result.rules_checked) == {"NO_HANDLER_PUBLISHING", "PURE_REDUCERS"}
        assert rule_a.check_count == 1
        assert rule_b.check_count == 1

    def test_all_supported_rule_ids_pass_construction(
        self,
        mock_container: MagicMock,
    ) -> None:
        """All supported rule_ids from the contract should be accepted.

        Verifies that each rule_id defined in SUPPORTED_RULE_IDS can be
        used to create a validator successfully.
        """
        # Test each supported rule_id individually
        supported_ids = [
            "NO_HANDLER_PUBLISHING",
            "PURE_REDUCERS",
            "NO_FSM_IN_ORCHESTRATORS",
            "NO_WORKFLOW_IN_REDUCERS",
            "NO_DIRECT_HANDLER_DISPATCH",
            "NO_LOCAL_ONLY_PATHS",
        ]

        for rule_id in supported_ids:
            rule = MockRule(
                rule_id=rule_id,
                name=f"Test Rule for {rule_id}",
                severity=EnumValidationSeverity.ERROR,
                should_pass=True,
            )

            # Should not raise any error
            validator = NodeArchitectureValidatorCompute(
                container=mock_container,
                rules=(rule,),
            )
            assert validator is not None

    def test_unsupported_rule_id_raises_error_during_construction(
        self,
        mock_container: MagicMock,
    ) -> None:
        """Unsupported rule IDs should raise RuntimeHostError during __init__.

        When a rule with an unsupported rule_id is provided during construction,
        the validator should raise RuntimeHostError immediately.

        This ensures:
        - Typos in rule_ids are caught early
        - Configuration errors are visible at startup
        - Version mismatches between validator and rules are detected
        """
        rule = MockRule(
            rule_id="NONEXISTENT_RULE",
            name="Nonexistent Rule",
            severity=EnumValidationSeverity.ERROR,
            should_pass=True,
        )

        with pytest.raises(RuntimeHostError):
            NodeArchitectureValidatorCompute(
                container=mock_container,
                rules=(rule,),
            )

    def test_error_message_includes_invalid_rule_id(
        self,
        mock_container: MagicMock,
    ) -> None:
        """Error message should clearly identify the unsupported rule_id.

        The error message should include the invalid rule_id so developers
        can quickly identify and fix configuration issues.
        """
        invalid_rule_id = "TOTALLY_INVALID_RULE_ID"
        rule = MockRule(
            rule_id=invalid_rule_id,
            name="Invalid Rule",
            severity=EnumValidationSeverity.ERROR,
            should_pass=True,
        )

        with pytest.raises(RuntimeHostError) as exc_info:
            NodeArchitectureValidatorCompute(
                container=mock_container,
                rules=(rule,),
            )

        # Error message should contain the invalid rule_id
        error_str = str(exc_info.value)
        assert invalid_rule_id in error_str

    def test_error_message_includes_supported_rule_ids(
        self,
        mock_container: MagicMock,
    ) -> None:
        """Error message should include list of supported rule_ids.

        To help developers quickly fix configuration issues, the error
        message should include the list of valid/supported rule_ids.
        """
        rule = MockRule(
            rule_id="UNKNOWN_RULE",
            name="Unknown Rule",
            severity=EnumValidationSeverity.ERROR,
            should_pass=True,
        )

        with pytest.raises(RuntimeHostError) as exc_info:
            NodeArchitectureValidatorCompute(
                container=mock_container,
                rules=(rule,),
            )

        error_str = str(exc_info.value)

        # Error should include at least one supported rule_id as suggestion
        supported_mentioned = any(
            supported in error_str
            for supported in [
                "NO_HANDLER_PUBLISHING",
                "PURE_REDUCERS",
                "NO_FSM_IN_ORCHESTRATORS",
            ]
        )
        assert supported_mentioned, (
            f"Error should list supported rule_ids. Got: {error_str}"
        )

    def test_mixed_valid_and_invalid_rule_ids_raises_error(
        self,
        mock_container: MagicMock,
    ) -> None:
        """Mix of valid and invalid rule_ids should fail on the invalid one.

        When both valid and invalid rule_ids are provided, the validator
        should raise an error for the invalid one during construction.
        """
        valid_rule = MockRule(
            rule_id="NO_HANDLER_PUBLISHING",
            name="Valid Rule",
            severity=EnumValidationSeverity.ERROR,
            should_pass=True,
        )
        invalid_rule = MockRule(
            rule_id="INVALID_RULE",
            name="Invalid Rule",
            severity=EnumValidationSeverity.WARNING,
            should_pass=True,
        )

        with pytest.raises(RuntimeHostError) as exc_info:
            NodeArchitectureValidatorCompute(
                container=mock_container,
                rules=(valid_rule, invalid_rule),
            )

        # Error message should contain the invalid rule_id
        assert "INVALID_RULE" in str(exc_info.value)

    def test_empty_rules_passes_construction(
        self,
        mock_container: MagicMock,
        sample_node: object,
    ) -> None:
        """Empty rules tuple should pass construction without error.

        A validator with no rules is valid (though it won't check anything).
        """
        # Should not raise any error
        validator = NodeArchitectureValidatorCompute(
            container=mock_container,
            rules=(),
        )

        request = ModelArchitectureValidationRequest(
            nodes=(sample_node,),
            handlers=(),
        )

        result = validator.compute(request)

        assert result.valid is True
        assert result.rules_checked == ()
        assert result.nodes_checked == 1

    def test_rule_ids_filter_works_with_supported_rules(
        self,
        mock_container: MagicMock,
        sample_node: object,
    ) -> None:
        """The rule_ids filter in request should work with supported rules.

        When rule_ids is specified in the request, only matching registered
        rules should be executed.
        """
        rule_a = MockRule(
            rule_id="NO_HANDLER_PUBLISHING",
            name="No Handler Publishing",
            severity=EnumValidationSeverity.ERROR,
            should_pass=True,
        )
        rule_b = MockRule(
            rule_id="PURE_REDUCERS",
            name="Pure Reducers",
            severity=EnumValidationSeverity.ERROR,
            should_pass=True,
        )

        validator = NodeArchitectureValidatorCompute(
            container=mock_container,
            rules=(rule_a, rule_b),
        )

        # Only check one of the two registered rules
        request = ModelArchitectureValidationRequest(
            nodes=(sample_node,),
            handlers=(),
            rule_ids=("NO_HANDLER_PUBLISHING",),
        )

        result = validator.compute(request)

        assert result.valid is True
        assert result.rules_checked == ("NO_HANDLER_PUBLISHING",)
        assert rule_a.check_count == 1
        assert rule_b.check_count == 0  # Not in rule_ids filter

    def test_none_rule_ids_checks_all_registered_rules(
        self,
        mock_container: MagicMock,
        sample_node: object,
    ) -> None:
        """rule_ids=None should check all registered rules.

        When rule_ids is None (default), all registered rules should be
        checked.
        """
        rule_a = MockRule(
            rule_id="NO_HANDLER_PUBLISHING",
            name="No Handler Publishing",
            severity=EnumValidationSeverity.ERROR,
            should_pass=True,
        )
        rule_b = MockRule(
            rule_id="PURE_REDUCERS",
            name="Pure Reducers",
            severity=EnumValidationSeverity.WARNING,
            should_pass=True,
        )

        validator = NodeArchitectureValidatorCompute(
            container=mock_container,
            rules=(rule_a, rule_b),
        )

        request = ModelArchitectureValidationRequest(
            nodes=(sample_node,),
            handlers=(),
            rule_ids=None,  # Check all rules
        )

        result = validator.compute(request)

        assert result.valid is True
        assert set(result.rules_checked) == {"NO_HANDLER_PUBLISHING", "PURE_REDUCERS"}
        assert rule_a.check_count == 1
        assert rule_b.check_count == 1

    def test_empty_rule_ids_filter_checks_no_rules(
        self,
        mock_container: MagicMock,
        sample_node: object,
    ) -> None:
        """rule_ids=() should check no rules (valid but unusual).

        An empty tuple for rule_ids means "check no rules" which is valid
        behavior, though unusual in practice.
        """
        rule_a = MockRule(
            rule_id="NO_HANDLER_PUBLISHING",
            name="No Handler Publishing",
            severity=EnumValidationSeverity.ERROR,
            should_pass=True,
        )

        validator = NodeArchitectureValidatorCompute(
            container=mock_container,
            rules=(rule_a,),
        )

        request = ModelArchitectureValidationRequest(
            nodes=(sample_node,),
            handlers=(),
            rule_ids=(),  # Check no rules
        )

        result = validator.compute(request)

        assert result.valid is True
        assert result.rules_checked == ()
        assert rule_a.check_count == 0
