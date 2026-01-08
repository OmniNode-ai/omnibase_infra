# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Rule check result model for architecture validation.

This module defines the result model returned by architecture validation rules
when checking targets (nodes, handlers, etc.) for compliance violations.

Thread Safety:
    ModelRuleCheckResult is immutable (frozen=True) after creation,
    making it thread-safe for concurrent read access.

Example:
    >>> from omnibase_infra.nodes.architecture_validator.models import ModelRuleCheckResult
    >>>
    >>> # Successful check
    >>> result = ModelRuleCheckResult(
    ...     passed=True,
    ...     rule_id="NO_HANDLER_PUBLISHING",
    ... )
    >>>
    >>> # Failed check with details
    >>> result = ModelRuleCheckResult(
    ...     passed=False,
    ...     rule_id="NO_HANDLER_PUBLISHING",
    ...     message="Handler has direct event bus access",
    ...     details={"handler_class": "MyHandler", "forbidden_attribute": "_event_bus"},
    ... )

See Also:
    omnibase_infra.nodes.architecture_validator.protocols.ProtocolArchitectureRule:
        Protocol that produces these results
    omnibase_infra.nodes.architecture_validator.enums.EnumValidationSeverity:
        Severity levels for rule violations
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["ModelRuleCheckResult"]


class ModelRuleCheckResult(BaseModel):
    """Result of checking a target against an architecture rule.

    Captures the outcome of validating a node, handler, or other architectural
    component against a specific rule. Used for aggregating validation results
    and reporting violations.

    Attributes:
        passed: Whether the target passed the rule check.
        rule_id: Unique identifier of the rule that was checked.
        message: Human-readable message describing the result or violation.
        details: Additional details about the check result for debugging.

    Example:
        >>> result = ModelRuleCheckResult(
        ...     passed=False,
        ...     rule_id="NO_WORKFLOW_LOGIC_IN_REDUCER",
        ...     message="Reducer contains orchestration logic",
        ...     details={
        ...         "reducer_class": "MyReducer",
        ...         "forbidden_method": "execute_workflow",
        ...     },
        ... )
        >>> result.passed
        False
        >>> result.rule_id
        'NO_WORKFLOW_LOGIC_IN_REDUCER'
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    passed: bool = Field(
        ...,
        description="Whether the target passed the rule check.",
    )

    rule_id: str = Field(
        ...,
        description="Unique identifier of the rule that was checked.",
        min_length=1,
    )

    message: str | None = Field(
        default=None,
        description=(
            "Human-readable message describing the result or violation. "
            "Typically populated when the check fails."
        ),
    )

    details: dict[str, object] | None = Field(
        default=None,
        description=(
            "Additional details about the check result for debugging. "
            "May include target class names, forbidden attributes, etc."
        ),
    )

    def is_violation(self) -> bool:
        """Check if this result represents a rule violation.

        Returns:
            True if the check failed (violation detected), False otherwise.

        Example:
            >>> result = ModelRuleCheckResult(passed=False, rule_id="RULE_1")
            >>> result.is_violation()
            True
        """
        return not self.passed

    def with_message(self, message: str) -> ModelRuleCheckResult:
        """Create a new result with an updated message.

        Args:
            message: The new message to set.

        Returns:
            New ModelRuleCheckResult with the updated message.

        Example:
            >>> result = ModelRuleCheckResult(passed=False, rule_id="RULE_1")
            >>> updated = result.with_message("Violation detected")
            >>> updated.message
            'Violation detected'
        """
        return self.model_copy(update={"message": message})

    def with_details(self, details: dict[str, object]) -> ModelRuleCheckResult:
        """Create a new result with updated details.

        Args:
            details: The new details to set.

        Returns:
            New ModelRuleCheckResult with the updated details.

        Example:
            >>> result = ModelRuleCheckResult(passed=False, rule_id="RULE_1")
            >>> updated = result.with_details({"class": "MyHandler"})
            >>> updated.details
            {'class': 'MyHandler'}
        """
        return self.model_copy(update={"details": details})
