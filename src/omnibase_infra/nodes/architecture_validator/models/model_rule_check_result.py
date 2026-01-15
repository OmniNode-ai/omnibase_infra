# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Rule check result model for architecture validation.

This module defines the result model returned by architecture validation rules
when checking targets (nodes, handlers, etc.) for compliance violations.

Result Semantics:
    The combination of ``passed`` and ``skipped`` fields determines the outcome:

    - ``passed=True, skipped=False``: Target was validated and passed the rule
    - ``passed=True, skipped=True``: Input not applicable to this rule (e.g.,
      rule checks file paths but received a non-path object)
    - ``passed=False, skipped=False``: Violation detected

    The ``skipped`` field disambiguates between "actually passed validation"
    and "input was not applicable to this rule". This is important for
    accurate reporting and metrics.

Thread Safety:
    ModelRuleCheckResult is immutable (frozen=True) after creation,
    making it thread-safe for concurrent read access.

Example:
    >>> from omnibase_infra.nodes.architecture_validator.models import ModelRuleCheckResult
    >>>
    >>> # Successful check - target validated and passed
    >>> result = ModelRuleCheckResult(
    ...     passed=True,
    ...     rule_id="NO_HANDLER_PUBLISHING",
    ... )
    >>> result.is_applicable()
    True
    >>>
    >>> # Skipped check - input not applicable to rule
    >>> result = ModelRuleCheckResult(
    ...     passed=True,
    ...     skipped=True,
    ...     rule_id="FILE_PATH_CONVENTION",
    ...     reason="Input is not a file path",
    ... )
    >>> result.is_applicable()
    False
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
    omnibase_infra.enums.EnumValidationSeverity:
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

    Result Semantics:
        The combination of ``passed`` and ``skipped`` determines the outcome:

        - ``passed=True, skipped=False``: Target was validated and passed
        - ``passed=True, skipped=True``: Input not applicable to this rule
        - ``passed=False, skipped=False``: Violation detected

        Use ``is_applicable()`` to check if the rule was actually evaluated,
        and ``is_violation()`` to check if a violation was detected.

    Attributes:
        passed: Whether the target passed the rule check. Note that when
            ``skipped=True``, this is ``True`` by convention but does not
            indicate actual validation success.
        skipped: Whether the check was skipped because the input was not
            applicable to this rule. For example, a file-path naming rule
            would skip non-path inputs. Defaults to ``False``.
        rule_id: Unique identifier of the rule that was checked.
        message: Human-readable message describing the result or violation.
        reason: Explanation for why the check was skipped. Only populated
            when ``skipped=True``. Example: "Input is not a file path".
        details: Additional details about the check result for debugging.

    Example:
        >>> # Violation detected
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
        >>> result.is_violation()
        True
        >>> result.is_applicable()
        True
        >>>
        >>> # Skipped - input not applicable
        >>> skipped_result = ModelRuleCheckResult(
        ...     passed=True,
        ...     skipped=True,
        ...     rule_id="FILE_NAMING_CONVENTION",
        ...     reason="Input is not a file path",
        ... )
        >>> skipped_result.passed
        True
        >>> skipped_result.is_applicable()
        False
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    passed: bool = Field(
        ...,
        description=(
            "Whether the target passed the rule check. Note that when "
            "skipped=True, this is True by convention but does not indicate "
            "actual validation success."
        ),
    )

    skipped: bool = Field(
        default=False,
        description=(
            "Whether the check was skipped because the input was not "
            "applicable to this rule. For example, a file-path naming rule "
            "would skip non-path inputs. When True, passed is also True by "
            "convention."
        ),
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

    reason: str | None = Field(
        default=None,
        description=(
            "Explanation for why the check was skipped. Only populated when "
            "skipped=True. Example: 'Input is not a file path'."
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

        A result is a violation only when the check was actually performed
        (not skipped) and failed.

        Returns:
            True if the check failed (violation detected), False otherwise.
            Returns False when skipped, even though passed=True.

        Example:
            >>> result = ModelRuleCheckResult(passed=False, rule_id="RULE_1")
            >>> result.is_violation()
            True
            >>> skipped = ModelRuleCheckResult(
            ...     passed=True, skipped=True, rule_id="RULE_1", reason="N/A"
            ... )
            >>> skipped.is_violation()
            False
        """
        return not self.passed

    def is_applicable(self) -> bool:
        """Check if the rule was applicable to the input.

        A rule is applicable when it was actually evaluated against the input,
        as opposed to being skipped because the input type was not relevant
        to the rule.

        Returns:
            True if the rule was evaluated (not skipped), False if skipped.

        Example:
            >>> # Rule was evaluated and passed
            >>> result = ModelRuleCheckResult(passed=True, rule_id="RULE_1")
            >>> result.is_applicable()
            True
            >>>
            >>> # Rule was skipped - input not applicable
            >>> skipped = ModelRuleCheckResult(
            ...     passed=True,
            ...     skipped=True,
            ...     rule_id="FILE_PATH_RULE",
            ...     reason="Input is not a file path",
            ... )
            >>> skipped.is_applicable()
            False
        """
        return not self.skipped

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
