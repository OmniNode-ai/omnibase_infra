# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Architecture Violation Model.

Defines the structure for a single architecture rule violation detected
during validation. Used by the architecture validator to report violations
with full context for debugging and remediation.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.enums import EnumValidationSeverity


class ModelArchitectureViolation(BaseModel):
    """A single architecture rule violation.

    Captures details about what rule was violated, where, and why. Each
    violation includes the rule context, target information, and optional
    remediation guidance.

    Architecture violations are detected during validation phases and can
    range from informational notices to blocking errors that prevent
    startup.

    Attributes:
        rule_id: Unique identifier for the rule that was violated.
        rule_name: Human-readable name of the violated rule.
        severity: Severity level determining how the violation is handled.
        target_type: Type of target that violated the rule (e.g., 'handler',
            'node', 'model').
        target_name: Name or identifier of the violating target.
        message: Human-readable description of the violation.
        location: File path and line number if available.
        suggestion: Suggested fix or remediation for this violation.
        details: Additional context-specific details for debugging.

    Example:
        >>> violation = ModelArchitectureViolation(
        ...     rule_id="no-any-types",
        ...     rule_name="No Any Types",
        ...     severity=EnumValidationSeverity.ERROR,
        ...     target_type="model",
        ...     target_name="ModelUserData",
        ...     message="Field 'payload' uses 'Any' type which is forbidden",
        ...     location="src/models/model_user_data.py:42",
        ...     suggestion="Use 'object' or a specific type instead of 'Any'",
        ... )
        >>> violation.blocks_startup()
        True
    """

    rule_id: str = Field(
        ...,
        min_length=1,
        description="Unique identifier of the rule that was violated",
    )
    rule_name: str = Field(
        ...,
        min_length=1,
        description="Human-readable name of the violated rule",
    )
    severity: EnumValidationSeverity = Field(
        ...,
        description="Severity level determining how this violation is handled",
    )
    target_type: str = Field(
        ...,
        min_length=1,
        description="Type of target that violated the rule (e.g., 'handler', 'node', 'model')",
    )
    target_name: str = Field(
        ...,
        min_length=1,
        description="Name or identifier of the violating target",
    )
    message: str = Field(
        ...,
        min_length=1,
        description="Human-readable description of the violation",
    )
    location: str | None = Field(
        default=None,
        description="File path and line number if available (e.g., 'src/foo.py:42')",
    )
    suggestion: str | None = Field(
        default=None,
        description="Suggested fix or remediation for this violation",
    )
    details: dict[str, object] | None = Field(
        default=None,
        description="Additional context-specific details for debugging",
    )

    model_config = ConfigDict(
        frozen=True,
        str_strip_whitespace=True,
        use_enum_values=False,  # Keep enum objects for type safety
    )

    def blocks_startup(self) -> bool:
        """Check if this violation should block runtime startup.

        Returns:
            True if severity is ERROR or CRITICAL, False otherwise.

        Example:
            >>> violation.blocks_startup()
            True
        """
        return self.severity.blocks_startup()

    def format_for_logging(self) -> str:
        """Format violation for structured logging output.

        Returns:
            Formatted string suitable for structured logging systems.
            Includes all relevant context for debugging violations.

        Example:
            >>> violation.format_for_logging()
            '[ERROR] no-any-types (No Any Types) on model/ModelUserData: ...'
        """
        location_str = f" at {self.location}" if self.location else ""
        suggestion_str = f" - Suggestion: {self.suggestion}" if self.suggestion else ""

        return (
            f"[{self.severity.value.upper()}] {self.rule_id} ({self.rule_name}) "
            f"on {self.target_type}/{self.target_name}{location_str}: "
            f"{self.message}{suggestion_str}"
        )

    def to_structured_dict(self) -> dict[str, str | None]:
        """Convert violation to a dictionary suitable for structured logging.

        Returns:
            Dictionary with all violation fields as strings, suitable for
            JSON logging or metrics systems.

        Note:
            The ``details`` field is intentionally excluded from this output
            because it may contain complex objects that are not JSON-serializable.
            For full violation details, access the ``details`` attribute directly.

        Example:
            >>> violation.to_structured_dict()
            {'rule_id': 'no-any-types', 'severity': 'error', ...}
        """
        return {
            "rule_id": self.rule_id,
            "rule_name": self.rule_name,
            "severity": self.severity.value,
            "target_type": self.target_type,
            "target_name": self.target_name,
            "message": self.message,
            "location": self.location,
            "suggestion": self.suggestion,
        }


__all__ = ["ModelArchitectureViolation"]
