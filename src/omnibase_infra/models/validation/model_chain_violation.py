# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Chain Violation Model.

Defines the result structure for detected correlation and causation chain
violations. Used by the chain validation system to report violations with
full context for debugging and observability.
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.enums import EnumChainViolationType, EnumValidationSeverity


class ModelChainViolation(BaseModel):
    """Result of a correlation or causation chain violation detection.

    Contains full context about a detected chain violation for debugging,
    structured logging, and observability integration. Each violation includes
    the expected and actual values, message identifiers, and severity.

    Correlation and causation chains are fundamental to distributed tracing
    and debugging in event-driven systems:

    - **Correlation Chain**: All messages derived from a single request share
      the same correlation_id, enabling end-to-end trace visibility.
    - **Causation Chain**: Each message's causation_id references its direct
      parent's message_id, forming an unbroken lineage back to the origin.

    Attributes:
        violation_type: The specific chain violation detected.
        expected_value: The UUID value that was expected (e.g., parent's
            correlation_id or message_id).
        actual_value: The UUID value that was actually found.
        message_id: The message_id where the violation was detected.
        parent_message_id: The parent message's message_id, if applicable.
        violation_message: Human-readable description of the violation.
        severity: Severity classification (error blocks processing, warning
            is advisory).

    Example:
        >>> from uuid import uuid4
        >>> violation = ModelChainViolation(
        ...     violation_type=EnumChainViolationType.CORRELATION_MISMATCH,
        ...     expected_value=uuid4(),
        ...     actual_value=uuid4(),
        ...     message_id=uuid4(),
        ...     parent_message_id=uuid4(),
        ...     violation_message="Child message has different correlation_id than parent",
        ...     severity=EnumValidationSeverity.ERROR,
        ... )
        >>> violation.is_blocking()
        True
        >>> log_line = violation.format_for_logging()
    """

    violation_type: EnumChainViolationType = Field(
        ...,
        description="The specific chain violation type detected",
    )
    expected_value: UUID | None = Field(
        default=None,
        description=(
            "The expected UUID value (e.g., parent's correlation_id or message_id). "
            "None if the expected value cannot be determined."
        ),
    )
    actual_value: UUID | None = Field(
        default=None,
        description=(
            "The actual UUID value found in the message. "
            "None if the value was missing entirely."
        ),
    )
    message_id: UUID = Field(
        ...,
        description="The message_id of the message where the violation was detected",
    )
    parent_message_id: UUID | None = Field(
        default=None,
        description=(
            "The parent message's message_id, if applicable. "
            "None for root messages or when parent cannot be determined."
        ),
    )
    violation_message: str = Field(
        ...,
        min_length=1,
        description="Human-readable description of the violation",
    )
    severity: EnumValidationSeverity = Field(
        default=EnumValidationSeverity.ERROR,
        description=(
            "Severity classification: 'error' indicates a critical chain break "
            "that should block processing, 'warning' is advisory for potential issues"
        ),
    )

    model_config = ConfigDict(
        frozen=True,
        str_strip_whitespace=True,
        use_enum_values=False,  # Keep enum objects for type safety
    )

    def is_blocking(self) -> bool:
        """Check if this violation should block message processing.

        Returns:
            True if severity is 'error', False for 'warning'.

        Example:
            >>> violation.is_blocking()
            True
        """
        return self.severity == EnumValidationSeverity.ERROR

    def format_for_logging(self) -> str:
        """Format violation for structured logging output.

        Returns:
            Formatted string suitable for structured logging systems.
            Includes all relevant context for debugging chain violations.

        Example:
            >>> violation.format_for_logging()
            '[error] CORRELATION_MISMATCH on message=abc123: Child message has...'
        """
        expected_str = str(self.expected_value) if self.expected_value else "None"
        actual_str = str(self.actual_value) if self.actual_value else "None"
        parent_str = str(self.parent_message_id) if self.parent_message_id else "None"

        return (
            f"[{self.severity}] {self.violation_type.value.upper()} "
            f"on message={self.message_id}: {self.violation_message} "
            f"(expected={expected_str}, actual={actual_str}, parent={parent_str})"
        )

    def to_structured_dict(self) -> dict[str, str | None]:
        """Convert violation to a dictionary suitable for structured logging.

        Returns:
            Dictionary with all violation fields as strings, suitable for
            JSON logging or metrics systems.

        Example:
            >>> violation.to_structured_dict()
            {'violation_type': 'correlation_mismatch', 'severity': 'error', ...}
        """
        return {
            "violation_type": self.violation_type.value,
            "severity": self.severity,
            "message_id": str(self.message_id),
            "parent_message_id": (
                str(self.parent_message_id) if self.parent_message_id else None
            ),
            "expected_value": str(self.expected_value) if self.expected_value else None,
            "actual_value": str(self.actual_value) if self.actual_value else None,
            "violation_message": self.violation_message,
        }


__all__ = ["ModelChainViolation"]
