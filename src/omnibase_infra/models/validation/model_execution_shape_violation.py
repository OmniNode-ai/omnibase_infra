# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Execution Shape Violation Result Model.

Defines the result structure for detected execution shape violations.
Used by the execution shape validator to report constraint breaches
with full context for debugging and CI gate integration.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.enums import (
    EnumExecutionShapeViolation,
    EnumNodeArchetype,
    EnumValidationSeverity,
)


class ModelExecutionShapeViolationResult(BaseModel):
    """Result of an execution shape violation detection.

    Contains full context about a detected violation for debugging,
    reporting, and CI gate integration. Each violation includes the
    source location, violation type, and severity classification.

    Attributes:
        violation_type: The specific violation detected.
        node_archetype: The node archetype where the violation occurred, or None
            if the node archetype cannot be determined (e.g., runtime validation
            without node context, or file-level errors like syntax errors).
        file_path: Absolute path to the source file containing the violation.
        line_number: Line number where the violation was detected.
        message: Human-readable description of the violation.
        severity: Severity classification (error blocks CI, warning is advisory).

    Example:
        >>> result = ModelExecutionShapeViolationResult(
        ...     violation_type=EnumExecutionShapeViolation.REDUCER_RETURNS_EVENTS,
        ...     node_archetype=EnumNodeArchetype.REDUCER,
        ...     file_path="/src/handlers/order_reducer.py",
        ...     line_number=42,
        ...     message="Reducer 'OrderReducer.handle' returns EVENT type 'OrderCreated'",
        ...     severity=EnumValidationSeverity.ERROR,
        ... )

    Note:
        Violations with severity=EnumValidationSeverity.ERROR should block CI pipelines.
        Violations with severity=EnumValidationSeverity.WARNING are advisory and should be logged.
    """

    violation_type: EnumExecutionShapeViolation = Field(
        ...,
        description="The specific execution shape violation detected",
    )
    node_archetype: EnumNodeArchetype | None = Field(
        default=None,
        description=(
            "The node archetype where the violation occurred, or None if unknown. "
            "Node archetype may be None for runtime validation without node context, "
            "file-level errors (syntax errors, file not found), or when the node "
            "archetype cannot be inferred from the code structure."
        ),
    )
    file_path: str = Field(
        ...,
        description="Absolute path to the source file containing the violation",
    )
    line_number: int = Field(
        ...,
        ge=1,
        description="Line number where the violation was detected (1-indexed)",
    )
    message: str = Field(
        ...,
        min_length=1,
        description="Human-readable description of the violation",
    )
    severity: EnumValidationSeverity = Field(
        default=EnumValidationSeverity.ERROR,
        description="Severity classification: 'error' blocks CI, 'warning' is advisory",
    )

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        str_strip_whitespace=True,
        use_enum_values=False,  # Keep enum objects for type safety
    )

    def is_blocking(self) -> bool:
        """Check if this violation should block CI.

        Returns:
            True if severity is 'error', False for 'warning'.
        """
        return self.severity == EnumValidationSeverity.ERROR

    def format_for_ci(self) -> str:
        """Format violation for CI output (GitHub Actions compatible).

        Returns:
            Formatted string in GitHub Actions annotation format.
            Includes node archetype when available for better diagnostics.

        Example:
            ::error file=src/handler.py,line=42::[REDUCER] REDUCER_RETURNS_EVENTS: ...
            ::error file=src/handler.py,line=42::UNMAPPED_MESSAGE_ROUTE: ...
        """
        annotation_type = "error" if self.is_blocking() else "warning"
        archetype_prefix = (
            f"[{self.node_archetype.value.upper()}] " if self.node_archetype else ""
        )
        return (
            f"::{annotation_type} file={self.file_path},line={self.line_number}::"
            f"{archetype_prefix}{self.violation_type.value}: {self.message}"
        )


__all__ = ["ModelExecutionShapeViolationResult"]
