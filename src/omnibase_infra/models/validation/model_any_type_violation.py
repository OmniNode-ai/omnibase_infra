# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Any Type Violation Model.

Defines the result structure for detected Any type violations.
Used by the Any type validator to report policy breaches with
full context for debugging and CI gate integration.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.enums import EnumAnyTypeViolation, EnumValidationSeverity


class ModelAnyTypeViolation(BaseModel):
    """Result of an Any type violation detection.

    Contains full context about a detected violation for debugging,
    reporting, and CI gate integration. Each violation includes the
    source location, violation type, and suggested fix.

    Attributes:
        file_path: Path to the source file containing the violation.
        line_number: Line number where the violation was detected (1-indexed).
        column: Column offset where Any appears (0-indexed).
        violation_type: The specific violation category detected.
        code_snippet: The relevant code snippet showing the violation.
        suggestion: Human-readable suggestion for fixing the violation.
        severity: Severity classification (error blocks CI, warning is advisory).
        context_name: Name of function, class, or variable where violation occurs.

    Example:
        >>> violation = ModelAnyTypeViolation(
        ...     file_path=Path("/src/handlers/data_handler.py"),
        ...     line_number=42,
        ...     column=15,
        ...     violation_type=EnumAnyTypeViolation.FUNCTION_PARAMETER,
        ...     code_snippet="def process(data: Any) -> str:",
        ...     suggestion="Replace Any with specific type...",
        ...     context_name="process",
        ... )

    Note:
        Violations with severity='error' should block CI pipelines.
        Violations with severity='warning' are advisory and should be logged.
    """

    file_path: Path = Field(
        ...,
        description="Path to the source file containing the violation",
    )
    line_number: int = Field(
        ...,
        ge=1,
        description="Line number where the violation was detected (1-indexed)",
    )
    column: int = Field(
        ...,
        ge=0,
        description="Column offset where Any appears in the line (0-indexed)",
    )
    violation_type: EnumAnyTypeViolation = Field(
        ...,
        description="The specific Any type violation category",
    )
    code_snippet: str = Field(
        ...,
        min_length=1,
        description="The relevant code snippet showing the violation",
    )
    suggestion: str = Field(
        ...,
        min_length=1,
        description="Human-readable suggestion for fixing the violation",
    )
    severity: EnumValidationSeverity = Field(
        default=EnumValidationSeverity.ERROR,
        description="Severity classification: 'error' blocks CI, 'warning' is advisory",
    )
    context_name: str = Field(
        default="",
        description="Name of function, class, or variable where violation occurs",
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

        Example:
            ::error file=src/handler.py,line=42,col=15::FUNCTION_PARAMETER: ...
        """
        annotation_type = "error" if self.is_blocking() else "warning"
        return (
            f"::{annotation_type} file={self.file_path},line={self.line_number},"
            f"col={self.column}::{self.violation_type.value}: {self.code_snippet}"
        )

    def format_human_readable(self) -> str:
        """Format violation for human-readable console output.

        Returns:
            Formatted string with file location and suggestion.

        Example:
            src/handler.py:42:15 - FUNCTION_PARAMETER
            Code: def process(data: Any) -> str:
            Suggestion: Replace Any with specific type...
        """
        lines = [
            f"{self.file_path}:{self.line_number}:{self.column} - {self.violation_type.value}",
            f"  Code: {self.code_snippet}",
            f"  Suggestion: {self.suggestion}",
        ]
        if self.context_name:
            lines.insert(1, f"  Context: {self.context_name}")
        return "\n".join(lines)


__all__ = ["ModelAnyTypeViolation"]
