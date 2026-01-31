# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Declarative Node Violation Model.

Defines the result structure for detected declarative node violations.
Used by the declarative node validator to report policy breaches with
full context for debugging and CI gate integration.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.enums import EnumValidationSeverity
from omnibase_infra.enums.enum_declarative_node_violation import (
    EnumDeclarativeNodeViolation,
)


class ModelDeclarativeNodeViolation(BaseModel):
    """Result of a declarative node violation detection.

    Contains full context about a detected violation for debugging,
    reporting, and CI gate integration. Each violation includes the
    source location, violation type, and suggested fix.

    Attributes:
        file_path: Path to the source file containing the violation.
        line_number: Line number where the violation was detected (1-indexed).
        violation_type: The specific violation category detected.
        code_snippet: The relevant code snippet showing the violation.
        suggestion: Human-readable suggestion for fixing the violation.
        severity: Severity classification (error blocks CI, warning is advisory).
        node_class_name: Name of the node class containing the violation.
        method_name: Name of the offending method/property if applicable.

    Example:
        >>> violation = ModelDeclarativeNodeViolation(
        ...     file_path=Path("/src/nodes/my_node/node.py"),
        ...     line_number=42,
        ...     violation_type=EnumDeclarativeNodeViolation.CUSTOM_METHOD,
        ...     code_snippet="def compute(self, data): ...",
        ...     suggestion="Move business logic to a Handler class...",
        ...     node_class_name="MyNode",
        ...     method_name="compute",
        ... )

    Note:
        Violations with severity='error' should block CI pipelines.
        All declarative node violations are errors by default.
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
    violation_type: EnumDeclarativeNodeViolation = Field(
        ...,
        description="The specific declarative node violation category",
    )
    code_snippet: str = Field(
        ...,
        description="The relevant code snippet showing the violation",
    )
    suggestion: str = Field(
        ...,
        description="Human-readable suggestion for fixing the violation",
    )
    severity: EnumValidationSeverity = Field(
        default=EnumValidationSeverity.ERROR,
        description="Severity classification: 'error' blocks CI, 'warning' is advisory",
    )
    node_class_name: str = Field(
        default="",
        description="Name of the node class containing the violation",
    )
    method_name: str = Field(
        default="",
        description="Name of the offending method/property if applicable",
    )

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        str_strip_whitespace=True,
        use_enum_values=False,
    )

    def is_blocking(self) -> bool:
        """Check if this violation should block CI.

        Returns:
            True if severity is 'error', False for 'warning'.
        """
        return self.severity == EnumValidationSeverity.ERROR

    @property
    def is_exemptable(self) -> bool:
        """Check if this violation can be exempted via ONEX_EXCLUDE comment.

        Delegates to the violation type's is_exemptable property.
        Some violations like SYNTAX_ERROR and NO_NODE_CLASS cannot be
        exempted because they indicate fundamental issues with the source file.

        Returns:
            True if the violation can be exempted via inline comment.
        """
        return self.violation_type.is_exemptable

    def format_for_ci(self) -> str:
        """Format violation for CI output (GitHub Actions compatible).

        Returns:
            Formatted string in GitHub Actions annotation format.

        Example:
            ::error file=src/nodes/my_node/node.py,line=42::CUSTOM_METHOD: ...
        """
        annotation_type = "error" if self.is_blocking() else "warning"
        context = f" in {self.node_class_name}" if self.node_class_name else ""
        return (
            f"::{annotation_type} file={self.file_path},line={self.line_number}::"
            f"{self.violation_type.value}{context}: {self.code_snippet}"
        )

    def format_human_readable(self) -> str:
        """Format violation for human-readable console output.

        Includes exemption hint for exemptable violation types.

        Returns:
            Formatted string with file location and suggestion.

        Example:
            src/nodes/my_node/node.py:42 - CUSTOM_METHOD
            Class: MyNode
            Method: compute
            Code: def compute(self, data): ...
            Suggestion: Move business logic to a Handler class...
            Exemption: Add '# ONEX_EXCLUDE: declarative_node' above the class
        """
        lines = [
            f"{self.file_path}:{self.line_number} - {self.violation_type.value}",
        ]
        if self.node_class_name:
            lines.append(f"  Class: {self.node_class_name}")
        if self.method_name:
            lines.append(f"  Method: {self.method_name}")
        lines.append(f"  Code: {self.code_snippet}")
        lines.append(f"  Suggestion: {self.suggestion}")
        if self.is_exemptable:
            lines.append(
                "  Exemption: Add '# ONEX_EXCLUDE: declarative_node' above the class"
            )
        return "\n".join(lines)


__all__ = ["ModelDeclarativeNodeViolation"]
