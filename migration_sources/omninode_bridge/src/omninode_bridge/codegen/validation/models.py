#!/usr/bin/env python3
"""
Data models for node validation pipeline.

Defines validation stages and results for ONEX v2.0 compliance checking.
"""

from dataclasses import dataclass, field
from enum import Enum


class EnumValidationStage(str, Enum):
    """
    Validation stage identifier.

    Each stage validates a specific aspect of generated node code:
    - SYNTAX: Python syntax correctness
    - AST: Abstract Syntax Tree structure
    - TYPE_CHECKING: Type hint validation (mypy)
    - IMPORTS: Import resolution and availability
    - ONEX_COMPLIANCE: ONEX v2.0 framework compliance
    - SECURITY: Security vulnerability scanning
    """

    SYNTAX = "syntax"
    AST = "ast"
    TYPE_CHECKING = "type_checking"
    IMPORTS = "imports"
    ONEX_COMPLIANCE = "onex_compliance"
    SECURITY = "security"


@dataclass
class ModelValidationResult:
    """
    Validation result for a single stage.

    Tracks pass/fail status, errors, warnings, and execution time
    for a specific validation stage.

    Attributes:
        stage: Which validation stage this result represents
        passed: Whether the stage passed validation
        errors: Critical errors found (cause validation failure)
        warnings: Non-critical warnings (informational)
        execution_time_ms: Time taken to run this stage
        suggestions: Actionable suggestions for fixing issues
    """

    stage: EnumValidationStage
    passed: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    execution_time_ms: float = 0.0
    suggestions: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        """Format validation result as human-readable string."""
        status = "✅ PASSED" if self.passed else "❌ FAILED"
        lines = [
            f"Stage: {self.stage.value} - {status} ({self.execution_time_ms:.1f}ms)"
        ]

        if self.errors:
            lines.append(f"  Errors ({len(self.errors)}):")
            for error in self.errors[:5]:  # Limit to first 5
                lines.append(f"    • {error}")
            if len(self.errors) > 5:
                lines.append(f"    ... and {len(self.errors) - 5} more")

        if self.warnings:
            lines.append(f"  Warnings ({len(self.warnings)}):")
            for warning in self.warnings[:3]:  # Limit to first 3
                lines.append(f"    ⚠ {warning}")
            if len(self.warnings) > 3:
                lines.append(f"    ... and {len(self.warnings) - 3} more")

        if self.suggestions:
            lines.append("  Suggestions:")
            for suggestion in self.suggestions[:3]:
                lines.append(f"    💡 {suggestion}")

        return "\n".join(lines)


__all__ = ["EnumValidationStage", "ModelValidationResult"]
