# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Architecture violation model for the architecture validator node.

This module defines the data model for individual architecture violations
detected during validation. Each violation captures the rule that was
broken, where it occurred, and how to fix it.

Example:
    >>> from omnibase_infra.nodes.architecture_validator.models import (
    ...     ModelArchitectureViolation,
    ...     EnumViolationSeverity,
    ... )
    >>> violation = ModelArchitectureViolation(
    ...     rule_id="ARCH-001",
    ...     rule_name="No Any Types",
    ...     severity=EnumViolationSeverity.ERROR,
    ...     file_path="src/mymodule/service.py",
    ...     line_number=42,
    ...     message="Use of 'Any' type is forbidden",
    ...     suggestion="Replace with 'object' or a specific type",
    ... )
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from .enum_violation_severity import EnumViolationSeverity


class ModelArchitectureViolation(BaseModel):
    """A single architecture violation detected during validation.

    This model captures all information needed to understand and fix
    an architecture violation, including the rule that was broken,
    where it occurred, and how to fix it.

    Attributes:
        rule_id: Unique identifier for the rule (e.g., "ARCH-001").
        rule_name: Human-readable name describing the rule.
        severity: Whether this is an error (blocking) or warning (advisory).
        file_path: Path to the file containing the violation.
        line_number: Line number where the violation occurs (if applicable).
        message: Detailed description of what is wrong.
        suggestion: How to fix the violation (if available).

    Example:
        >>> violation = ModelArchitectureViolation(
        ...     rule_id="ARCH-002",
        ...     rule_name="One Model Per File",
        ...     severity=EnumViolationSeverity.ERROR,
        ...     file_path="src/models/combined.py",
        ...     line_number=15,
        ...     message="File contains multiple Model* classes",
        ...     suggestion="Split into separate files: model_user.py, model_order.py",
        ... )
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
    )

    rule_id: str = Field(
        ...,
        description="Unique identifier for the rule (e.g., 'ARCH-001').",
        min_length=1,
    )
    rule_name: str = Field(
        ...,
        description="Human-readable name describing the rule.",
        min_length=1,
    )
    severity: EnumViolationSeverity = Field(
        default=EnumViolationSeverity.ERROR,
        description="Severity level of the violation.",
    )
    file_path: str = Field(
        ...,
        description="Path to the file containing the violation.",
        min_length=1,
    )
    line_number: int | None = Field(
        default=None,
        description="Line number where the violation occurs (if applicable).",
        ge=1,
    )
    message: str = Field(
        ...,
        description="Detailed description of what is wrong.",
        min_length=1,
    )
    suggestion: str | None = Field(
        default=None,
        description="How to fix the violation (if available).",
    )

    def __str__(self) -> str:
        """Return a human-readable string representation for debugging.

        Returns:
            String showing rule ID, file path, and message.
        """
        location = f"{self.file_path}"
        if self.line_number is not None:
            location += f":{self.line_number}"
        return f"[{self.rule_id}] {location}: {self.message}"


__all__ = ["ModelArchitectureViolation"]
