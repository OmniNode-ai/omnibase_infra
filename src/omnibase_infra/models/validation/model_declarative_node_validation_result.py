# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Declarative Node Validation Result Model.

Defines the aggregate result structure for declarative node validation operations.
Used by the validator to provide a structured result for CI pipeline integration
with convenience methods for reporting.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.models.validation.model_declarative_node_violation import (
    ModelDeclarativeNodeViolation,
)


class ModelDeclarativeNodeValidationResult(BaseModel):
    """Aggregate result of declarative node validation.

    Provides a structured result for CI pipeline integration with
    convenience methods for reporting.

    Attributes:
        passed: True if no blocking violations found.
        violations: List of all detected violations.
        files_checked: Number of node.py files that were validated.
        total_violations: Total count of violations.
        blocking_count: Count of blocking (error severity) violations.
        imperative_nodes: List of node class names that are imperative.

    Example:
        >>> result = ModelDeclarativeNodeValidationResult.from_violations(
        ...     violations=[violation1, violation2],
        ...     files_checked=10,
        ... )
        >>> if not result.passed:
        ...     print(result.format_summary())
    """

    passed: bool = Field(
        ...,
        description="True if no blocking violations found",
    )
    violations: list[ModelDeclarativeNodeViolation] = Field(
        default_factory=list,
        description="List of all detected violations",
    )
    files_checked: int = Field(
        default=0,
        ge=0,
        description="Number of node.py files that were validated",
    )
    total_violations: int = Field(
        default=0,
        ge=0,
        description="Total count of violations",
    )
    blocking_count: int = Field(
        default=0,
        ge=0,
        description="Count of blocking (error severity) violations",
    )
    imperative_nodes: list[str] = Field(
        default_factory=list,
        description="List of node class names that are imperative (have violations)",
    )

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
    )

    def __bool__(self) -> bool:
        """Allow boolean context: if result: ...

        Warning:
            Non-standard __bool__: Returns True only when passed is True.
            This differs from Pydantic default where bool(model) always returns True.
        """
        return self.passed

    @classmethod
    def from_violations(
        cls,
        violations: list[ModelDeclarativeNodeViolation],
        files_checked: int = 0,
    ) -> ModelDeclarativeNodeValidationResult:
        """Create a result from a list of violations.

        Args:
            violations: List of detected violations.
            files_checked: Number of node.py files that were checked.

        Returns:
            A ModelDeclarativeNodeValidationResult instance.
        """
        blocking_count = sum(1 for v in violations if v.is_blocking())
        imperative_nodes = sorted(
            {v.node_class_name for v in violations if v.node_class_name}
        )
        return cls(
            passed=blocking_count == 0,
            violations=violations,
            files_checked=files_checked,
            total_violations=len(violations),
            blocking_count=blocking_count,
            imperative_nodes=imperative_nodes,
        )

    def format_for_ci(self) -> list[str]:
        """Format all violations for CI output.

        Returns:
            List of formatted strings for CI annotation.
        """
        return [v.format_for_ci() for v in self.violations]

    def format_summary(self) -> str:
        """Format a summary for console output.

        Returns:
            Summary string with pass/fail status and counts.
        """
        status = "PASSED" if self.passed else "FAILED"
        summary = (
            f"Declarative Node Validation: {status}\n"
            f"  Files checked: {self.files_checked}\n"
            f"  Total violations: {self.total_violations}\n"
            f"  Blocking violations: {self.blocking_count}"
        )
        if self.imperative_nodes:
            summary += f"\n  Imperative nodes: {', '.join(self.imperative_nodes)}"
        return summary


__all__ = ["ModelDeclarativeNodeValidationResult"]
