# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Any Type Validation Result Model.

Defines the aggregate result structure for Any type validation operations.
Used by the Any type validator to provide a structured result for CI pipeline
integration with convenience methods for reporting.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.models.validation.model_any_type_violation import (
    ModelAnyTypeViolation,
)


class ModelAnyTypeValidationResult(BaseModel):
    """Aggregate result of Any type validation.

    Provides a structured result for CI pipeline integration with
    convenience methods for reporting.

    Attributes:
        passed: True if no blocking violations found.
        violations: List of all detected violations.
        files_checked: Number of files that were validated.
        total_violations: Total count of violations.
        blocking_count: Count of blocking (error severity) violations.

    Example:
        >>> result = ModelAnyTypeValidationResult.from_violations(
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
    violations: list[ModelAnyTypeViolation] = Field(
        default_factory=list,
        description="List of all detected violations",
    )
    files_checked: int = Field(
        default=0,
        ge=0,
        description="Number of files that were validated",
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

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
    )

    @classmethod
    def from_violations(
        cls,
        violations: list[ModelAnyTypeViolation],
        files_checked: int = 0,
    ) -> ModelAnyTypeValidationResult:
        """Create a result from a list of violations.

        Args:
            violations: List of detected violations.
            files_checked: Number of files that were checked.

        Returns:
            A ModelAnyTypeValidationResult instance.
        """
        blocking_count = sum(1 for v in violations if v.is_blocking())
        return cls(
            passed=blocking_count == 0,
            violations=violations,
            files_checked=files_checked,
            total_violations=len(violations),
            blocking_count=blocking_count,
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
        return (
            f"Any Type Validation: {status}\n"
            f"  Files checked: {self.files_checked}\n"
            f"  Total violations: {self.total_violations}\n"
            f"  Blocking violations: {self.blocking_count}"
        )


__all__ = ["ModelAnyTypeValidationResult"]
