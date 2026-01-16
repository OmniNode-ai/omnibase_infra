# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""LocalHandler Validation Result Model.

Defines the aggregate result structure for LocalHandler validation operations.
Used by the LocalHandler validator to provide a structured result for CI pipeline
integration with convenience methods for reporting.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.models.validation.model_localhandler_violation import (
    ModelLocalHandlerViolation,
)


class ModelLocalHandlerValidationResult(BaseModel):
    """Aggregate result of LocalHandler validation.

    Provides a structured result for CI pipeline integration with
    convenience methods for reporting.

    Attributes:
        passed: True if no violations found.
        violations: List of all detected violations.
        files_checked: Number of files that were validated.

    Custom Boolean Behavior:
        This model implements ``__bool__`` to return ``passed``.
        This enables idiomatic usage like ``if result: ...``.

        Warning:
            This differs from standard Pydantic behavior where
            ``bool(model)`` always returns True for non-empty models.

    Example:
        >>> result = ModelLocalHandlerValidationResult.from_violations(
        ...     violations=[violation1],
        ...     files_checked=10,
        ... )
        >>> if not result:  # Uses __bool__, checks passed
        ...     print(result.format_summary())
    """

    passed: bool = Field(
        ...,
        description="True if no LocalHandler violations found",
    )
    violations: list[ModelLocalHandlerViolation] = Field(
        default_factory=list,
        description="List of all detected violations",
    )
    files_checked: int = Field(
        default=0,
        ge=0,
        description="Number of files that were validated",
    )

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
    )

    def __bool__(self) -> bool:
        """Return True if validation passed (no violations).

        Warning:
            This differs from standard Pydantic behavior where
            ``bool(model)`` always returns True for non-empty models.
            Here, ``bool(result)`` returns ``result.passed``.
        """
        return self.passed

    @classmethod
    def from_violations(
        cls,
        violations: list[ModelLocalHandlerViolation],
        files_checked: int = 0,
    ) -> ModelLocalHandlerValidationResult:
        """Create a result from a list of violations.

        Args:
            violations: List of detected violations.
            files_checked: Number of files that were checked.

        Returns:
            A ModelLocalHandlerValidationResult instance.
        """
        return cls(
            passed=len(violations) == 0,
            violations=violations,
            files_checked=files_checked,
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
            Summary string with pass/fail status and violation count.

        Example:
            LocalHandler: FAIL
              Blocked imports found in src/omnibase_infra/:
              - src/.../bad_handler.py:5: from omnibase_core.handlers import LocalHandler

              LocalHandler is for test-only use. Remove from production code.
        """
        if self.passed:
            return (
                f"LocalHandler: PASS\n"
                f"  Files checked: {self.files_checked}\n"
                f"  No LocalHandler imports found in production code."
            )

        lines = [
            "LocalHandler: FAIL",
            "  Blocked imports found in src/omnibase_infra/:",
        ]
        for v in self.violations:
            lines.append(f"  - {v.file_path}:{v.line_number}: {v.import_line}")
        lines.append("")
        lines.append(
            "  LocalHandler is for test-only use. Remove from production code."
        )
        return "\n".join(lines)


__all__ = ["ModelLocalHandlerValidationResult"]
