# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Execution Shape Validation Result Model.

Defines the aggregate result structure for execution shape validation,
replacing the tuple pattern `tuple[bool, list[ModelExecutionShapeViolationResult]]`
used by `validate_execution_shapes_ci`. This model provides richer context
for CI gate integration and violation aggregation.

Note:
    This model aggregates multiple violations and determines pass/fail status
    based on whether any blocking (error-severity) violations exist.

.. versionadded:: 0.6.0
    Created as part of Union Reduction Phase 3 (OMN-1003).
"""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.models.validation.model_execution_shape_violation import (
    ModelExecutionShapeViolationResult,
)


class ModelExecutionShapeValidationResult(BaseModel):
    """Aggregate result of execution shape validation.

    Contains the overall pass/fail status and all detected violations.
    This model replaces the `tuple[bool, list[ModelExecutionShapeViolationResult]]`
    pattern used by `validate_execution_shapes_ci`, providing a more expressive
    API with factory methods and computed properties.

    Attributes:
        passed: Whether validation passed (no blocking violations).
        violations: All violations detected during validation.

    Example:
        >>> # Create from violations list
        >>> result = ModelExecutionShapeValidationResult.from_violations(violations)
        >>> if not result.passed:
        ...     print(f"Found {result.blocking_count} blocking violations")
        ...     for line in result.format_for_ci():
        ...         print(line)

        >>> # Create successful result
        >>> result = ModelExecutionShapeValidationResult.success()
        >>> result.passed
        True

    .. versionadded:: 0.6.0
    """

    passed: bool = Field(
        description="Whether validation passed (no blocking violations)",
    )
    violations: list[ModelExecutionShapeViolationResult] = Field(
        default_factory=list,
        description="All violations found during validation",
    )

    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
        str_strip_whitespace=True,
    )

    @property
    def has_blocking(self) -> bool:
        """Check if any blocking violations exist.

        Returns:
            True if any violation has severity 'error'.

        Example:
            >>> result = ModelExecutionShapeValidationResult.success()
            >>> result.has_blocking
            False

        .. versionadded:: 0.6.0
        """
        return any(v.is_blocking() for v in self.violations)

    @property
    def violation_count(self) -> int:
        """Total number of violations.

        Returns:
            Count of all violations (both blocking and warnings).

        .. versionadded:: 0.6.0
        """
        return len(self.violations)

    @property
    def blocking_count(self) -> int:
        """Number of blocking (error-severity) violations.

        Returns:
            Count of violations that would block CI.

        .. versionadded:: 0.6.0
        """
        return sum(1 for v in self.violations if v.is_blocking())

    @property
    def warning_count(self) -> int:
        """Number of warning-severity violations.

        Returns:
            Count of non-blocking violations.

        .. versionadded:: 0.6.0
        """
        return sum(1 for v in self.violations if not v.is_blocking())

    @classmethod
    def from_violations(
        cls, violations: list[ModelExecutionShapeViolationResult]
    ) -> Self:
        """Create from violations list, auto-determining pass/fail.

        The result passes if there are no blocking (error-severity) violations.
        Warning-severity violations do not cause failure.

        Args:
            violations: List of all detected violations.

        Returns:
            ModelExecutionShapeValidationResult with auto-determined pass/fail.

        Example:
            >>> violations = [
            ...     ModelExecutionShapeViolationResult(
            ...         violation_type=EnumExecutionShapeViolation.REDUCER_RETURNS_EVENTS,
            ...         file_path="/src/handler.py",
            ...         line_number=42,
            ...         message="Reducer returns EVENT",
            ...         severity="error",
            ...     )
            ... ]
            >>> result = ModelExecutionShapeValidationResult.from_violations(violations)
            >>> result.passed
            False

        .. versionadded:: 0.6.0
        """
        has_blocking = any(v.is_blocking() for v in violations)
        return cls(passed=not has_blocking, violations=violations)

    @classmethod
    def success(cls) -> Self:
        """Create a successful validation result with no violations.

        Returns:
            ModelExecutionShapeValidationResult with passed=True and empty violations.

        Example:
            >>> result = ModelExecutionShapeValidationResult.success()
            >>> result.passed
            True
            >>> result.violation_count
            0

        .. versionadded:: 0.6.0
        """
        return cls(passed=True, violations=[])

    @classmethod
    def from_legacy_result(
        cls, result: tuple[bool, list[ModelExecutionShapeViolationResult]]
    ) -> Self:
        """Create from legacy tuple-based validation result.

        This factory method handles conversion from the old tuple pattern
        to the new model structure for gradual migration.

        Args:
            result: Legacy result as (passed, violations).

        Returns:
            ModelExecutionShapeValidationResult with equivalent values.

        Example:
            >>> legacy = (False, [violation])
            >>> result = ModelExecutionShapeValidationResult.from_legacy_result(legacy)
            >>> result.passed
            False

        .. versionadded:: 0.6.0
        """
        passed, violations = result
        return cls(passed=passed, violations=violations)

    def to_legacy_result(self) -> tuple[bool, list[ModelExecutionShapeViolationResult]]:
        """Convert back to legacy tuple format.

        This method enables gradual migration by allowing conversion back
        to the original format where needed.

        Returns:
            Tuple of (passed, violations).

        Example:
            >>> result = ModelExecutionShapeValidationResult.success()
            >>> result.to_legacy_result()
            (True, [])

        .. versionadded:: 0.6.0
        """
        return (self.passed, list(self.violations))

    def format_for_ci(self) -> list[str]:
        """Format all violations for CI output.

        Returns:
            List of formatted violation strings in GitHub Actions format.

        Example:
            >>> for line in result.format_for_ci():
            ...     print(line)

        .. versionadded:: 0.6.0
        """
        return [v.format_for_ci() for v in self.violations]

    def __bool__(self) -> bool:
        """Allow using result in boolean context.

        Warning:
            **Non-standard __bool__ behavior**: This model overrides ``__bool__`` to
            return ``True`` only when ``passed`` is True (no blocking violations).
            This differs from typical Pydantic model behavior where ``bool(model)``
            always returns ``True`` for any valid model instance.

            This design enables idiomatic CI gate checks::

                if result:
                    # Validation passed - can proceed
                    deploy()
                else:
                    # Has blocking violations - fail CI
                    for line in result.format_for_ci():
                        print(line)
                    sys.exit(1)

            If you need to check model validity instead, use explicit attribute access::

                # Check for pass (uses __bool__)
                if result:
                    ...

                # Check model is valid (always True for constructed instance)
                if result is not None:
                    ...

                # Explicit pass check (preferred for clarity)
                if result.passed:
                    ...

        Returns:
            True if validation passed, False otherwise.

        Example:
            >>> if ModelExecutionShapeValidationResult.success():
            ...     print("All clear!")
            All clear!

        .. versionadded:: 0.6.0
        """
        return self.passed

    def __str__(self) -> str:
        """Return a human-readable string representation for debugging.

        Returns:
            String format showing pass/fail status and violation counts.

        .. versionadded:: 0.6.0
        """
        status = "passed" if self.passed else "failed"
        return (
            f"ModelExecutionShapeValidationResult("
            f"{status}, "
            f"violations={self.violation_count}, "
            f"blocking={self.blocking_count})"
        )


__all__ = ["ModelExecutionShapeValidationResult"]
