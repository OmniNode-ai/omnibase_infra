# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Architecture validation result model for the architecture validator node.

This module defines the output model for architecture validation results.
It contains the overall validation status, any violations found, and
metadata about the validation run.

Design Pattern:
    ModelArchitectureValidationResult overrides ``__bool__`` to return
    ``True`` only when validation passed (no violations). This enables
    idiomatic conditional checks::

        result = validator.validate(request)
        if result:
            print("Validation passed!")
        else:
            print(f"Found {len(result.violations)} violations")

Example:
    >>> from omnibase_infra.nodes.architecture_validator.models import (
    ...     ModelArchitectureValidationResult,
    ...     ModelArchitectureViolation,
    ...     EnumValidationSeverity,
    ... )
    >>> # Passing result
    >>> result = ModelArchitectureValidationResult(
    ...     valid=True,
    ...     violations=[],
    ...     files_checked=42,
    ...     rules_checked=["ARCH-001", "ARCH-002"],
    ... )
    >>> bool(result)
    True
    >>>
    >>> # Failing result
    >>> violation = ModelArchitectureViolation(
    ...     rule_id="ARCH-001",
    ...     rule_name="No Any Types",
    ...     severity=EnumValidationSeverity.ERROR,
    ...     target_type="model",
    ...     target_name="BadModel",
    ...     message="Found Any type",
    ... )
    >>> result = ModelArchitectureValidationResult(
    ...     valid=False,
    ...     violations=[violation],
    ...     files_checked=42,
    ...     rules_checked=["ARCH-001"],
    ... )
    >>> bool(result)
    False
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.architecture_validator.models.model_architecture_violation import (
    ModelArchitectureViolation,
)


class ModelArchitectureValidationResult(BaseModel):
    """Result of architecture validation containing status and violations.

    This model represents the complete result of a validation run,
    including whether validation passed, any violations found,
    and metadata about what was checked.

    Attributes:
        valid: True if no violations were found, False otherwise.
        violations: List of architecture violations detected.
        files_checked: Number of files that were validated.
        rules_checked: List of rule IDs that were applied.

    Warning:
        **Non-standard __bool__ behavior**: This model overrides ``__bool__``
        to return ``True`` only when validation passed (``valid`` is True).
        This differs from typical Pydantic model behavior where ``bool(model)``
        always returns ``True`` for any valid model instance.

        This design enables idiomatic conditional checks::

            if result:
                # Validation passed - no violations
                deploy()
            else:
                # Validation failed - has violations
                print(result.violations)

        If you need to check model validity instead, use explicit attribute access::

            # Check validation passed (uses __bool__)
            if result:
                ...

            # Check model is valid (always True for constructed instance)
            if result is not None:
                ...

            # Explicit validation check (preferred for clarity)
            if result.valid:
                ...

    Example:
        >>> # Create a passing result
        >>> result = ModelArchitectureValidationResult(
        ...     valid=True,
        ...     files_checked=100,
        ...     rules_checked=["ARCH-001", "ARCH-002", "ARCH-003"],
        ... )
        >>> result.valid
        True
        >>> bool(result)
        True
        >>>
        >>> # Create a failing result
        >>> result = ModelArchitectureValidationResult(
        ...     valid=False,
        ...     violations=[violation],
        ...     files_checked=100,
        ...     rules_checked=["ARCH-001"],
        ... )
        >>> result.valid
        False
        >>> bool(result)
        False
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
    )

    valid: bool = Field(
        ...,
        description="True if no violations were found, False otherwise.",
    )
    violations: list[ModelArchitectureViolation] = Field(
        default_factory=list,
        description="List of architecture violations detected during validation.",
    )
    files_checked: int = Field(
        default=0,
        description="Number of files that were validated.",
        ge=0,
    )
    rules_checked: list[str] = Field(
        default_factory=list,
        description="List of rule IDs that were applied during validation.",
    )

    @property
    def violation_count(self) -> int:
        """Get the number of violations found.

        Returns:
            Number of violations in the violations list.
        """
        return len(self.violations)

    @property
    def error_count(self) -> int:
        """Get the number of ERROR severity violations.

        Returns:
            Count of violations with ERROR severity.
        """
        from omnibase_infra.nodes.architecture_validator.enums import (
            EnumValidationSeverity,
        )

        return sum(
            1 for v in self.violations if v.severity == EnumValidationSeverity.ERROR
        )

    @property
    def warning_count(self) -> int:
        """Get the number of WARNING severity violations.

        Returns:
            Count of violations with WARNING severity.
        """
        from omnibase_infra.nodes.architecture_validator.enums import (
            EnumValidationSeverity,
        )

        return sum(
            1 for v in self.violations if v.severity == EnumValidationSeverity.WARNING
        )

    def __bool__(self) -> bool:
        """Allow using result in boolean context to check validation status.

        Returns True if validation passed (no violations found),
        indicating the code meets architecture requirements.

        Warning:
            This differs from typical Pydantic model behavior where ``bool(model)``
            always returns ``True`` for any valid model instance. Here, ``bool(result)``
            returns ``False`` for valid results that have violations.

            Use ``result.valid`` for explicit, self-documenting code.
            Use ``result is not None`` if you need to check model existence.

        Returns:
            True if valid is True, False otherwise.

        Example:
            >>> result_pass = ModelArchitectureValidationResult(valid=True)
            >>> bool(result_pass)
            True
            >>>
            >>> result_fail = ModelArchitectureValidationResult(
            ...     valid=False,
            ...     violations=[some_violation],
            ... )
            >>> bool(result_fail)  # False even though model is valid!
            False
        """
        return self.valid

    def __str__(self) -> str:
        """Return a human-readable string representation for debugging.

        Returns:
            String showing validation status, file count, and violation count.
        """
        status = "PASSED" if self.valid else "FAILED"
        return (
            f"ModelArchitectureValidationResult("
            f"status={status}, "
            f"files={self.files_checked}, "
            f"violations={self.violation_count})"
        )

    @classmethod
    def passed(
        cls,
        files_checked: int = 0,
        rules_checked: list[str] | None = None,
    ) -> ModelArchitectureValidationResult:
        """Create a passing validation result.

        Factory method for creating a result with no violations.

        Args:
            files_checked: Number of files that were validated.
            rules_checked: List of rule IDs that were applied.

        Returns:
            ModelArchitectureValidationResult with valid=True and no violations.

        Example:
            >>> result = ModelArchitectureValidationResult.passed(
            ...     files_checked=50,
            ...     rules_checked=["ARCH-001", "ARCH-002"],
            ... )
            >>> result.valid
            True
        """
        return cls(
            valid=True,
            violations=[],
            files_checked=files_checked,
            rules_checked=rules_checked or [],
        )

    @classmethod
    def failed(
        cls,
        violations: list[ModelArchitectureViolation],
        files_checked: int = 0,
        rules_checked: list[str] | None = None,
    ) -> ModelArchitectureValidationResult:
        """Create a failing validation result.

        Factory method for creating a result with violations.

        Args:
            violations: List of architecture violations found.
            files_checked: Number of files that were validated.
            rules_checked: List of rule IDs that were applied.

        Returns:
            ModelArchitectureValidationResult with valid=False and violations.

        Example:
            >>> result = ModelArchitectureValidationResult.failed(
            ...     violations=[violation1, violation2],
            ...     files_checked=50,
            ...     rules_checked=["ARCH-001"],
            ... )
            >>> result.valid
            False
            >>> result.violation_count
            2
        """
        return cls(
            valid=False,
            violations=violations,
            files_checked=files_checked,
            rules_checked=rules_checked or [],
        )


__all__ = ["ModelArchitectureValidationResult"]
