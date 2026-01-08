# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Architecture validation result model.

This module defines the output model for NodeArchitectureValidatorCompute,
containing all validation results including violations found and summary statistics.

Design Pattern:
    ModelArchitectureValidationResult serves as the complete output from architecture
    validation operations. It aggregates all violations found during validation and
    provides computed properties for quick status checks.

    The model uses immutable tuple fields to ensure thread safety and prevent
    accidental modification of validation results after creation.

Thread Safety:
    ModelArchitectureValidationResult is immutable (frozen=True) after creation,
    making it thread-safe for concurrent read access. All collection fields use
    immutable tuple types.

Example:
    >>> from omnibase_infra.nodes.architecture_validator.models import (
    ...     ModelArchitectureValidationResult,
    ... )
    >>>
    >>> # Create a passing result
    >>> result = ModelArchitectureValidationResult(
    ...     rules_checked=("ARCH-001", "ARCH-002"),
    ...     nodes_checked=5,
    ...     handlers_checked=10,
    ... )
    >>> result.valid
    True
    >>> bool(result)  # Uses custom __bool__
    True
    >>>
    >>> # Create a result with violations
    >>> result_with_violations = ModelArchitectureValidationResult(
    ...     violations=(violation1, violation2),
    ...     rules_checked=("ARCH-001",),
    ...     nodes_checked=5,
    ... )
    >>> result_with_violations.valid
    False
    >>> result_with_violations.violation_count
    2

.. versionadded:: 0.8.0
    Created as part of OMN-1138 Architecture Validator implementation.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.architecture_validator.models.model_architecture_violation import (
    ModelArchitectureViolation,
)


class ModelArchitectureValidationResult(BaseModel):
    """Result of architecture validation.

    Contains all violations found during validation and summary statistics
    about what was checked. Used as the output from NodeArchitectureValidatorCompute.

    Attributes:
        violations: All violations found during validation. Empty tuple if
            validation passed.
        rules_checked: Rule IDs that were evaluated during validation.
        nodes_checked: Number of nodes that were validated.
        handlers_checked: Number of handlers that were validated.
        correlation_id: Optional correlation ID for distributed tracing.

    Warning:
        **Non-standard __bool__ behavior**: This model overrides ``__bool__`` to return
        ``True`` only when validation passed (i.e., no violations found). This
        differs from typical Pydantic model behavior where ``bool(model)`` always
        returns ``True`` for any valid model instance.

        This design enables idiomatic conditional checks for validation status::

            if result:
                # Validation passed - proceed with deployment
                deploy_nodes()
            else:
                # Validation failed - handle violations
                report_violations(result.violations)

        If you need to check model validity instead, use explicit attribute access::

            # Check for validation success (uses __bool__)
            if result:
                ...

            # Check model is valid (always True for constructed instance)
            if result is not None:
                ...

            # Explicit validation check (preferred for clarity)
            if result.valid:
                ...

    Example:
        >>> # Create a clean validation result
        >>> result = ModelArchitectureValidationResult(
        ...     rules_checked=("ARCH-001", "ARCH-002", "ARCH-003"),
        ...     nodes_checked=10,
        ...     handlers_checked=25,
        ... )
        >>> result.valid
        True
        >>> result.violation_count
        0

        >>> # Create result with violations
        >>> result = ModelArchitectureValidationResult(
        ...     violations=(violation,),
        ...     rules_checked=("ARCH-001",),
        ...     nodes_checked=5,
        ... )
        >>> result.valid
        False

    .. versionadded:: 0.8.0
        Created as part of OMN-1138 Architecture Validator implementation.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
    )

    violations: tuple[ModelArchitectureViolation, ...] = Field(
        default_factory=tuple,
        description="All violations found during validation. Empty if validation passed.",
    )

    rules_checked: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Rule IDs that were checked during validation (e.g., 'ARCH-001').",
    )

    nodes_checked: int = Field(
        default=0,
        ge=0,
        description="Number of nodes that were validated.",
    )

    handlers_checked: int = Field(
        default=0,
        ge=0,
        description="Number of handlers that were validated.",
    )

    correlation_id: str | None = Field(
        default=None,
        description="Correlation ID for distributed tracing.",
    )

    @property
    def valid(self) -> bool:
        """Whether validation passed (no violations found).

        Returns:
            True if no violations were found, False otherwise.

        Example:
            >>> result = ModelArchitectureValidationResult(violations=())
            >>> result.valid
            True

        .. versionadded:: 0.8.0
        """
        return len(self.violations) == 0

    @property
    def violation_count(self) -> int:
        """Total number of violations found.

        Returns:
            Number of violations in the violations tuple.

        Example:
            >>> result = ModelArchitectureValidationResult(
            ...     violations=(v1, v2, v3),
            ... )
            >>> result.violation_count
            3

        .. versionadded:: 0.8.0
        """
        return len(self.violations)

    @property
    def rules_checked_count(self) -> int:
        """Total number of rules that were evaluated.

        Returns:
            Number of rules in the rules_checked tuple.

        .. versionadded:: 0.8.0
        """
        return len(self.rules_checked)

    def __bool__(self) -> bool:
        """Allow using result in boolean context to check validation status.

        Returns True if validation passed (no violations), enabling idiomatic
        conditional checks for validation success.

        Warning:
            This differs from typical Pydantic model behavior where ``bool(model)``
            always returns ``True`` for any valid model instance. Here, ``bool(result)``
            returns ``False`` for valid results that contain violations.

            Use ``result.valid`` for explicit, self-documenting code.
            Use ``result is not None`` if you need to check model existence.

        Returns:
            True if no violations found (validation passed), False otherwise.

        Example:
            >>> # Passing validation
            >>> result_pass = ModelArchitectureValidationResult(
            ...     rules_checked=("ARCH-001",),
            ...     nodes_checked=5,
            ... )
            >>> bool(result_pass)
            True

            >>> # Failing validation
            >>> result_fail = ModelArchitectureValidationResult(
            ...     violations=(some_violation,),
            ...     rules_checked=("ARCH-001",),
            ... )
            >>> bool(result_fail)  # False even though model is valid!
            False

            >>> # Idiomatic usage
            >>> if result_pass:
            ...     print("Validation passed")
            ... else:
            ...     print("Validation failed")
            Validation passed

        .. versionadded:: 0.8.0
        """
        return self.valid

    def __str__(self) -> str:
        """Return a human-readable string representation for debugging.

        Returns:
            String showing validation status and summary counts.

        .. versionadded:: 0.8.0
        """
        status = "PASSED" if self.valid else "FAILED"
        return (
            f"ModelArchitectureValidationResult("
            f"status={status}, "
            f"violations={self.violation_count}, "
            f"rules={self.rules_checked_count}, "
            f"nodes={self.nodes_checked}, "
            f"handlers={self.handlers_checked})"
        )

    @classmethod
    def passed(
        cls,
        *,
        rules_checked: tuple[str, ...] = (),
        nodes_checked: int = 0,
        handlers_checked: int = 0,
        correlation_id: str | None = None,
    ) -> ModelArchitectureValidationResult:
        """Create a passing validation result with no violations.

        Factory method for creating clean validation results.

        Args:
            rules_checked: Rule IDs that were evaluated.
            nodes_checked: Number of nodes validated.
            handlers_checked: Number of handlers validated.
            correlation_id: Optional correlation ID for tracing.

        Returns:
            ModelArchitectureValidationResult with no violations.

        Example:
            >>> result = ModelArchitectureValidationResult.passed(
            ...     rules_checked=("ARCH-001", "ARCH-002"),
            ...     nodes_checked=10,
            ... )
            >>> result.valid
            True

        .. versionadded:: 0.8.0
        """
        return cls(
            violations=(),
            rules_checked=rules_checked,
            nodes_checked=nodes_checked,
            handlers_checked=handlers_checked,
            correlation_id=correlation_id,
        )


__all__ = ["ModelArchitectureValidationResult"]
