# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Architecture Violation Error.

This module defines the error raised when architecture validation fails during
runtime startup. The error contains detailed information about all blocking
violations that prevented startup.

Related:
    - OMN-1138: Architecture Validator for omnibase_infra
    - RuntimeHostProcess: Validates architecture before starting

Example:
    >>> from omnibase_infra.errors import ArchitectureViolationError
    >>> from omnibase_infra.nodes.architecture_validator import ModelArchitectureViolation
    >>>
    >>> violations = (
    ...     ModelArchitectureViolation(
    ...         rule_id="NO_HANDLER_PUBLISHING",
    ...         rule_name="No Handler Publishing",
    ...         severity=EnumValidationSeverity.ERROR,
    ...         target_type="handler",
    ...         target_name="MyHandler",
    ...         message="Handler must not have direct event bus access",
    ...     ),
    ... )
    >>> raise ArchitectureViolationError(
    ...     message=f"Architecture validation failed with {len(violations)} blocking violations",
    ...     violations=violations,
    ... )
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from omnibase_core.enums.enum_core_error_code import EnumCoreErrorCode
from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors.error_infra import RuntimeHostError
from omnibase_infra.models.errors.model_infra_error_context import (
    ModelInfraErrorContext,
)

if TYPE_CHECKING:
    from omnibase_infra.nodes.architecture_validator import ModelArchitectureViolation


class ArchitectureViolationError(RuntimeHostError):
    """Error raised when architecture validation fails during startup.

    This error is raised by RuntimeHostProcess when architecture validation
    detects blocking violations (ERROR severity) that should prevent runtime
    startup. The error contains all blocking violations for inspection.

    Use this error to:
    - Block runtime startup when architecture rules are violated
    - Report detailed violation information for debugging
    - Provide context for CI/CD pipeline failures

    Attributes:
        violations: Tuple of architecture violations that caused the failure.
            Only violations with ERROR severity should be included.

    Example:
        >>> # In RuntimeHostProcess._validate_architecture:
        >>> blocking_violations = tuple(
        ...     v for v in result.violations if v.blocks_startup()
        ... )
        >>> if blocking_violations:
        ...     raise ArchitectureViolationError(
        ...         message=f"Architecture validation failed with {len(blocking_violations)} blocking violations",
        ...         violations=blocking_violations,
        ...     )

        >>> # Catching and inspecting the error:
        >>> try:
        ...     await runtime.start()
        ... except ArchitectureViolationError as e:
        ...     for v in e.violations:
        ...         print(v.format_for_logging())
        ...     sys.exit(1)
    """

    def __init__(
        self,
        message: str,
        violations: tuple[ModelArchitectureViolation, ...],
        context: ModelInfraErrorContext | None = None,
        correlation_id: UUID | None = None,
        **extra_context: object,
    ) -> None:
        """Initialize ArchitectureViolationError.

        Args:
            message: Human-readable error message describing the validation failure.
            violations: Tuple of architecture violations that caused the failure.
                Should only include violations with ERROR severity (blocking).
            context: Optional infrastructure error context. If not provided,
                a default context for architecture validation is created.
            correlation_id: Optional correlation ID for distributed tracing.
                Used when building the default context if context is None.
                Auto-generated using uuid4() if not provided.
            **extra_context: Additional context information for debugging.

        Example:
            >>> raise ArchitectureViolationError(
            ...     message="Architecture validation failed with 3 blocking violations",
            ...     violations=blocking_violations,
            ...     correlation_id=request_correlation_id,
            ... )
        """
        self.violations = violations

        # Auto-generate correlation_id if not provided (per ONEX standards)
        if correlation_id is None:
            correlation_id = uuid4()

        # Build default context if not provided
        if context is None:
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="validate_architecture",
                correlation_id=correlation_id,
            )

        # Add violation summary to extra context
        extra_context["violation_count"] = len(violations)
        extra_context["violation_rule_ids"] = tuple(v.rule_id for v in violations)

        super().__init__(
            message=message,
            error_code=EnumCoreErrorCode.CONTRACT_VIOLATION,
            context=context,
            **extra_context,
        )

    def format_violations(self) -> str:
        """Format all violations for display.

        Returns:
            Multi-line string with each violation formatted for logging.

        Example:
            >>> print(error.format_violations())
            [ERROR] NO_HANDLER_PUBLISHING (No Handler Publishing) on handler/MyHandler: ...
            [ERROR] NO_ANY_TYPES (No Any Types) on model/MyModel: ...
        """
        return "\n".join(v.format_for_logging() for v in self.violations)


__all__ = ["ArchitectureViolationError"]
