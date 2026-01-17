# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Chain Propagation Error.

Defines the error class for correlation and causation chain validation failures.
Used when message chains fail to maintain proper correlation_id or causation_id
propagation, which breaks distributed tracing capabilities.
"""

from __future__ import annotations

from uuid import uuid4

from omnibase_core.enums import EnumCoreErrorCode
from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors.error_infra import (
    ProtocolConfigurationError,
    RuntimeHostError,
)
from omnibase_infra.models.errors.model_infra_error_context import (
    ModelInfraErrorContext,
)
from omnibase_infra.models.validation.model_chain_violation import ModelChainViolation


class ChainPropagationError(RuntimeHostError):
    """Raised when correlation or causation chain validation fails.

    Used when messages fail to properly maintain their correlation and
    causation chains during propagation. This error indicates a break
    in the distributed tracing chain that will impact observability
    and debugging capabilities.

    Common scenarios:
        - Child message has different correlation_id than parent
        - Message's causation_id doesn't reference parent's message_id
        - Causation chain skips ancestors in the message lineage

    The error includes all detected violations for comprehensive reporting
    and supports correlation_id from the error context for meta-level tracing.

    Attributes:
        violations: Immutable tuple of all chain violations detected.

    Example:
        >>> from uuid import uuid4
        >>> from omnibase_infra.enums import EnumChainViolationType
        >>> from omnibase_infra.models.validation import ModelChainViolation
        >>>
        >>> violations = [
        ...     ModelChainViolation(
        ...         violation_type=EnumChainViolationType.CORRELATION_MISMATCH,
        ...         expected_value=uuid4(),
        ...         actual_value=uuid4(),
        ...         message_id=uuid4(),
        ...         violation_message="correlation_id mismatch",
        ...     ),
        ... ]
        >>> context = ModelInfraErrorContext(
        ...     operation="validate_chain",
        ...     correlation_id=uuid4(),
        ... )
        >>> raise ChainPropagationError(
        ...     "Chain validation failed",
        ...     violations=violations,
        ...     context=context,
        ... )
    """

    def __init__(
        self,
        message: str,
        violations: list[ModelChainViolation],
        context: ModelInfraErrorContext | None = None,
        **extra_context: object,
    ) -> None:
        """Initialize ChainPropagationError with violation details.

        Args:
            message: Human-readable error message summarizing the violations.
            violations: List of all chain violations detected. Must contain
                at least one violation.
            context: Bundled infrastructure context including correlation_id
                for meta-level tracing of the validation operation itself.
            **extra_context: Additional context information for debugging.

        Raises:
            ProtocolConfigurationError: If violations list is empty.
        """
        if not violations:
            err_context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="ChainPropagationError.__init__",
                correlation_id=context.correlation_id if context else uuid4(),
            )
            raise ProtocolConfigurationError(
                "ChainPropagationError requires at least one violation",
                context=err_context,
                parameter="violations",
            )

        # Store violations for access
        self._violations = violations

        # Build summary for error message
        error_count = sum(1 for v in violations if v.severity == "error")
        warning_count = sum(1 for v in violations if v.severity == "warning")
        violation_summary = f"{error_count} error(s), {warning_count} warning(s)"

        # Build detailed message
        full_message = f"{message}: {violation_summary}"

        # Add violation types summary
        violation_types = {v.violation_type.value for v in violations}
        full_message += f" [{', '.join(sorted(violation_types))}]"

        super().__init__(
            message=full_message,
            error_code=EnumCoreErrorCode.VALIDATION_ERROR,
            context=context,
            violation_count=len(violations),
            error_count=error_count,
            warning_count=warning_count,
            violation_types=list(violation_types),
            **extra_context,
        )

    @property
    def violations(self) -> tuple[ModelChainViolation, ...]:
        """Get all detected chain violations.

        Returns:
            Immutable tuple of ModelChainViolation instances describing each violation.
        """
        return tuple(self._violations)

    def get_blocking_violations(self) -> list[ModelChainViolation]:
        """Get only violations that should block processing.

        Returns:
            List of violations with severity='error'.

        Example:
            >>> blocking = error.get_blocking_violations()
            >>> len(blocking)
            1
        """
        return [v for v in self._violations if v.is_blocking()]

    def get_warnings(self) -> list[ModelChainViolation]:
        """Get only advisory violations.

        Returns:
            List of violations with severity='warning'.

        Example:
            >>> warnings = error.get_warnings()
            >>> len(warnings)
            2
        """
        return [v for v in self._violations if not v.is_blocking()]

    def format_violations_for_logging(self) -> str:
        """Format all violations for structured logging output.

        Returns:
            Multi-line string with each violation formatted for logging.

        Example:
            >>> error.format_violations_for_logging()
            '[error] CORRELATION_MISMATCH on message=abc...\\n[warning] ...'
        """
        return "\n".join(v.format_for_logging() for v in self._violations)

    def has_blocking_violations(self) -> bool:
        """Check if any violations should block processing.

        Returns:
            True if at least one violation has severity='error'.

        Example:
            >>> error.has_blocking_violations()
            True
        """
        return any(v.is_blocking() for v in self._violations)


__all__ = ["ChainPropagationError"]
