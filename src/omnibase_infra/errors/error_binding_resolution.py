# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Binding Resolution Error for declarative operation bindings.

This module defines the error raised when binding resolution fails during
declarative operation processing. Bindings use ${source.path} expressions
to extract values from envelopes, payloads, or context.
"""

from uuid import UUID

from omnibase_core.enums import EnumCoreErrorCode
from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors.error_infra import RuntimeHostError
from omnibase_infra.models.errors.model_infra_error_context import (
    ModelInfraErrorContext,
)


class BindingResolutionError(RuntimeHostError):
    """Error when binding resolution fails.

    Raised when a required binding cannot be resolved from the envelope,
    payload, or context. Includes full diagnostic context for debugging.

    This error is typically raised during declarative operation processing
    when a ${source.path} expression cannot be resolved because:
    - The source (payload, envelope, context) is missing
    - A path segment does not exist in the source
    - The value is required but None

    Attributes:
        operation_name: The operation being resolved (e.g., "db.query")
        parameter_name: The parameter that failed (e.g., "sql")
        expression: The ${source.path} expression that failed (e.g., "${payload.sql}")
        missing_segment: Which path segment was not found (if applicable)
        binding_correlation_id: Request correlation for tracing (stored separately from context)

    Example:
        >>> context = ModelInfraErrorContext.with_correlation(
        ...     transport_type=EnumInfraTransportType.RUNTIME,
        ...     operation="binding_resolution",
        ...     target_name="db.query",
        ... )
        >>> raise BindingResolutionError(
        ...     "Required parameter 'sql' could not be resolved",
        ...     operation_name="db.query",
        ...     parameter_name="sql",
        ...     expression="${payload.sql}",
        ...     missing_segment="sql",
        ...     context=context,
        ... )
    """

    def __init__(
        self,
        message: str,
        *,
        operation_name: str,
        parameter_name: str,
        expression: str,
        missing_segment: str | None = None,
        correlation_id: UUID | None = None,
        context: ModelInfraErrorContext | None = None,
    ) -> None:
        """Initialize binding resolution error.

        Args:
            message: Human-readable error description
            operation_name: Operation being resolved (e.g., "db.query")
            parameter_name: Parameter that failed (e.g., "sql")
            expression: Original expression (e.g., "${payload.sql}")
            missing_segment: Path segment not found (e.g., "sql")
            correlation_id: Request correlation ID (used if context not provided)
            context: Infrastructure error context (preferred over correlation_id)
        """
        self.operation_name = operation_name
        self.parameter_name = parameter_name
        self.expression = expression
        self.missing_segment = missing_segment
        # Store binding-specific correlation_id separately
        self.binding_correlation_id = correlation_id

        # Build context if not provided
        if context is None:
            context = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="binding_resolution",
                target_name=operation_name,
            )

        super().__init__(
            message=message,
            error_code=EnumCoreErrorCode.INVALID_INPUT,
            context=context,
            operation_name=operation_name,
            parameter_name=parameter_name,
            expression=expression,
            missing_segment=missing_segment,
        )

    def __str__(self) -> str:
        """Return detailed error message with diagnostic info.

        Provides a structured diagnostic message that includes all relevant
        context for debugging binding resolution failures.

        Returns:
            Pipe-delimited string with operation, parameter, expression,
            and optionally missing segment and correlation ID.
        """
        parts = [
            f"Binding resolution failed for operation '{self.operation_name}'",
            f"Parameter: {self.parameter_name}",
            f"Expression: {self.expression}",
        ]
        if self.missing_segment:
            parts.append(f"Missing segment: {self.missing_segment}")
        # Include correlation_id from context (model attribute, always present after __init__)
        if hasattr(self, "model") and self.model.correlation_id:
            parts.append(f"Correlation ID: {self.model.correlation_id}")
        elif self.binding_correlation_id:
            parts.append(f"Correlation ID: {self.binding_correlation_id}")
        return " | ".join(parts)


__all__ = ["BindingResolutionError"]
