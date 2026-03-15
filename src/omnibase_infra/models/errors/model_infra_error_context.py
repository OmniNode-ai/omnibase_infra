# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Infrastructure Error Context Configuration Model.

This module defines the configuration model for infrastructure error context,
encapsulating common structured fields to reduce __init__ parameter count
while maintaining strong typing per ONEX standards.

Enhanced in OMN-518 with:
- ``suggested_resolution``: Human-readable resolution guidance
- ``retry_after_seconds``: Retry delay guidance for transient errors
- ``original_error_type``: Preserved original exception class name for error chaining
"""

from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.enums import EnumInfraTransportType


class ModelInfraErrorContext(BaseModel):
    """Configuration model for infrastructure error context.

    Encapsulates common structured fields for infrastructure errors
    to reduce __init__ parameter count while maintaining strong typing.
    This follows the ONEX pattern of using configuration models to
    bundle related parameters.

    Attributes:
        transport_type: Type of infrastructure transport (HTTP, DATABASE, KAFKA, etc.)
        operation: Operation being performed (connect, query, authenticate, etc.)
        target_name: Target resource or endpoint name
        correlation_id: Request correlation ID for distributed tracing
        namespace: Vault namespace (Enterprise feature) or other service-specific namespace
        suggested_resolution: Human-readable suggestion for resolving the error
        retry_after_seconds: Recommended delay before retrying (for transient errors)
        original_error_type: Preserved original exception class name for error chaining

    Example:
        >>> context = ModelInfraErrorContext(
        ...     transport_type=EnumInfraTransportType.DATABASE,
        ...     operation="connect",
        ...     suggested_resolution="Check PostgreSQL is running and credentials are valid",
        ...     retry_after_seconds=30,
        ... )
        >>> raise InfraConnectionError("Failed to connect", context=context)
    """

    model_config = ConfigDict(
        frozen=True,  # Immutable for thread safety
        extra="forbid",  # Strict validation - no extra fields
        from_attributes=True,  # Support pytest-xdist compatibility
    )

    transport_type: EnumInfraTransportType | None = Field(
        default=None,
        description="Type of infrastructure transport (HTTP, DATABASE, KAFKA, etc.)",
    )
    operation: str | None = Field(
        default=None,
        description="Operation being performed (connect, query, authenticate, etc.)",
    )
    target_name: str | None = Field(
        default=None,
        description="Target resource or endpoint name",
    )
    correlation_id: UUID | None = Field(
        default=None,
        description="Request correlation ID for distributed tracing",
    )
    namespace: str | None = Field(
        default=None,
        description="Vault namespace (Enterprise feature) or other service-specific namespace",
    )
    suggested_resolution: str | None = Field(
        default=None,
        description="Human-readable suggestion for resolving the error",
    )
    retry_after_seconds: float | None = Field(
        default=None,
        ge=0.0,
        description="Recommended delay in seconds before retrying (for transient errors)",
    )
    original_error_type: str | None = Field(
        default=None,
        description="Preserved original exception class name for error chaining diagnostics",
    )

    @classmethod
    def with_correlation(
        cls,
        correlation_id: UUID | None = None,
        **kwargs: object,
    ) -> "ModelInfraErrorContext":
        """Create context with auto-generated correlation_id if not provided.

        This factory method ensures a correlation_id is always present,
        generating one if not explicitly provided. Useful for distributed
        tracing scenarios where every error should be traceable.

        Args:
            correlation_id: Optional correlation ID. If None, one is auto-generated.
            **kwargs: Additional context fields (transport_type, operation, etc.).

        Returns:
            ModelInfraErrorContext with guaranteed correlation_id.

        Example:
            >>> context = ModelInfraErrorContext.with_correlation(
            ...     transport_type=EnumInfraTransportType.HTTP,
            ...     operation="process_request",
            ... )
            >>> assert context.correlation_id is not None
        """
        return cls(correlation_id=correlation_id or uuid4(), **kwargs)

    @classmethod
    def from_exception(
        cls,
        exc: BaseException,
        correlation_id: UUID | None = None,
        **kwargs: object,
    ) -> "ModelInfraErrorContext":
        """Create context from an exception, preserving the original error type.

        This factory captures the original exception's class name into
        ``original_error_type`` for diagnostic purposes during error chaining.

        Args:
            exc: The original exception to capture type information from.
            correlation_id: Optional correlation ID. If None, one is auto-generated.
            **kwargs: Additional context fields (transport_type, operation, etc.).

        Returns:
            ModelInfraErrorContext with original_error_type and guaranteed correlation_id.

        Example:
            >>> try:
            ...     connection.execute(query)
            ... except psycopg.OperationalError as e:
            ...     context = ModelInfraErrorContext.from_exception(
            ...         e,
            ...         transport_type=EnumInfraTransportType.DATABASE,
            ...         operation="execute_query",
            ...     )
            ...     raise InfraConnectionError("Query failed", context=context) from e
        """
        return cls(
            correlation_id=correlation_id or uuid4(),
            original_error_type=type(exc).__name__,
            **kwargs,
        )


__all__ = ["ModelInfraErrorContext"]
