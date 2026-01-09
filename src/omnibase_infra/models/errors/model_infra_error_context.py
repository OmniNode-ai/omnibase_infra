# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Infrastructure Error Context Configuration Model.

This module defines the configuration model for infrastructure error context,
encapsulating common structured fields to reduce __init__ parameter count
while maintaining strong typing per ONEX standards.
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

    Example:
        >>> context = ModelInfraErrorContext(
        ...     transport_type=EnumInfraTransportType.HTTP,
        ...     operation="process_request",
        ...     target_name="api-gateway",
        ...     correlation_id=uuid4(),
        ...     namespace="engineering",
        ... )
        >>> raise RuntimeHostError("Operation failed", context=context)
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
            **kwargs: Additional context fields (transport_type, operation, target_name).

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


__all__ = ["ModelInfraErrorContext"]
