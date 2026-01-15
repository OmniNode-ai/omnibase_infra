# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Timeout Error Context Model - Stricter Typing for Timeout Errors.

This module defines a specialized context model for timeout errors that enforces
correlation_id is always present. This is stricter than ModelInfraErrorContext
which allows optional correlation_id.

Design Decision:
    Timeout errors require correlation_id for several reasons:
    1. Timeout events are often critical for debugging distributed systems
    2. Timeout chains need to be traceable across services
    3. Operators need to correlate timeout events with upstream requests
    4. Replay and debugging require deterministic correlation tracking

    By using a stricter model, we ensure at compile-time (via type checking)
    that all timeout error contexts include correlation_id.

Related:
    - CLAUDE.md: Error Context section requires correlation_id for timeout errors
    - ModelInfraErrorContext: Base model with optional fields
    - InfraTimeoutError: Uses this model for stricter typing
"""

from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.enums import EnumInfraTransportType


class ModelTimeoutErrorContext(BaseModel):
    """Specialized error context for timeout errors with required correlation_id.

    This model enforces stricter typing than ModelInfraErrorContext by requiring
    correlation_id to always be present. If not provided during construction,
    a UUID is auto-generated to ensure all timeout errors are traceable.

    The auto-generation behavior allows existing code patterns like:
        context = ModelTimeoutErrorContext(
            transport_type=EnumInfraTransportType.DATABASE,
            operation="execute_query",
        )

    While still guaranteeing correlation_id is present:
        assert context.correlation_id is not None  # Always passes

    Attributes:
        transport_type: Type of infrastructure transport (HTTP, DATABASE, KAFKA, etc.)
            Required for timeout context to identify the transport layer.
        operation: Operation being performed (required for timeout context).
        target_name: Target resource or endpoint name (optional).
        correlation_id: Request correlation ID for distributed tracing (required,
            auto-generated if not provided).
        timeout_seconds: The timeout value that was exceeded (optional).

    Example:
        >>> # With explicit correlation_id
        >>> context = ModelTimeoutErrorContext(
        ...     transport_type=EnumInfraTransportType.DATABASE,
        ...     operation="execute_query",
        ...     target_name="postgresql-primary",
        ...     correlation_id=request.correlation_id,
        ...     timeout_seconds=30.0,
        ... )
        >>> raise InfraTimeoutError("Query exceeded timeout", context=context)

        >>> # Auto-generated correlation_id
        >>> context = ModelTimeoutErrorContext(
        ...     transport_type=EnumInfraTransportType.HTTP,
        ...     operation="fetch_resource",
        ... )
        >>> assert context.correlation_id is not None  # Guaranteed
    """

    model_config = ConfigDict(
        frozen=True,  # Immutable for thread safety
        extra="forbid",  # Strict validation - no extra fields
        from_attributes=True,  # Support pytest-xdist compatibility
    )

    transport_type: EnumInfraTransportType = Field(
        ...,
        description="Type of infrastructure transport - required for timeout context",
    )
    operation: str = Field(
        ...,
        min_length=1,
        description="Operation being performed - required for timeout context",
    )
    target_name: str | None = Field(
        default=None,
        description="Target resource or endpoint name",
    )
    correlation_id: UUID = Field(
        default_factory=uuid4,
        description="Request correlation ID - required, auto-generated if not provided",
    )
    timeout_seconds: float | None = Field(
        default=None,
        ge=0.0,
        description="The timeout value that was exceeded",
    )

    # NOTE: No model_validator needed - default_factory=uuid4 on correlation_id
    # guarantees the field is always populated in Pydantic v2. The type annotation
    # `UUID` (not `UUID | None`) documents the invariant at compile-time.


__all__ = ["ModelTimeoutErrorContext"]
