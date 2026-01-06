# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Registration Result Model for Service Discovery Operations.

This module provides ModelRegistrationResult, representing the result
of service registration operations from the NodeServiceDiscoveryEffect node.

Architecture:
    ModelRegistrationResult captures the outcome of a service registration
    or deregistration operation:
    - success: Whether the operation completed successfully
    - service_id: ID of the registered/deregistered service
    - error: Error message if operation failed

    This model is backend-agnostic and represents a normalized view
    of registration results regardless of underlying backend.

Related:
    - ModelServiceRegistration: Input model for registration
    - ProtocolServiceDiscoveryHandler: Handler protocol for backends
    - NodeServiceDiscoveryEffect: Effect node that produces these results
    - OMN-1131: Capability-oriented node architecture
"""

from __future__ import annotations

from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class ModelRegistrationResult(BaseModel):
    """Result of service registration or deregistration operation.

    Contains the outcome of a registration operation along with
    any error information if the operation failed.

    Immutability:
        This model uses frozen=True to ensure results are immutable
        once created, enabling safe concurrent access.

    Attributes:
        success: Whether the operation completed successfully.
        service_id: ID of the registered/deregistered service.
        error: Error message if operation failed (None on success).
        backend_type: Type of backend that processed the operation.
        correlation_id: Correlation ID for distributed tracing.

    Example:
        >>> from uuid import uuid4
        >>> # Successful registration
        >>> result = ModelRegistrationResult(
        ...     success=True,
        ...     service_id=uuid4(),
        ...     backend_type="consul",
        ... )
        >>> result.success
        True

        >>> # Failed registration
        >>> result = ModelRegistrationResult(
        ...     success=False,
        ...     service_id=uuid4(),
        ...     error="Connection timeout to Consul agent",
        ...     backend_type="consul",
        ... )
        >>> result.success
        False
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    success: bool = Field(
        ...,
        description="Whether the operation completed successfully",
    )
    service_id: UUID | None = Field(
        default=None,
        description="ID of the registered/deregistered service",
    )
    error: str | None = Field(
        default=None,
        description="Error message if operation failed (None on success)",
    )
    backend_type: str | None = Field(
        default=None,
        description="Type of backend that processed the operation",
    )
    correlation_id: UUID = Field(
        default_factory=uuid4,
        description="Correlation ID for distributed tracing",
    )

    def __bool__(self) -> bool:
        """Return True if the operation was successful.

        Warning:
            This overrides standard Pydantic behavior where `bool(model)`
            always returns True. This model returns True only when the
            operation was successful.

        Returns:
            True if success is True, False otherwise.

        Example:
            >>> result = ModelRegistrationResult(success=True)
            >>> if result:  # Works intuitively
            ...     print("Registration succeeded")
        """
        return self.success


__all__ = ["ModelRegistrationResult"]
