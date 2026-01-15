# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Handler Registration Result Model for Service Discovery Operations.

This module provides ModelHandlerRegistrationResult, representing the result
of service registration operations from service discovery handlers.

Architecture:
    ModelHandlerRegistrationResult captures the outcome of a service registration
    or deregistration operation at the handler level:
    - success: Whether the operation completed successfully
    - service_id: ID of the registered/deregistered service
    - error: Error message if operation failed

    This model is handler-level specific and distinct from the node-level
    ModelRegistrationResult to avoid namespace conflicts while maintaining
    similar semantics.

Related:
    - ModelServiceInfo: Handler-level service info model
    - ProtocolServiceDiscoveryHandler: Handler protocol for backends
    - ModelRegistrationResult (node-level): Node I/O model
    - OMN-1131: Capability-oriented node architecture
"""

from __future__ import annotations

from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.node_service_discovery_effect.models.enum_service_discovery_operation import (
    EnumServiceDiscoveryOperation,
)


class ModelHandlerRegistrationResult(BaseModel):
    """Result of service registration or deregistration handler operation.

    Contains the outcome of a registration operation along with
    any error information if the operation failed.

    Note:
        This is the handler-level model distinct from the node-level
        ModelRegistrationResult. Use this model in handler implementations
        and protocol definitions.

    Immutability:
        This model uses frozen=True to ensure results are immutable
        once created, enabling safe concurrent access.

    Attributes:
        success: Whether the operation completed successfully.
        service_id: ID of the registered/deregistered service.
        operation: Type of operation performed (register or deregister).
        error: Error message if operation failed (None on success).
        duration_ms: Time taken for the operation in milliseconds.
        backend_type: Type of backend that processed the operation.
        correlation_id: Correlation ID for distributed tracing.

    Example:
        >>> from uuid import uuid4
        >>> # Successful registration
        >>> result = ModelHandlerRegistrationResult(
        ...     success=True,
        ...     service_id=uuid4(),
        ...     operation=EnumServiceDiscoveryOperation.REGISTER,
        ...     duration_ms=45.2,
        ...     backend_type="consul",
        ... )
        >>> result.success
        True

        >>> # Failed deregistration
        >>> result = ModelHandlerRegistrationResult(
        ...     success=False,
        ...     service_id=uuid4(),
        ...     operation=EnumServiceDiscoveryOperation.DEREGISTER,
        ...     error="Connection timeout to Consul agent",
        ...     duration_ms=5000.0,
        ...     backend_type="consul",
        ... )
        >>> result.success
        False
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    success: bool = Field(
        ...,
        description="Whether the operation completed successfully",
    )
    service_id: UUID | None = Field(
        default=None,
        description="ID of the registered/deregistered service",
    )
    operation: EnumServiceDiscoveryOperation | None = Field(
        default=None,
        description="Type of operation performed (register or deregister)",
    )
    error: str | None = Field(
        default=None,
        description="Error message if operation failed (None on success)",
    )
    duration_ms: float = Field(
        default=0.0,
        description="Time taken for the operation in milliseconds",
        ge=0.0,
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
            >>> result = ModelHandlerRegistrationResult(success=True)
            >>> if result:  # Works intuitively
            ...     print("Registration succeeded")
        """
        return self.success


__all__ = ["ModelHandlerRegistrationResult"]
