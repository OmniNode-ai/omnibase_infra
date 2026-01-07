# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Registration Result Model for Service Discovery Handlers.

This module provides the ModelRegistrationResult class representing the outcome
of a service registration operation at the handler level.

Note:
    This model is aligned with the node-level ModelRegistrationResult at
    nodes/node_service_discovery_effect/models/model_registration_result.py
    with an additional duration_ms field for handler-level timing metrics.
"""

from __future__ import annotations

from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.node_service_discovery_effect.models.enum_service_discovery_operation import (
    EnumServiceDiscoveryOperation,
)


class ModelRegistrationResult(BaseModel):
    """Result of a service registration operation.

    Captures the outcome of registering a service with the service discovery
    backend. Immutable once created.

    This model is schema-compatible with the node-level ModelRegistrationResult,
    with an additional duration_ms field for handler timing metrics.

    Attributes:
        success: Whether the registration completed successfully.
        service_id: ID of the registered/deregistered service.
        operation: Type of operation performed (register or deregister).
        error: Sanitized error message if registration failed.
        duration_ms: Time taken for the operation in milliseconds.
        backend_type: The backend that handled the registration.
        correlation_id: Correlation ID for distributed tracing.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    success: bool = Field(
        ...,
        description="Whether the registration completed successfully",
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
        description="Sanitized error message if registration failed",
    )
    duration_ms: float = Field(
        default=0.0,
        description="Time taken for the operation in milliseconds",
        ge=0.0,
    )
    backend_type: str | None = Field(
        default=None,
        description="The backend type that handled the registration",
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
