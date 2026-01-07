# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Storage Health Check Result Model.

This module provides the Pydantic model for health check results from
registration storage backends, using strongly-typed model instances
for all fields including backend-specific details.

Note:
    This model is domain-specific to registration storage. For
    general-purpose health checks, see
    ``omnibase_infra.models.health.ModelHealthCheckResult``.

Related:
    - ModelStorageHealthCheckDetails: Backend-specific diagnostic details
    - ProtocolRegistrationStorageHandler: Protocol for storage backends
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from .model_storage_health_check_details import ModelStorageHealthCheckDetails


class ModelStorageHealthCheckResult(BaseModel):
    """Result of a storage backend health check.

    This model represents the outcome of a health check operation on a
    registration storage backend, providing structured diagnostics about
    the storage backend's operational status.

    Attributes:
        healthy: Whether the storage backend is healthy and operational.
            When True, the backend can accept operations normally.
            When False, consult the reason field for diagnostics.
        backend_type: Identifier for the storage backend (e.g., "postgresql", "mock").
        latency_ms: Connection latency in milliseconds.
        reason: Human-readable explanation of the health status.
        error_type: Exception type name if health check failed.
            Only populated when healthy is False.
        details: Backend-specific diagnostic information as a typed model.
            Contains pool_size, active_connections, server_version, etc.
        correlation_id: Correlation ID for distributed tracing.

    Example:
        >>> from omnibase_infra.nodes.node_registration_storage_effect.models import (
        ...     ModelStorageHealthCheckDetails,
        ... )
        >>> # Healthy storage backend
        >>> result = ModelStorageHealthCheckResult(
        ...     healthy=True,
        ...     backend_type="postgresql",
        ...     latency_ms=2.5,
        ...     reason="ok",
        ...     details=ModelStorageHealthCheckDetails(
        ...         pool_size=10,
        ...         active_connections=3,
        ...     ),
        ... )
        >>> if result.healthy:
        ...     print(f"{result.backend_type} is operational")
        postgresql is operational

        >>> # Failed health check
        >>> result = ModelStorageHealthCheckResult(
        ...     healthy=False,
        ...     backend_type="postgresql",
        ...     latency_ms=0.0,
        ...     reason="Connection refused",
        ...     error_type="ConnectionRefusedError",
        ... )
        >>> if not result.healthy:
        ...     print(f"Backend unhealthy: {result.reason}")
        Backend unhealthy: Connection refused
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    healthy: bool = Field(
        description="Whether the storage backend is healthy and operational",
    )
    backend_type: str = Field(
        description="Identifier for the storage backend (e.g., 'postgresql', 'mock')",
    )
    latency_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Connection latency in milliseconds",
    )
    reason: str = Field(
        default="ok",
        description="Human-readable explanation of the health status",
    )
    error_type: str | None = Field(
        default=None,
        description="Exception type name if health check failed",
    )
    details: ModelStorageHealthCheckDetails = Field(
        default_factory=ModelStorageHealthCheckDetails,
        description="Backend-specific diagnostic information as a typed model",
    )
    correlation_id: UUID | None = Field(
        default=None,
        description="Correlation ID for distributed tracing",
    )


__all__: list[str] = [
    "ModelStorageHealthCheckResult",
]
