# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Service Discovery Health Check Result Model.

This module provides the Pydantic model for health check results from
service discovery backends, using strongly-typed model instances
for all fields including backend-specific details.

Note:
    This model is domain-specific to service discovery. For
    general-purpose health checks, see
    ``omnibase_infra.models.health.ModelHealthCheckResult``.

Related:
    - ModelServiceDiscoveryHealthCheckDetails: Backend-specific diagnostic details
    - ProtocolServiceDiscoveryHandler: Protocol for service discovery backends
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from .model_service_discovery_health_check_details import (
    ModelServiceDiscoveryHealthCheckDetails,
)


class ModelServiceDiscoveryHealthCheckResult(BaseModel):
    """Result of a service discovery backend health check.

    This model represents the outcome of a health check operation on a
    service discovery backend, providing structured diagnostics about
    the backend's operational status.

    Attributes:
        healthy: Whether the service discovery backend is healthy and operational.
            When True, the backend can accept operations normally.
            When False, consult the reason field for diagnostics.
        backend_type: Identifier for the service discovery backend (e.g., "consul", "mock").
        latency_ms: Connection latency in milliseconds.
        reason: Human-readable explanation of the health status.
        error_type: Exception type name if health check failed.
            Only populated when healthy is False.
        details: Backend-specific diagnostic information as a typed model.
            Contains agent_address, service_count, datacenter, etc.
        correlation_id: Correlation ID for distributed tracing.

    Example:
        >>> from omnibase_infra.nodes.node_service_discovery_effect.models import (
        ...     ModelServiceDiscoveryHealthCheckDetails,
        ... )
        >>> # Healthy service discovery backend
        >>> result = ModelServiceDiscoveryHealthCheckResult(
        ...     healthy=True,
        ...     backend_type="consul",
        ...     latency_ms=5.2,
        ...     reason="ok",
        ...     details=ModelServiceDiscoveryHealthCheckDetails(
        ...         agent_address="localhost:8500",
        ...         service_count=15,
        ...     ),
        ... )
        >>> if result.healthy:
        ...     print(f"{result.backend_type} is operational")
        consul is operational

        >>> # Failed health check
        >>> result = ModelServiceDiscoveryHealthCheckResult(
        ...     healthy=False,
        ...     backend_type="consul",
        ...     latency_ms=0.0,
        ...     reason="Connection refused to agent",
        ...     error_type="ConnectionRefusedError",
        ... )
        >>> if not result.healthy:
        ...     print(f"Backend unhealthy: {result.reason}")
        Backend unhealthy: Connection refused to agent
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    healthy: bool = Field(
        description="Whether the service discovery backend is healthy and operational",
    )
    backend_type: str = Field(
        description="Identifier for the service discovery backend (e.g., 'consul', 'mock')",
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
    details: ModelServiceDiscoveryHealthCheckDetails = Field(
        default_factory=ModelServiceDiscoveryHealthCheckDetails,
        description="Backend-specific diagnostic information as a typed model",
    )
    correlation_id: UUID | None = Field(
        default=None,
        description="Correlation ID for distributed tracing",
    )


__all__: list[str] = [
    "ModelServiceDiscoveryHealthCheckResult",
]
