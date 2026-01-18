# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Service Registration Model for Service Discovery Operations.

This module provides ModelServiceRegistration, representing input for
service registration operations in the NodeServiceDiscoveryEffect node.

Architecture:
    ModelServiceRegistration captures all information needed to register
    a service with a service discovery backend:
    - Service identity (service_id, service_name)
    - Network location (address, port)
    - Categorization (tags)
    - Additional context (metadata)
    - Liveness verification (health_check)

    This model is backend-agnostic and can be used with Consul,
    Kubernetes, Etcd, or other service discovery implementations.

Related:
    - NodeServiceDiscoveryEffect: Effect node that consumes this model
    - ProtocolServiceDiscoveryHandler: Handler protocol for backends
    - ModelRegistrationResult: Result model for registration operations
    - OMN-1131: Capability-oriented node architecture
"""

from __future__ import annotations

from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from .model_health_check_config import ModelHealthCheckConfig


class ModelServiceRegistration(BaseModel):
    """Input model for service registration operations.

    Contains all information needed to register a service with a
    service discovery backend. The node delegates actual registration
    to the configured ProtocolServiceDiscoveryHandler implementation.

    Immutability:
        This model uses frozen=True to ensure registrations are immutable
        once created, enabling safe concurrent access.

    Attributes:
        service_id: Unique identifier for this service instance.
            Auto-generated if not provided.
        service_name: Logical name of the service (e.g., 'user-service').
            Used for service discovery queries.
        address: Network address where the service is accessible.
            Can be hostname or IP address.
        port: Port number where the service listens.
        tags: Tags for service categorization and filtering.
            Used in discovery queries to find specific service subsets.
        metadata: Additional key-value metadata for the service.
            Backend-specific attributes can be passed here.
        health_check: Optional health check configuration model.
            Defines how the backend should verify service health.
        correlation_id: Correlation ID for distributed tracing.

    Example:
        >>> from uuid import uuid4
        >>> registration = ModelServiceRegistration(
        ...     service_id=uuid4(),
        ...     service_name="user-service",
        ...     address="localhost",
        ...     port=8080,
        ...     tags=("api", "v2", "production"),
        ...     metadata={"version": "2.1.0", "region": "us-east-1"},
        ...     health_check=ModelHealthCheckConfig(
        ...         endpoint="/health",
        ...         interval="10s",
        ...         timeout="5s",
        ...     ),
        ... )
        >>> registration.service_name
        'user-service'
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    service_id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier for this service instance",
    )
    service_name: str = Field(
        ...,
        min_length=1,
        description="Logical name of the service (e.g., 'user-service')",
    )
    address: str | None = Field(
        default=None,
        description="Network address where the service is accessible",
    )
    port: int | None = Field(
        default=None,
        ge=1,
        le=65535,
        description="Port number where the service listens",
    )
    tags: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Tags for service categorization and filtering",
    )
    metadata: dict[str, str] = Field(
        default_factory=dict,
        description="Additional key-value metadata for the service",
    )
    health_check: ModelHealthCheckConfig | None = Field(
        default=None,
        description="Health check configuration (interval, timeout, endpoint)",
    )
    correlation_id: UUID = Field(
        default_factory=uuid4,
        description="Correlation ID for distributed tracing",
    )


__all__ = ["ModelServiceRegistration"]
