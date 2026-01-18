# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Service Info Model for Service Discovery Results.

This module provides ModelServiceInfo, representing information about
a discovered service instance from service discovery queries.

Architecture:
    ModelServiceInfo contains all relevant information about a service
    instance returned from a discovery query:
    - Service identity (service_id, service_name)
    - Network location (address, port)
    - Categorization (tags)
    - Health status (health_status)
    - Additional context (metadata)

    This model is backend-agnostic and represents the normalized view
    of service information across Consul, Kubernetes, Etcd, etc.

Related:
    - ModelDiscoveryResult: Contains list of ModelServiceInfo
    - ModelDiscoveryQuery: Query that returns these results
    - ProtocolServiceDiscoveryHandler: Handler protocol for backends
    - OMN-1131: Capability-oriented node architecture
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from .enum_health_status import EnumHealthStatus


class ModelServiceInfo(BaseModel):
    """Information about a discovered service instance.

    Represents a single service instance returned from a service
    discovery query. Contains normalized information regardless
    of the underlying backend (Consul, K8s, Etcd).

    Immutability:
        This model uses frozen=True to ensure service info is immutable
        once created, enabling safe concurrent access and caching.

    Attributes:
        service_id: Unique identifier for this service instance.
        service_name: Logical name of the service.
        address: Network address where the service is accessible.
        port: Port number where the service listens.
        tags: Tags associated with the service.
        health_status: Current health status of the service.
        metadata: Additional metadata associated with the service.

    Example:
        >>> from uuid import uuid4
        >>> info = ModelServiceInfo(
        ...     service_id=uuid4(),
        ...     service_name="user-service",
        ...     address="localhost",
        ...     port=8080,
        ...     tags=("api", "v2"),
        ...     health_status=EnumHealthStatus.HEALTHY,
        ...     metadata={"version": "2.1.0"},
        ... )
        >>> info.health_status
        <EnumHealthStatus.HEALTHY: 'healthy'>
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    service_id: UUID = Field(
        ...,
        description="Unique identifier for this service instance",
    )
    service_name: str = Field(
        ...,
        min_length=1,
        description="Logical name of the service",
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
        description="Tags associated with the service",
    )
    health_status: EnumHealthStatus = Field(
        default=EnumHealthStatus.UNKNOWN,
        description="Current health status of the service",
    )
    metadata: dict[str, str] = Field(
        default_factory=dict,
        description="Additional metadata associated with the service",
    )


__all__ = ["ModelServiceInfo"]
