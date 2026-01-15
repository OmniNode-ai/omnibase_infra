# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Service Discovery Health Check Details Model.

This module provides the Pydantic model for backend-specific diagnostic
information in service discovery health check results, replacing untyped
dict[str, object] with strongly-typed fields.

Architecture:
    ModelServiceDiscoveryHealthCheckDetails captures backend-specific diagnostics:
    - Agent/server connectivity information
    - Registered service counts
    - Backend version information
    - Custom backend-specific extensions

    The model uses optional fields to accommodate different backends
    that may provide varying levels of diagnostic information.

Related:
    - ModelServiceDiscoveryHealthCheckResult: Parent model that contains details
    - ProtocolServiceDiscoveryHandler: Protocol for service discovery backends
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelServiceDiscoveryHealthCheckDetails(BaseModel):
    """Backend-specific diagnostic information for service discovery health checks.

    This model captures diagnostic details from service discovery backends.
    All fields are optional as different backends provide varying
    levels of information.

    Immutability:
        This model uses frozen=True to ensure details are immutable
        once created, enabling safe sharing and logging.

    Attributes:
        agent_address: Address of the service discovery agent (e.g., "localhost:8500").
        server_version: Service discovery server version string.
        service_count: Number of services registered with this backend.
        datacenter: Datacenter or cluster name (for Consul).
        leader: Current leader node address (for clustered backends).
        cluster_size: Number of nodes in the cluster.
        healthy_services: Count of healthy service instances.
        unhealthy_services: Count of unhealthy service instances.

    Example (Consul backend):
        >>> details = ModelServiceDiscoveryHealthCheckDetails(
        ...     agent_address="localhost:8500",
        ...     server_version="1.15.4",
        ...     service_count=15,
        ...     datacenter="dc1",
        ...     healthy_services=12,
        ...     unhealthy_services=3,
        ... )
        >>> details.service_count
        15

    Example (minimal details from mock backend):
        >>> details = ModelServiceDiscoveryHealthCheckDetails(
        ...     server_version="mock-1.0.0",
        ...     service_count=5,
        ... )
        >>> details.agent_address is None
        True
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    agent_address: str | None = Field(
        default=None,
        description="Address of the service discovery agent",
    )
    server_version: str | None = Field(
        default=None,
        description="Service discovery server version string",
    )
    service_count: int | None = Field(
        default=None,
        description="Number of services registered with this backend",
        ge=0,
    )
    datacenter: str | None = Field(
        default=None,
        description="Datacenter or cluster name (for Consul)",
    )
    leader: str | None = Field(
        default=None,
        description="Current leader node address (for clustered backends)",
    )
    cluster_size: int | None = Field(
        default=None,
        description="Number of nodes in the cluster",
        ge=0,
    )
    healthy_services: int | None = Field(
        default=None,
        description="Count of healthy service instances",
        ge=0,
    )
    unhealthy_services: int | None = Field(
        default=None,
        description="Count of unhealthy service instances",
        ge=0,
    )


__all__ = ["ModelServiceDiscoveryHealthCheckDetails"]
