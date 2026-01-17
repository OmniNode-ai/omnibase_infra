# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Discovery Query Model for Service Discovery Operations.

This module provides ModelDiscoveryQuery, representing query parameters
for service discovery operations in the NodeServiceDiscoveryEffect node.

Architecture:
    ModelDiscoveryQuery captures query criteria for discovering services:
    - service_name: Filter by service name (optional)
    - tags: Filter by tags (optional, all must match)
    - health_filter: Filter by health status (optional)

    All fields are optional to support flexible querying:
    - No filters: Return all registered services
    - service_name only: Return all instances of a specific service
    - tags only: Return services with matching tags
    - health_filter: Return only healthy/unhealthy services
    - Combinations: Apply multiple filters (AND logic)

Related:
    - ModelDiscoveryResult: Result model for discovery queries
    - ModelServiceInfo: Individual service information
    - ProtocolServiceDiscoveryHandler: Handler protocol for backends
    - OMN-1131: Capability-oriented node architecture
"""

from __future__ import annotations

from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from .enum_health_status import EnumHealthStatus


class ModelDiscoveryQuery(BaseModel):
    """Query parameters for service discovery operations.

    Defines criteria for discovering services from the service
    discovery backend. All fields are optional to support flexible
    querying patterns.

    Immutability:
        This model uses frozen=True to ensure queries are immutable
        once created, enabling safe concurrent access and caching.

    Query Logic:
        - All specified fields use AND logic
        - Empty/None fields are ignored
        - Tags use AND logic (all specified tags must be present)
        - health_filter=None returns services regardless of health

    Attributes:
        service_name: Service name to search for (optional).
            When provided, only services with this name are returned.
        tags: Tags to filter by (optional).
            When provided, only services with ALL specified tags are returned.
        health_filter: Health status filter (optional).
            When provided, only services with matching health status are returned.
        correlation_id: Correlation ID for distributed tracing.

    Example:
        >>> # Query all instances of user-service that are healthy
        >>> query = ModelDiscoveryQuery(
        ...     service_name="user-service",
        ...     health_filter=EnumHealthStatus.HEALTHY,
        ... )

        >>> # Query all services with 'production' and 'api' tags
        >>> query = ModelDiscoveryQuery(
        ...     tags=("production", "api"),
        ... )

        >>> # Query all registered services
        >>> query = ModelDiscoveryQuery()
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    service_name: str | None = Field(
        default=None,
        description="Service name to search for (optional)",
    )
    tags: tuple[str, ...] | None = Field(
        default=None,
        description="Tags to filter by - all must match (optional)",
    )
    health_filter: EnumHealthStatus | None = Field(
        default=None,
        description="Health status filter (optional)",
    )
    correlation_id: UUID = Field(
        default_factory=uuid4,
        description="Correlation ID for distributed tracing",
    )


__all__ = ["ModelDiscoveryQuery"]
