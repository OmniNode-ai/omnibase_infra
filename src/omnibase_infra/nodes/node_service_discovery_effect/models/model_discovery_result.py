# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Discovery Result Model for Service Discovery Operations.

This module provides ModelDiscoveryResult, representing the result
of service discovery queries from the NodeServiceDiscoveryEffect node.

Architecture:
    ModelDiscoveryResult contains the results of a service discovery
    query along with metadata about the query execution:
    - services: List of discovered service instances
    - query_metadata: Information about query execution (timing, backend)

    The result is backend-agnostic and represents a normalized view
    of discovered services regardless of underlying backend.

Related:
    - ModelServiceInfo: Individual service information
    - ModelDiscoveryQuery: Query that produced these results
    - ProtocolServiceDiscoveryHandler: Handler protocol for backends
    - OMN-1131: Capability-oriented node architecture
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from .model_query_metadata import ModelQueryMetadata
from .model_service_info import ModelServiceInfo


class ModelDiscoveryResult(BaseModel):
    """Result of service discovery query.

    Contains the list of discovered services along with metadata
    about the query execution. This is the primary output model
    for discovery operations.

    Immutability:
        This model uses frozen=True to ensure results are immutable
        once created, enabling safe concurrent access and caching.

    Attributes:
        services: List of discovered service instances.
            Empty list indicates no services matched the query.
        query_metadata: Metadata about query execution.
        correlation_id: Correlation ID for distributed tracing.

    Example:
        >>> from datetime import UTC, datetime
        >>> from uuid import uuid4
        >>> result = ModelDiscoveryResult(
        ...     services=(
        ...         ModelServiceInfo(
        ...             service_id=uuid4(),
        ...             service_name="user-service",
        ...             health_status=EnumHealthStatus.HEALTHY,
        ...         ),
        ...     ),
        ...     query_metadata=ModelQueryMetadata(
        ...         backend_type="consul",
        ...         query_duration_ms=5.2,
        ...         timestamp=datetime.now(UTC),
        ...     ),
        ... )
        >>> len(result.services)
        1
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    services: tuple[ModelServiceInfo, ...] = Field(
        default_factory=tuple,
        description="List of discovered service instances",
    )
    query_metadata: ModelQueryMetadata | None = Field(
        default=None,
        description="Metadata about query execution",
    )
    correlation_id: UUID | None = Field(
        default=None,
        description="Correlation ID for distributed tracing",
    )

    @property
    def service_count(self) -> int:
        """Return the number of discovered services."""
        return len(self.services)

    @property
    def is_empty(self) -> bool:
        """Return True if no services were discovered."""
        return len(self.services) == 0


__all__ = ["ModelDiscoveryResult"]
