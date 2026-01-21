# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Service Information Model for Service Discovery Handlers.

This module provides the ModelServiceInfo class representing service metadata
for service discovery handler operations.

Note:
    This model is aligned with the node-level ModelServiceInfo at
    nodes/node_service_discovery_effect/models/model_service_info.py
    with additional handler-level fields (health_check_url, registered_at,
    correlation_id) for handler context and tracing.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.node_service_discovery_effect.models.enum_health_status import (
    EnumHealthStatus,
)


class ModelServiceInfo(BaseModel):
    """Service information for discovery and registration handlers.

    Represents a service instance with its metadata for service discovery
    handler operations. Immutable once created.

    This model is schema-compatible with the node-level ModelServiceInfo,
    with additional handler-level fields for tracing and context:
    - health_check_url: URL for health check verification
    - registered_at: Timestamp for registration tracking
    - correlation_id: Correlation ID for distributed tracing

    Attributes:
        service_id: Unique identifier for the service instance.
        service_name: Human-readable name of the service.
        address: Network address (hostname or IP) of the service.
        port: Network port the service listens on.
        tags: Tags for filtering and categorization.
        health_status: Current health status of the service.
        metadata: Additional key-value metadata for the service.
        health_check_url: Optional URL for health checks (handler extension).
        health_output: Output message from the last health check (handler extension).
        last_check_at: Timestamp of the last health check (handler extension).
        registered_at: Timestamp when the service was registered (handler extension).
        correlation_id: Correlation ID for tracing (handler extension).
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    service_id: UUID = Field(
        ...,
        description="Unique identifier for the service instance",
    )
    service_name: str = Field(
        ...,
        description="Human-readable name of the service",
        min_length=1,
    )
    address: str | None = Field(
        default=None,
        description="Network address (hostname or IP) of the service",
    )
    port: int | None = Field(
        default=None,
        description="Network port the service listens on",
        ge=1,
        le=65535,
    )
    tags: tuple[str, ...] = Field(
        default=(),
        description="Tags for filtering and categorization",
    )
    health_status: EnumHealthStatus = Field(
        default=EnumHealthStatus.UNKNOWN,
        description="Current health status of the service",
    )
    metadata: dict[str, str] = Field(
        default_factory=dict,
        description="Additional key-value metadata",
    )
    health_check_url: str | None = Field(
        default=None,
        description="Optional URL for health checks",
    )
    health_output: str | None = Field(
        default=None,
        description="Output message from the last health check",
    )
    last_check_at: datetime | None = Field(
        default=None,
        description="Timestamp of the last health check",
    )
    registered_at: datetime | None = Field(
        default=None,
        description="Timestamp when the service was registered",
    )
    correlation_id: UUID | None = Field(
        default=None,
        description="Correlation ID for tracing",
    )


__all__ = ["ModelServiceInfo"]
