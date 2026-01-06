# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Service Information Model.

This module provides the ModelServiceInfo class representing service metadata
for service discovery operations.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelServiceInfo(BaseModel):
    """Service information for discovery and registration.

    Represents a service instance with its metadata for service discovery
    operations. Immutable once created.

    Attributes:
        service_id: Unique identifier for the service instance.
        service_name: Human-readable name of the service.
        address: Network address (hostname or IP) of the service.
        port: Network port the service listens on.
        tags: List of tags for filtering and categorization.
        metadata: Additional key-value metadata for the service.
        health_check_url: Optional URL for health checks.
        registered_at: Timestamp when the service was registered.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    service_id: UUID = Field(
        ...,
        description="Unique identifier for the service instance",
    )
    service_name: str = Field(
        ...,
        description="Human-readable name of the service",
        min_length=1,
    )
    address: str = Field(
        ...,
        description="Network address (hostname or IP) of the service",
        min_length=1,
    )
    port: int = Field(
        ...,
        description="Network port the service listens on",
        ge=1,
        le=65535,
    )
    tags: tuple[str, ...] = Field(
        default=(),
        description="Tags for filtering and categorization",
    )
    metadata: dict[str, str] = Field(
        default_factory=dict,
        description="Additional key-value metadata",
    )
    health_check_url: str | None = Field(
        default=None,
        description="Optional URL for health checks",
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
