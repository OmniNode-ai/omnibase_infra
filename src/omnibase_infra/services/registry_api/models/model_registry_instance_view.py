# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Registry instance view model for dashboard display.

Related Tickets:
    - OMN-1278: Contract-Driven Dashboard - Registry Discovery
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelRegistryInstanceView(BaseModel):
    """Live service instance view for dashboard display.

    Represents a live Consul service instance, providing real-time
    health and connectivity information.

    Attributes:
        node_id: Node UUID (may be derived from service_id if not UUID)
        service_name: Consul service name
        service_id: Consul service instance ID
        instance_id: Unique instance identifier (same as service_id)
        address: Network address (IP or hostname)
        port: Service port number
        health_status: Health check status (passing, warning, critical, unknown)
        health_output: Health check output message (nullable)
        last_check_at: Timestamp of last health check (nullable)
        tags: Consul service tags
        meta: Consul service metadata
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    node_id: UUID = Field(
        ...,
        description="Node UUID (may be derived from service_id)",
    )
    service_name: str = Field(  # ONEX_EXCLUDE: pattern - Consul discovery identifier
        ...,
        description="Consul service name (external Consul identifier, not entity reference)",
    )
    service_id: UUID = Field(
        ...,
        description="Consul service instance ID",
    )
    instance_id: UUID = Field(
        ...,
        description="Unique instance identifier",
    )
    address: str = Field(
        ...,
        description="Network address (IP or hostname)",
    )
    port: int = Field(
        ...,
        ge=1,
        le=65535,
        description="Service port number",
    )
    health_status: Literal["passing", "warning", "critical", "unknown"] = Field(
        ...,
        description="Health check status",
    )
    health_output: str | None = Field(
        default=None,
        description="Health check output message",
    )
    last_check_at: datetime | None = Field(
        default=None,
        description="Timestamp of last health check",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Consul service tags",
    )
    meta: dict[str, str] = Field(
        default_factory=dict,
        description="Consul service metadata",
    )


__all__ = ["ModelRegistryInstanceView"]
