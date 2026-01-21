# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Registry node view model for dashboard display.

Related Tickets:
    - OMN-1278: Contract-Driven Dashboard - Registry Discovery
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_core.models.primitives.model_semver import ModelSemVer


class ModelRegistryNodeView(BaseModel):
    """Node view for dashboard display.

    Represents a registered ONEX node from the registration projection,
    flattened for dashboard consumption.

    Attributes:
        node_id: Unique identifier (entity_id from projection)
        name: Human-readable node name (from service_name or node_type)
        service_name: Consul service name for discovery
        namespace: Optional namespace for multi-tenant deployments
        display_name: Optional human-friendly display name
        node_type: ONEX node archetype (EFFECT, COMPUTE, REDUCER, ORCHESTRATOR)
        version: Semantic version (ModelSemVer instance)
        state: Current FSM registration state
        capabilities: List of capability tags
        registered_at: Timestamp of initial registration
        last_heartbeat_at: Timestamp of last heartbeat (nullable)
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    node_id: UUID = Field(
        ...,
        description="Unique node identifier",
    )
    name: str = Field(
        ...,
        description="Node name (service_name or derived from node_type)",
    )
    service_name: str = Field(  # ONEX_EXCLUDE: pattern - Consul discovery identifier
        ...,
        description="Consul service name (external Consul identifier, not entity reference)",
    )
    namespace: str | None = Field(
        default=None,
        description="Optional namespace for multi-tenant deployments",
    )
    display_name: str | None = Field(
        default=None,
        description="Optional human-friendly display name",
    )
    node_type: Literal["EFFECT", "COMPUTE", "REDUCER", "ORCHESTRATOR"] = Field(
        ...,
        description="ONEX node archetype",
    )
    version: ModelSemVer = Field(
        ...,
        description="Semantic version (ONEX standard)",
    )
    state: str = Field(
        ...,
        description="Current FSM registration state",
    )
    capabilities: list[str] = Field(
        default_factory=list,
        description="List of capability tags",
    )
    registered_at: datetime = Field(
        ...,
        description="Timestamp of initial registration",
    )
    last_heartbeat_at: datetime | None = Field(
        default=None,
        description="Timestamp of last heartbeat",
    )


__all__ = ["ModelRegistryNodeView"]
