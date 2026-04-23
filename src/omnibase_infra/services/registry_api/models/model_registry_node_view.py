# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
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
from omnibase_core.types import JsonType


class ModelRegistryNodeView(BaseModel):
    """Node view for dashboard display.

    Represents a registered ONEX node from the registration projection,
    flattened for dashboard consumption.

    Attributes:
        node_id: Canonical registry identifier (entity_id from projection)
        name: Stable registry label. Until projections carry a canonical
            human name, this mirrors the canonical node_id string.
        service_name: Deprecated compatibility field for legacy service discovery
            consumers. Not canonical registry identity.
        namespace: Optional registry namespace derived from projection domain
        display_name: Optional human-friendly fallback label
        node_type: ONEX node archetype (EFFECT, COMPUTE, REDUCER, ORCHESTRATOR)
        version: Semantic version (ModelSemVer instance)
        state: Current FSM registration state
        capabilities: List of capability tags
        capability_details: Safe typed capability subset derived from projection
        contract_type: Contract type declared in projection
        contract_version: Contract version declared in projection
        registered_at: Timestamp of initial registration
        last_heartbeat_at: Timestamp of last heartbeat (nullable)
        updated_at: Timestamp of last projection update
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    node_id: UUID = Field(
        ...,
        description="Canonical registry node identifier",
    )
    name: str = Field(
        ...,
        description=(
            "Stable registry node label. Mirrors node_id until projections expose "
            "a canonical human name."
        ),
    )
    service_name: str | None = (
        Field(  # ONEX_EXCLUDE: pattern - legacy discovery compatibility
            default=None,
            description=(
                "Deprecated compatibility field for legacy service discovery naming. "
                "Not canonical registry identity."
            ),
        )
    )
    namespace: str | None = Field(
        default=None,
        description="Optional registry namespace derived from projection domain",
    )
    display_name: str | None = Field(
        default=None,
        description="Optional human-friendly fallback label",
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
    capability_details: dict[str, JsonType] = Field(
        default_factory=dict,
        description="Safe typed capability subset derived from projection data",
    )
    contract_type: str | None = Field(
        default=None,
        description="Contract type declared in the registration projection",
    )
    contract_version: str | None = Field(
        default=None,
        description="Contract version declared in the registration projection",
    )
    registered_at: datetime = Field(
        ...,
        description="Timestamp of initial registration",
    )
    last_heartbeat_at: datetime | None = Field(
        default=None,
        description="Timestamp of last heartbeat",
    )
    updated_at: datetime = Field(
        ...,
        description="Timestamp of the last projection update",
    )


__all__ = ["ModelRegistryNodeView"]
