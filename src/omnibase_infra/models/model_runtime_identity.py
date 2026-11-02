# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The runtime's own identity: the fields a node announcement does not carry.

A node announcement names a contract and its package. The registration
orchestrator reads ``ModelNodeIntrospectionEvent``, whose identity fields
(node id, node type, endpoints, deployment) come from the running runtime.
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_core.enums import EnumNodeKind
from omnibase_core.models.primitives.model_semver import ModelSemVer
from omnibase_infra.models.registration.model_node_event_bus_config import (
    ModelNodeEventBusConfig,
)
from omnibase_infra.models.registration.model_node_introspection_event import (
    ModelNodeIntrospectionEvent,
)


class ModelRuntimeIdentity(BaseModel):
    """Identity of the runtime that hosts the announced nodes."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    node_id: UUID = Field(..., description="Runtime node identifier")
    node_type: EnumNodeKind = Field(..., description="ONEX node type of the runtime")
    node_version: ModelSemVer = Field(
        default_factory=lambda: ModelSemVer(major=1, minor=0, patch=0),
        description="Fallback version when the announced package version is not semver",
    )
    endpoints: dict[str, str] = Field(
        default_factory=dict, description="Exposed endpoints (name -> URL)"
    )
    node_role: str | None = None
    network_id: str | None = None
    deployment_id: str | None = None
    event_bus: ModelNodeEventBusConfig | None = None

    @classmethod
    def from_introspection_event(
        cls, event: ModelNodeIntrospectionEvent
    ) -> ModelRuntimeIdentity:
        """Take the identity fields from the runtime's own introspection data."""
        return cls(
            node_id=event.node_id,
            node_type=event.node_type,
            node_version=event.node_version,
            endpoints=dict(event.endpoints),
            node_role=event.node_role,
            network_id=event.network_id,
            deployment_id=event.deployment_id,
            event_bus=event.event_bus,
        )


__all__ = ["ModelRuntimeIdentity"]
