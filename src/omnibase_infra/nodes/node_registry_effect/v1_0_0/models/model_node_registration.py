# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Node registration model for registry operations."""

from __future__ import annotations

from datetime import datetime

from omnibase_core.models.node_metadata import ModelNodeCapabilitiesInfo
from pydantic import BaseModel, ConfigDict, Field

from .model_node_registration_metadata import ModelNodeRegistrationMetadata


class ModelNodeRegistration(BaseModel):
    """Node registration record from storage.

    Represents a persisted node registration in PostgreSQL, containing
    introspection data along with registration timestamps.

    Attributes:
        node_id: Unique identifier for the registered node.
        node_type: ONEX node classification (effect, compute, reducer, orchestrator).
        node_version: Semantic version of the node.
        capabilities: Structured node capabilities information including supported
            operations, dependencies, and performance metrics.
        endpoints: Map of endpoint names to URLs.
        runtime_metadata: Runtime/deployment metadata including environment,
            tags, labels, release channel, and region information.
        health_endpoint: Optional dedicated health check URL.
        last_heartbeat: Timestamp of last heartbeat (if heartbeat enabled).
        registered_at: Timestamp when node was first registered.
        updated_at: Timestamp of last registration update.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    node_id: str
    node_type: str
    node_version: str = "1.0.0"
    capabilities: ModelNodeCapabilitiesInfo = Field(
        default_factory=ModelNodeCapabilitiesInfo,
        description="Structured node capabilities including operations and dependencies",
    )
    endpoints: dict[str, str] = Field(
        default_factory=dict,
        description="Map of endpoint names to URLs",
    )
    runtime_metadata: ModelNodeRegistrationMetadata = Field(
        ...,
        description="Runtime/deployment metadata (environment, tags, labels)",
    )
    health_endpoint: str | None = None
    last_heartbeat: datetime | None = None
    registered_at: datetime
    updated_at: datetime
