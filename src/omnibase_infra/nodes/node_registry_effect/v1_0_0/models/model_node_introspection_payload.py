# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Node introspection payload model for registry operations."""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from omnibase_core.models.node_metadata import ModelNodeCapabilitiesInfo
from pydantic import BaseModel, ConfigDict, Field

from .model_node_registration_metadata import ModelNodeRegistrationMetadata


class ModelNodeIntrospectionPayload(BaseModel):
    """Introspection data for node registration.

    Attributes:
        node_id: Unique identifier for the node being registered.
        node_type: ONEX node classification (effect, compute, reducer, orchestrator).
        node_version: Semantic version of the node (default: "1.0.0").
        capabilities: Structured node capabilities information including supported
            operations, dependencies, and performance metrics.
        endpoints: Map of endpoint names to URLs (e.g., {"health": "http://..."}).
        runtime_metadata: Runtime/deployment metadata including environment,
            tags, labels, release channel, and region information.
        health_endpoint: Optional dedicated health check URL for monitoring.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    node_id: UUID = Field(..., description="Unique node identifier")
    node_type: Literal["effect", "compute", "reducer", "orchestrator"]
    node_version: str = Field(default="1.0.0", description="Node version")
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
    health_endpoint: str | None = Field(None, description="Health check URL")
