# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Node introspection payload model for registry operations."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# JSON-serializable value type for Pydantic models.
# We use str | int | float | bool | None | dict | list to represent JSON values
# without recursion (which Pydantic 2.x cannot handle in schema generation).
# This is a documented ONEX exception for JSON-value containers, as the full
# structure is validated at runtime by JSON serialization.
JsonValue = str | int | float | bool | None | dict | list


class ModelNodeIntrospectionPayload(BaseModel):
    """Introspection data for node registration.

    Attributes:
        node_id: Unique identifier for the node being registered.
        node_type: ONEX node classification (effect, compute, reducer, orchestrator).
        node_version: Semantic version of the node (default: "1.0.0").
        capabilities: Key-value pairs describing node capabilities (e.g., protocols,
            supported operations). Values can be primitives or nested structures.
        endpoints: Map of endpoint names to URLs (e.g., {"health": "http://..."}).
        metadata: Additional node metadata (e.g., tags, labels, runtime info).
        health_endpoint: Optional dedicated health check URL for monitoring.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    node_id: str = Field(..., description="Unique node identifier")
    node_type: Literal["effect", "compute", "reducer", "orchestrator"]
    node_version: str = Field(default="1.0.0", description="Node version")
    capabilities: dict[str, JsonValue] = Field(
        default_factory=dict,
        description="Node capabilities as key-value pairs (e.g., supported protocols)",
    )
    endpoints: dict[str, str] = Field(
        default_factory=dict,
        description="Map of endpoint names to URLs",
    )
    metadata: dict[str, JsonValue] = Field(
        default_factory=dict,
        description="Additional node metadata (tags, labels, runtime info)",
    )
    health_endpoint: str | None = Field(None, description="Health check URL")
