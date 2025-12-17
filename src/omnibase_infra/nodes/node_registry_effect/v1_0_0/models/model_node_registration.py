# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Node registration model for registry operations."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

# JSON-serializable value type for Pydantic models.
# We use str | int | float | bool | None | dict | list to represent JSON values
# without recursion (which Pydantic 2.x cannot handle in schema generation).
# This is a documented ONEX exception for JSON-value containers, as the full
# structure is validated at runtime by JSON serialization.
JsonValue = str | int | float | bool | None | dict | list


class ModelNodeRegistration(BaseModel):
    """Node registration record from storage.

    Represents a persisted node registration in PostgreSQL, containing
    introspection data along with registration timestamps.

    Attributes:
        node_id: Unique identifier for the registered node.
        node_type: ONEX node classification (effect, compute, reducer, orchestrator).
        node_version: Semantic version of the node.
        capabilities: Key-value pairs describing node capabilities.
        endpoints: Map of endpoint names to URLs.
        metadata: Additional node metadata (tags, labels, runtime info).
        health_endpoint: Optional dedicated health check URL.
        last_heartbeat: Timestamp of last heartbeat (if heartbeat enabled).
        registered_at: Timestamp when node was first registered.
        updated_at: Timestamp of last registration update.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    node_id: str
    node_type: str
    node_version: str = "1.0.0"
    capabilities: dict[str, JsonValue] = Field(
        default_factory=dict,
        description="Node capabilities as key-value pairs",
    )
    endpoints: dict[str, str] = Field(
        default_factory=dict,
        description="Map of endpoint names to URLs",
    )
    metadata: dict[str, JsonValue] = Field(
        default_factory=dict,
        description="Additional node metadata (tags, labels, runtime info)",
    )
    health_endpoint: str | None = None
    last_heartbeat: datetime | None = None
    registered_at: datetime
    updated_at: datetime
