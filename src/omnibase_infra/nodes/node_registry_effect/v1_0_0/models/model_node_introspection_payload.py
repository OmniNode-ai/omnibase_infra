# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Node introspection payload model for registry operations."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ModelNodeIntrospectionPayload(BaseModel):
    """Introspection data for node registration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    node_id: str = Field(..., description="Unique node identifier")
    node_type: Literal["effect", "compute", "reducer", "orchestrator"]
    node_version: str = Field(default="1.0.0", description="Node version")
    capabilities: dict[str, object] = Field(default_factory=dict)
    endpoints: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, object] = Field(default_factory=dict)
    health_endpoint: str | None = Field(None, description="Health check URL")
