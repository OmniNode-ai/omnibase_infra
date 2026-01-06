# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""PostgreSQL intent payload model for registration orchestrator.

This module provides the typed payload model for PostgreSQL registration intents.

Design Note:
    This model uses strongly-typed ModelNodeCapabilities and ModelNodeMetadata
    instead of generic dict[str, JsonType] to adhere to ONEX "no Any types"
    principle. The JsonType type alias contains list[Any] and dict[str, Any]
    internally, which violates strict typing requirements.

    Using the same typed models as ModelNodeIntrospectionEvent ensures:
    1. Type safety throughout the registration pipeline
    2. Consistent validation between event source and database persistence
    3. No implicit Any types in the payload structure
"""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.models.registration.model_node_capabilities import (
    ModelNodeCapabilities,
)
from omnibase_infra.models.registration.model_node_metadata import ModelNodeMetadata


class ModelPostgresIntentPayload(BaseModel):
    """Payload for PostgreSQL registration intents.

    Contains the full node introspection data to upsert into the
    registration database. This is a typed representation of the
    data previously passed via model_dump().

    Uses strongly-typed capability and metadata models matching
    ModelNodeIntrospectionEvent for type-safe pipeline processing.

    Attributes:
        node_id: Unique node identifier.
        node_type: ONEX node type (effect, compute, reducer, orchestrator).
        node_version: Semantic version of the node.
        capabilities: Strongly-typed node capabilities.
        endpoints: Exposed endpoints (name -> URL).
        node_role: Optional role descriptor.
        metadata: Strongly-typed node metadata.
        correlation_id: Correlation ID for distributed tracing.
        network_id: Network/cluster identifier.
        deployment_id: Deployment/release identifier.
        epoch: Registration epoch for ordering.
        timestamp: Event timestamp as ISO string.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    node_id: UUID = Field(..., description="Unique node identifier")
    # Design Note: node_type uses Literal validation matching ModelNodeIntrospectionEvent
    # to ensure only valid ONEX node types can be persisted to PostgreSQL.
    node_type: Literal["effect", "compute", "reducer", "orchestrator"] = Field(
        ..., description="ONEX node type"
    )
    node_version: str = Field(default="1.0.0", description="Semantic version")
    capabilities: ModelNodeCapabilities = Field(
        default_factory=ModelNodeCapabilities,
        description="Strongly-typed node capabilities",
    )
    endpoints: dict[str, str] = Field(
        default_factory=dict, description="Exposed endpoints"
    )
    node_role: str | None = Field(default=None, description="Node role")
    metadata: ModelNodeMetadata = Field(
        default_factory=ModelNodeMetadata,
        description="Strongly-typed node metadata",
    )
    correlation_id: UUID = Field(..., description="Correlation ID for tracing")
    network_id: str | None = Field(default=None, description="Network identifier")
    deployment_id: str | None = Field(default=None, description="Deployment identifier")
    epoch: int | None = Field(default=None, ge=0, description="Registration epoch")
    timestamp: str = Field(..., description="Event timestamp as ISO string")


__all__ = [
    "ModelPostgresIntentPayload",
]
