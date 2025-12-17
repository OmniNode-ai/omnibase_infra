# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Registry request model for node registration operations."""

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.node_registry_effect.v1_0_0.models.model_node_introspection_payload import (
    ModelNodeIntrospectionPayload,
)


class ModelRegistryRequest(BaseModel):
    """Request model for registry operations."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    operation: Literal["register", "deregister", "discover", "request_introspection"]
    node_id: str | None = Field(None, description="Node ID for deregister/discover")
    introspection_event: ModelNodeIntrospectionPayload | None = Field(
        None, description="Introspection data for registration"
    )
    filters: dict[str, object] | None = Field(
        None, description="Filters for discover operation"
    )
    correlation_id: UUID = Field(..., description="Request correlation ID")
