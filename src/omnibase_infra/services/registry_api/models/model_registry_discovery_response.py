# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Registry discovery response model for dashboard payload.

Related Tickets:
    - OMN-1278: Contract-Driven Dashboard - Registry Discovery
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.services.registry_api.models.model_pagination_info import (
    ModelPaginationInfo,
)
from omnibase_infra.services.registry_api.models.model_registry_instance_view import (
    ModelRegistryInstanceView,
)
from omnibase_infra.services.registry_api.models.model_registry_node_view import (
    ModelRegistryNodeView,
)
from omnibase_infra.services.registry_api.models.model_registry_summary import (
    ModelRegistrySummary,
)
from omnibase_infra.services.registry_api.models.model_warning import ModelWarning


class ModelRegistryDiscoveryResponse(BaseModel):
    """Full dashboard payload combining nodes, instances, and summary.

    The primary response model for the GET /registry/discovery endpoint,
    providing everything a dashboard needs in a single request.

    Attributes:
        timestamp: When this response was generated
        warnings: List of warnings for partial success scenarios
        summary: Aggregate statistics
        nodes: List of registered nodes
        live_instances: List of live Consul instances
        pagination: Pagination info for nodes list
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    timestamp: datetime = Field(
        ...,
        description="When this response was generated",
    )
    warnings: list[ModelWarning] = Field(
        default_factory=list,
        description="Warnings for partial success scenarios",
    )
    summary: ModelRegistrySummary = Field(
        ...,
        description="Aggregate statistics",
    )
    nodes: list[ModelRegistryNodeView] = Field(
        default_factory=list,
        description="List of registered nodes",
    )
    live_instances: list[ModelRegistryInstanceView] = Field(
        default_factory=list,
        description="List of live Consul instances",
    )
    pagination: ModelPaginationInfo = Field(
        ...,
        description="Pagination info for nodes list",
    )


__all__ = ["ModelRegistryDiscoveryResponse"]
