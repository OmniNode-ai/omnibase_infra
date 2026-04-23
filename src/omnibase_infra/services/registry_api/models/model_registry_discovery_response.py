# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Registry discovery response model for dashboard payload.

Related Tickets:
    - OMN-1278: Contract-Driven Dashboard - Registry Discovery
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

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
        instance_discovery_status: Availability of legacy instance discovery
        instance_discovery_message: Optional explanation for degraded instance discovery
        live_instances: Compatibility list for legacy instance-oriented consumers
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
    instance_discovery_status: Literal["available", "unavailable"] = Field(
        default="unavailable",
        description="Availability of legacy instance discovery in the current runtime",
    )
    instance_discovery_message: str | None = Field(
        default=None,
        description="Optional explanation when legacy instance discovery is unavailable",
    )
    live_instances: list[ModelRegistryInstanceView] = Field(
        default_factory=list,
        description="Compatibility list for legacy instance-oriented consumers",
    )
    pagination: ModelPaginationInfo = Field(
        ...,
        description="Pagination info for nodes list",
    )


__all__ = ["ModelRegistryDiscoveryResponse"]
