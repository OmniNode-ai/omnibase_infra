# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Response model for list_nodes endpoint.

Related Tickets:
    - OMN-1278: Contract-Driven Dashboard - Registry Discovery
    - PR #182: Add Pydantic response models for API endpoints
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.services.registry_api.models.model_pagination_info import (
    ModelPaginationInfo,
)
from omnibase_infra.services.registry_api.models.model_registry_node_view import (
    ModelRegistryNodeView,
)
from omnibase_infra.services.registry_api.models.model_warning import ModelWarning


class ModelResponseListNodes(BaseModel):
    """Response model for the GET /registry/nodes endpoint.

    Provides a paginated list of registered nodes with optional warnings
    for partial success scenarios.

    Attributes:
        nodes: List of registered nodes matching the query
        pagination: Pagination information for the result set
        warnings: List of warnings for partial success scenarios
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    nodes: list[ModelRegistryNodeView] = Field(
        default_factory=list,
        description="List of registered nodes matching the query",
    )
    pagination: ModelPaginationInfo = Field(
        ...,
        description="Pagination information for the result set",
    )
    warnings: list[ModelWarning] = Field(
        default_factory=list,
        description="Warnings for partial success scenarios",
    )


__all__ = ["ModelResponseListNodes"]
