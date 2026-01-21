# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Registry summary model for aggregate statistics.

Related Tickets:
    - OMN-1278: Contract-Driven Dashboard - Registry Discovery
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelRegistrySummary(BaseModel):
    """Summary statistics for the registry.

    Provides aggregate counts for dashboard widgets.

    Attributes:
        total_nodes: Total number of registered nodes
        active_nodes: Number of nodes in ACTIVE state
        healthy_instances: Number of passing health check instances
        unhealthy_instances: Number of failing health check instances
        by_node_type: Count of nodes by type
        by_state: Count of nodes by registration state
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    total_nodes: int = Field(
        ...,
        ge=0,
        description="Total number of registered nodes",
    )
    active_nodes: int = Field(
        ...,
        ge=0,
        description="Number of nodes in ACTIVE state",
    )
    healthy_instances: int = Field(
        ...,
        ge=0,
        description="Number of passing health check instances",
    )
    unhealthy_instances: int = Field(
        ...,
        ge=0,
        description="Number of failing health check instances",
    )
    by_node_type: dict[str, int] = Field(
        default_factory=dict,
        description="Count of nodes by type",
    )
    by_state: dict[str, int] = Field(
        default_factory=dict,
        description="Count of nodes by registration state",
    )


__all__ = ["ModelRegistrySummary"]
