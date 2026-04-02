# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Result model for RSD data fetch effect node."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.node_rsd_data_fetch_effect.models.model_agent_request_data import (
    ModelAgentRequestData,
)
from omnibase_infra.nodes.node_rsd_data_fetch_effect.models.model_dependency_edge import (
    ModelDependencyEdge,
)
from omnibase_infra.nodes.node_rsd_data_fetch_effect.models.model_plan_override_data import (
    ModelPlanOverrideData,
)
from omnibase_infra.nodes.node_rsd_data_fetch_effect.models.model_ticket_data import (
    ModelTicketData,
)


class ModelRsdDataFetchResult(BaseModel):
    """Result of fetching all RSD-relevant data for scoring."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    correlation_id: UUID = Field(..., description="Workflow correlation ID.")
    tickets: tuple[ModelTicketData, ...] = Field(
        default_factory=tuple, description="Fetched ticket data."
    )
    dependency_edges: tuple[ModelDependencyEdge, ...] = Field(
        default_factory=tuple, description="Dependency graph edges."
    )
    agent_requests: tuple[ModelAgentRequestData, ...] = Field(
        default_factory=tuple, description="Agent requests per ticket."
    )
    plan_overrides: tuple[ModelPlanOverrideData, ...] = Field(
        default_factory=tuple, description="Plan overrides per ticket."
    )
    fetch_errors: tuple[str, ...] = Field(
        default_factory=tuple, description="Errors encountered during fetch."
    )
    success: bool = Field(default=True, description="Whether fetch succeeded.")
