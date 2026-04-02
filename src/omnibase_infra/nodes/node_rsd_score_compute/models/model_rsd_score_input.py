# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Input model for RSD score compute node."""

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
from omnibase_infra.nodes.node_rsd_score_compute.models.model_rsd_factor_weights import (
    ModelRsdFactorWeights,
)


class ModelRsdScoreInput(BaseModel):
    """Input containing all data needed for pure RSD priority scoring."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    correlation_id: UUID = Field(..., description="Workflow correlation ID.")
    tickets: tuple[ModelTicketData, ...] = Field(
        ..., description="Ticket data to score."
    )
    dependency_edges: tuple[ModelDependencyEdge, ...] = Field(
        default_factory=tuple, description="Dependency graph edges."
    )
    agent_requests: tuple[ModelAgentRequestData, ...] = Field(
        default_factory=tuple, description="Agent requests."
    )
    plan_overrides: tuple[ModelPlanOverrideData, ...] = Field(
        default_factory=tuple, description="Plan overrides."
    )
    weights: ModelRsdFactorWeights = Field(
        default_factory=ModelRsdFactorWeights,
        description="Factor weights for the 5-factor algorithm.",
    )
