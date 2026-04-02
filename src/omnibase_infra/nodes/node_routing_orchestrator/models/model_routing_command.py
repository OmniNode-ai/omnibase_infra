# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Routing command model — initiates the routing workflow."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.node_model_router_compute.models.enum_task_type import (
    EnumTaskType,
)
from omnibase_infra.nodes.node_model_router_compute.models.model_routing_constraints import (
    ModelRoutingConstraints,
)


class ModelRoutingCommand(BaseModel):
    """Command to initiate an intelligent routing decision."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    correlation_id: UUID = Field(..., description="Workflow correlation ID.")
    task_description: str = Field(..., description="Human-readable task description.")
    task_type: EnumTaskType = Field(..., description="Classified task type.")
    constraints: ModelRoutingConstraints = Field(
        default_factory=ModelRoutingConstraints,
        description="Hard constraints for model selection.",
    )
    context_length_estimate: int = Field(
        default=4096, description="Estimated context length in tokens."
    )
    chain_hit: bool = Field(default=False, description="Chain retrieval hit.")
    chain_hit_model_key: str | None = Field(
        default=None, description="Model from matched chain."
    )
