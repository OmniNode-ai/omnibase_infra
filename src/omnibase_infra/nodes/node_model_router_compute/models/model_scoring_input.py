# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Complete input to the scoring algorithm."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.node_model_health_effect.models.model_endpoint_health import (
    ModelEndpointHealth,
)
from omnibase_infra.nodes.node_model_router_compute.models.enum_task_type import (
    EnumTaskType,
)
from omnibase_infra.nodes.node_model_router_compute.models.model_live_metrics import (
    ModelLiveMetrics,
)
from omnibase_infra.nodes.node_model_router_compute.models.model_registry_entry import (
    ModelRegistryEntry,
)
from omnibase_infra.nodes.node_model_router_compute.models.model_routing_constraints import (
    ModelRoutingConstraints,
)


class ModelScoringInput(BaseModel):
    """Complete input to the scoring algorithm — all data needed for pure computation."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    correlation_id: UUID = Field(..., description="Workflow correlation ID.")
    task_type: EnumTaskType = Field(..., description="Task type to route.")
    task_description: str = Field(default="", description="Human-readable description.")
    constraints: ModelRoutingConstraints = Field(..., description="Hard constraints.")
    context_length_estimate: int = Field(default=4096, description="Estimated tokens.")
    chain_hit: bool = Field(default=False, description="Chain retrieval hit.")
    chain_hit_model_key: str | None = Field(
        default=None, description="Model from matched chain."
    )
    registry: tuple[ModelRegistryEntry, ...] = Field(
        ..., description="All models from registry."
    )
    health: tuple[ModelEndpointHealth, ...] = Field(
        default_factory=tuple, description="Health snapshots per endpoint."
    )
    live_metrics: tuple[ModelLiveMetrics, ...] = Field(
        default_factory=tuple,
        description="Live performance metrics from reducer.",
    )
    scoring_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "quality": 0.4,
            "cost": 0.3,
            "speed": 0.2,
            "chain_bonus": 0.1,
        },
        description="Scoring formula weights.",
    )
