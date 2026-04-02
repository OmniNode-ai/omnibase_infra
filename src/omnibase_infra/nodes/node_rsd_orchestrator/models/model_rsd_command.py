# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Command model for RSD orchestrator node."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.node_rsd_score_compute.models.model_rsd_factor_weights import (
    ModelRsdFactorWeights,
)


class ModelRsdCommand(BaseModel):
    """Command to initiate an RSD priority scoring cycle."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    correlation_id: UUID = Field(..., description="Workflow correlation ID.")
    ticket_ids: tuple[str, ...] = Field(..., description="Ticket IDs to score.")
    weights: ModelRsdFactorWeights = Field(
        default_factory=ModelRsdFactorWeights,
        description="Custom factor weights (defaults to standard RSD weights).",
    )
    include_dependencies: bool = Field(
        default=True, description="Whether to fetch dependency data."
    )
    include_agent_requests: bool = Field(
        default=True, description="Whether to fetch agent request data."
    )
    include_plan_overrides: bool = Field(
        default=True, description="Whether to fetch plan override data."
    )
