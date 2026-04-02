# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Result model for RSD orchestrator node."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.node_rsd_score_compute.models.model_rsd_ticket_score import (
    ModelRsdTicketScore,
)
from omnibase_infra.nodes.node_rsd_state_reducer.models.model_rsd_rank_change import (
    ModelRsdRankChange,
)


class ModelRsdResult(BaseModel):
    """Final result of an RSD priority scoring cycle."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    correlation_id: UUID = Field(..., description="Workflow correlation ID.")
    ticket_scores: tuple[ModelRsdTicketScore, ...] = Field(
        ..., description="Final priority scores."
    )
    ranked_ticket_ids: tuple[str, ...] = Field(
        ..., description="Ticket IDs ordered by descending priority."
    )
    rank_changes: tuple[ModelRsdRankChange, ...] = Field(
        default_factory=tuple, description="Rank changes from previous cycle."
    )
    total_cycles: int = Field(default=1, description="Total scoring cycles completed.")
    success: bool = Field(default=True, description="Whether scoring succeeded.")
    error_message: str = Field(default="", description="Error message if failed.")
