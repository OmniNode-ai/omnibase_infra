# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""State model for RSD workflow reducer."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.node_rsd_score_compute.models.model_rsd_ticket_score import (
    ModelRsdTicketScore,
)
from omnibase_infra.nodes.node_rsd_state_reducer.models.model_rsd_rank_change import (
    ModelRsdRankChange,
)
from omnibase_infra.nodes.node_rsd_state_reducer.models.model_rsd_score_snapshot import (
    ModelRsdScoreSnapshot,
)


class ModelRsdState(BaseModel):
    """Complete RSD workflow state tracked by the reducer."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    correlation_id: UUID = Field(..., description="Workflow correlation ID.")
    workflow_state: str = Field(default="pending", description="Current FSM state.")
    current_scores: tuple[ModelRsdTicketScore, ...] = Field(
        default_factory=tuple, description="Most recent ticket scores."
    )
    current_ranked_ids: tuple[str, ...] = Field(
        default_factory=tuple, description="Current ranked order."
    )
    score_history: tuple[ModelRsdScoreSnapshot, ...] = Field(
        default_factory=tuple, description="Historical score snapshots."
    )
    rank_changes: tuple[ModelRsdRankChange, ...] = Field(
        default_factory=tuple, description="Rank changes from the last cycle."
    )
    total_cycles: int = Field(default=0, description="Total scoring cycles completed.")
    error_message: str = Field(
        default="", description="Error message if workflow failed."
    )
