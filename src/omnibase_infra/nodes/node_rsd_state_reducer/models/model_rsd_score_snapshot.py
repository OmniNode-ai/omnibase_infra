# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Score snapshot model for RSD state tracking."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.node_rsd_score_compute.models.model_rsd_ticket_score import (
    ModelRsdTicketScore,
)


class ModelRsdScoreSnapshot(BaseModel):
    """A point-in-time snapshot of ticket scores for history tracking."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    calculated_at: datetime = Field(..., description="When scores were calculated.")
    ticket_scores: tuple[ModelRsdTicketScore, ...] = Field(
        ..., description="Scores at this point in time."
    )
    ranked_ticket_ids: tuple[str, ...] = Field(
        ..., description="Ranked order at this point in time."
    )
