# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Rank change model for RSD state tracking."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelRsdRankChange(BaseModel):
    """Tracks a rank change between two scoring cycles."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    ticket_id: str = Field(min_length=1)  # pattern-ok: Linear ticket IDs are strings
    previous_rank: int = Field(..., description="Previous rank position (1-based).")
    new_rank: int = Field(..., description="New rank position (1-based).")
    score_delta: float = Field(..., description="Change in score value.")
