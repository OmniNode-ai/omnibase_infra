# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Individual factor score model for RSD scoring."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelRsdFactorScore(BaseModel):
    """Individual factor score within an RSD calculation."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    factor_name: str = Field(
        min_length=1
    )  # pattern-ok: factor name is a label not an ID
    raw_score: float = Field(..., description="Raw score before weighting (0.0-1.0).")
    weight: float = Field(..., description="Weight applied to this factor.")
    weighted_score: float = Field(..., description="Score after weighting.")
