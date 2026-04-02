# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Configurable factor weights for the RSD 5-factor algorithm."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelRsdFactorWeights(BaseModel):
    """Configurable weights for the 5-factor RSD algorithm."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    dependency_distance: float = Field(
        default=0.40, description="Weight for dependency distance factor."
    )
    failure_surface: float = Field(
        default=0.25, description="Weight for failure surface factor."
    )
    time_decay: float = Field(default=0.15, description="Weight for time decay factor.")
    agent_utility: float = Field(
        default=0.10, description="Weight for agent utility factor."
    )
    user_weighting: float = Field(
        default=0.10, description="Weight for user weighting factor."
    )
