# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Outcome of one entry-point census run."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.models.lane_entry_point_census.model_lane_entry_point_census import (
    ModelLaneEntryPointCensus,
)


class ModelLaneEntryPointCensusResult(BaseModel):
    """One census record per lane, and the markdown table regenerated from them."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    success: bool
    error_message: str = Field(default="")
    censuses: tuple[ModelLaneEntryPointCensus, ...] = Field(default=())
    table_markdown: str = Field(default="")


__all__: list[str] = ["ModelLaneEntryPointCensusResult"]
