# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Plan override data model for RSD scoring."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ModelPlanOverrideData(BaseModel):
    """Fetched plan override for a ticket."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    ticket_id: str = Field(min_length=1)  # pattern-ok: Linear ticket IDs are strings
    override_score: float = Field(..., description="Override priority score (0-100).")
    previous_score: float = Field(default=50.0, description="Score before override.")
    reason: str = Field(default="", description="Override reason.")
    authorized_by: str = Field(default="unknown", description="Who authorized.")
    timestamp: datetime | None = Field(
        default=None, description="When override was created."
    )
    expires_at: datetime | None = Field(
        default=None, description="When override expires."
    )
    is_active: bool = Field(default=True, description="Whether override is active.")
