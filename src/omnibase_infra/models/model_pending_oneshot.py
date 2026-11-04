# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A particular execution whose completion deadline has not elapsed."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ModelPendingOneshot(BaseModel):
    """A particular execution whose completion deadline has not elapsed."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)
    lane: str
    container: str
    started_at: str
    deadline: str
    remaining_seconds: float
