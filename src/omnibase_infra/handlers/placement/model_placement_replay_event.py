# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One stored event a new primary folds when it rebuilds its projection (OMN-21421)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelPlacementReplayEvent(BaseModel):
    """An event at (partition, offset) of a node's replay topic."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    partition: int = Field(ge=0)
    offset: int = Field(ge=0)
    key: str = Field(min_length=1)
    value: str = ""


__all__ = ["ModelPlacementReplayEvent"]
