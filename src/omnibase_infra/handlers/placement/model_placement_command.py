# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A command issued by a placed node, stamped with its election epoch (OMN-21421)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelPlacementCommand(BaseModel):
    """Every command a primary issues carries the epoch it won with."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    node_key: str = Field(min_length=1)
    placement_epoch: int = Field(gt=0)
    runtime_key: str = Field(min_length=1)
    body: str = ""


__all__ = ["ModelPlacementCommand"]
