# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Wait-while-busy decision for one deploy agent reading (OMN-21866)."""

from pydantic import BaseModel, ConfigDict, Field


class ModelLabLaneComposeBusyDecision(BaseModel):
    """The busy detail to wait on; empty when the lane may be graded now."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    busy_reason: str = Field(
        default="", description="Why the lane is mid compose deploy; empty when not."
    )


__all__ = ["ModelLabLaneComposeBusyDecision"]
