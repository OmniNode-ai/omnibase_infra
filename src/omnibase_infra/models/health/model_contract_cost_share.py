# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One contract's share of loop time within a shard proposal (OMN-26250)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelContractCostShare(BaseModel):
    """One contract's summed measurements over the proposal's window."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    contract_label: str = Field(..., min_length=1, description="Attached contract.")
    on_loop_wall_ms: float = Field(..., ge=0.0, description="Summed loop time.")
    off_loop_wall_ms: float = Field(..., ge=0.0, description="Summed worker time.")
    dispatch_count: int = Field(..., ge=0, description="Summed dispatches.")
    payload_bytes: int = Field(..., ge=0, description="Summed message bytes.")
    share_of_loop_time: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="This contract's loop time over every contract's loop time.",
    )


__all__: list[str] = ["ModelContractCostShare"]
