# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Contracts ranked by measured loop time, the input of a shard split (OMN-26250)."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ModelContractCostShare(BaseModel):
    """One contract's summed measurements over the proposal's window."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    contract_name: str = Field(..., min_length=1, description="Attached contract.")
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


class ModelContractCostShardProposal(BaseModel):
    """Every read contract, heaviest loop time first, over the window read.

    The proposal is a sorted sum of stored readings and carries no value that
    was not read: the window is the earliest start and latest end of the
    readings, and ``top_10_share_of_loop_time`` is the share of the ten
    heaviest contracts. ``unread_contracts`` lists served contracts with no
    reading at all.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    window_start: datetime = Field(..., description="Earliest reading start (UTC).")
    window_end: datetime = Field(..., description="Latest reading end (UTC).")
    reading_count: int = Field(..., ge=1, description="Readings summed.")
    total_on_loop_wall_ms: float = Field(
        ..., ge=0.0, description="Loop time of every contract read."
    )
    top_10_share_of_loop_time: float = Field(
        ..., ge=0.0, le=1.0, description="Share of loop time of the ten heaviest."
    )
    contracts: tuple[ModelContractCostShare, ...] = Field(
        ..., description="Contracts sorted by loop time, heaviest first."
    )
    unread_contracts: tuple[str, ...] = Field(
        ..., description="Served contracts that have no reading, sorted."
    )

    @model_validator(mode="after")
    def _validate_proposal(self) -> ModelContractCostShardProposal:
        if self.window_end <= self.window_start:
            raise ValueError("window_end must be after window_start")
        loop_times = [c.on_loop_wall_ms for c in self.contracts]
        if loop_times != sorted(loop_times, reverse=True):
            raise ValueError("contracts must be sorted by loop time, heaviest first")
        return self


__all__: list[str] = ["ModelContractCostShardProposal", "ModelContractCostShare"]
