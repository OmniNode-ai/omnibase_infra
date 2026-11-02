# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Timings of one placement runtime, read from the node contract (OMN-22132)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ModelPlacementRuntimeSettings(BaseModel):
    """Lease and loop timings; the contract's profile block supplies every value."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    ttl_s: int = Field(gt=0)
    margin_s: float = Field(gt=0, allow_inf_nan=False)
    keepalive_interval_s: float = Field(gt=0, allow_inf_nan=False)
    campaign_interval_s: float = Field(gt=0, allow_inf_nan=False)
    fence_tick_s: float = Field(gt=0, allow_inf_nan=False)
    min_serving: int = Field(gt=0)

    @model_validator(mode="after")
    def _margin_below_ttl(self) -> ModelPlacementRuntimeSettings:
        if self.margin_s >= self.ttl_s:
            raise ValueError("margin_s must be below ttl_s")
        return self


__all__ = ["ModelPlacementRuntimeSettings"]
