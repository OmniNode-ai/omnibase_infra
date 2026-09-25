# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Declared delegation retry ceilings."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from omnibase_infra.models.delegation.model_delegation_tier_retry_bound import (
    ModelDelegationTierRetryBound,
)


def _tier_sort_key(item: object) -> str:
    """Sort key that never raises; a non-string tier is refused by validation."""
    if isinstance(item, ModelDelegationTierRetryBound):
        return item.tier
    if isinstance(item, dict):
        tier = item.get("tier")
        return tier if isinstance(tier, str) else ""
    return ""


class ModelDelegationRetryBounds(BaseModel):
    """The distinct retry ceilings declared by routing and task contracts."""

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    per_tier: tuple[ModelDelegationTierRetryBound, ...] = Field(min_length=1)
    max_escalations: int = Field(ge=0)

    @field_validator("per_tier", mode="before")
    @classmethod
    def _sort_tier_bounds(cls, value: object) -> object:
        if isinstance(value, (list, tuple)):
            return tuple(sorted(value, key=_tier_sort_key))
        return value

    @model_validator(mode="after")
    def _validate_tier_retry_limits(self) -> ModelDelegationRetryBounds:
        tiers = [item.tier for item in self.per_tier]
        if len(tiers) != len(set(tiers)):
            raise ValueError("retry tier names must be unique")
        return self
