# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""One immutable retry ceiling declared for a routing tier."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ModelDelegationTierRetryBound(BaseModel):
    """One immutable retry ceiling declared for a routing tier."""

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    tier: str = Field(min_length=1)
    max_retries: int = Field(ge=0)

    @model_validator(mode="after")
    def _validate_tier(self) -> ModelDelegationTierRetryBound:
        if not self.tier.strip() or self.tier != self.tier.strip():
            raise ValueError("retry tier name must be nonblank and trimmed")
        return self
