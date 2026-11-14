# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The contracts boot leaves unwired."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["ModelBootWiringDeferralResult"]


class ModelBootWiringDeferralResult(BaseModel):
    """Names of the contracts none of whose topics is in use, sorted."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    deferred_contract_names: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Contracts whose handlers are not imported or registered at boot.",
    )
