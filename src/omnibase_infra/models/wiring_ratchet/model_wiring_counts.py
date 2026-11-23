# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The four wiring-report counts the ratchet holds down (OMN-26335)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["ModelWiringCounts"]


class ModelWiringCounts(BaseModel):
    """Failed, duplicate and quarantined wiring plus subscriptions with no producer."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    failed: int = Field(ge=0, description="Contracts or handlers that cannot wire.")
    duplicates: int = Field(
        ge=0,
        description=(
            "Topics with more than one subscribing contract, plus contract names "
            "registered more than once."
        ),
    )
    quarantined: int = Field(
        ge=0, description="Handlers declared as a Protocol, which wiring quarantines."
    )
    unproduced_subscriptions: int = Field(
        ge=0, description="Subscribed topics that no contract publishes."
    )
