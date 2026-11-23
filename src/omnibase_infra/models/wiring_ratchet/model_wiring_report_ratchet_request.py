# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Request of the wiring-report ratchet check (OMN-26335)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.models.wiring_ratchet.model_wiring_counts import ModelWiringCounts

__all__ = ["ModelWiringReportRatchetRequest"]


class ModelWiringReportRatchetRequest(BaseModel):
    """Observed counts, the baseline held in the contract, and the base ref's baseline."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    observed: ModelWiringCounts = Field(description="Counts read from the catalog.")
    baseline: ModelWiringCounts = Field(
        description="The baseline the node contract holds in the change."
    )
    base_baseline: ModelWiringCounts | None = Field(
        default=None,
        description=(
            "The baseline the node contract held at the base ref; None when the "
            "contract is new at the base."
        ),
    )
    scope: str = Field(
        min_length=1, description="Which catalog was read, for messages."
    )
