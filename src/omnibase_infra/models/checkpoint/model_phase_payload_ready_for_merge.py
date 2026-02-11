# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Phase payload for the ready_for_merge phase.

Records the timestamp when the merge-ready label was applied.

Ticket: OMN-2143
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ModelPhasePayloadReadyForMerge(BaseModel):
    """Payload captured after the ready_for_merge phase completes."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    phase: Literal["ready_for_merge"] = Field(
        default="ready_for_merge",
        description="Discriminator field for phase payload union.",
    )
    label_applied_at: datetime = Field(
        ...,
        description="UTC timestamp when the merge-ready label was applied.",
    )


__all__: list[str] = ["ModelPhasePayloadReadyForMerge"]
