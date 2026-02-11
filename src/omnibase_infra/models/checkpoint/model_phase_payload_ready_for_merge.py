# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Phase payload for the ready_for_merge phase.

Records the timestamp when the merge-ready label was applied.

Ticket: OMN-2143
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


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

    @field_validator("label_applied_at", mode="after")
    @classmethod
    def _ensure_utc(cls, v: datetime) -> datetime:
        """Reject naive datetimes; normalize to UTC."""
        if v.tzinfo is None:
            msg = "label_applied_at must be timezone-aware (use datetime.now(UTC))"
            raise ValueError(msg)
        return v.astimezone(UTC)


__all__: list[str] = ["ModelPhasePayloadReadyForMerge"]
