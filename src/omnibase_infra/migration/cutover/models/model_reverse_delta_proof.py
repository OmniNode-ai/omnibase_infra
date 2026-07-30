# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Complete reverse-delta coverage for a quiesced writer."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from omnibase_infra.migration.cutover.models.model_reverse_delta_entry import (
    ModelReverseDeltaEntry,
)


class ModelReverseDeltaProof(BaseModel):
    """Require contiguous inverse coverage through the quiesced sequence."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    proof_id: UUID
    family_id: UUID
    start_sequence: int = Field(..., ge=1)
    end_sequence: int = Field(..., ge=1)
    entries: tuple[ModelReverseDeltaEntry, ...]
    quiescence_event_id: UUID
    reconciliation_receipt_id: UUID
    behavioral_readback_ref: str = Field(..., min_length=1, max_length=500)
    proven_at: datetime

    @model_validator(mode="after")
    def _coverage_is_contiguous(self) -> ModelReverseDeltaProof:
        if self.proven_at.tzinfo is None:
            raise ValueError("reverse-delta proof timestamp must be timezone-aware")
        if self.end_sequence < self.start_sequence:
            raise ValueError("reverse-delta end precedes start")
        expected = list(range(self.start_sequence, self.end_sequence + 1))
        actual = [entry.target_sequence for entry in self.entries]
        if actual != expected:
            raise ValueError("reverse-delta entries must cover every sequence in order")
        if any(entry.family_id != self.family_id for entry in self.entries):
            raise ValueError("reverse-delta entry belongs to another family")
        return self


__all__ = ["ModelReverseDeltaProof"]
