# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Control-plane snapshot and final-delta parity evidence."""

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

_SHA256_PATTERN = r"^[0-9a-f]{64}$"


class ModelControlPlaneDeltaEvidence(BaseModel):
    """Snapshot and final-delta parity for non-replayable control-plane truth."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    snapshot_id: UUID
    source_snapshot_hash: str = Field(..., pattern=_SHA256_PATTERN)
    target_snapshot_hash: str = Field(..., pattern=_SHA256_PATTERN)
    final_delta_id: UUID
    source_final_delta_hash: str = Field(..., pattern=_SHA256_PATTERN)
    target_final_delta_hash: str = Field(..., pattern=_SHA256_PATTERN)
    source_watermark: str = Field(..., min_length=1)
    target_watermark: str = Field(..., min_length=1)


__all__ = ["ModelControlPlaneDeltaEvidence"]
