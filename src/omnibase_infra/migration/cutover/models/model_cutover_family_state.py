# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Durable state projection for one cutover family."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.migration.cutover.enums import EnumCutoverFamilyStatus
from omnibase_infra.migration.cutover.models.model_cutover_family_contract import (
    ModelCutoverFamilyContract,
)

_SHA256_PATTERN = r"^[0-9a-f]{64}$"


class ModelCutoverFamilyState(BaseModel):
    """Represent the materialized family-local journal state."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    contract: ModelCutoverFamilyContract
    status: EnumCutoverFamilyStatus
    last_known_good_receipt_id: UUID | None = None
    blocked_receipt_id: UUID | None = None
    checkpoint_event_id: UUID | None = None
    first_target_write_event_id: UUID | None = None
    first_target_sequence: int | None = Field(default=None, ge=1)
    quiescence_event_id: UUID | None = None
    quiesced_target_sequence: int | None = Field(default=None, ge=1)
    verified_reverse_delta_proof_id: UUID | None = None
    dual_write_expires_at: datetime | None = None
    observation_ends_at: datetime | None = None
    last_event_at: datetime | None = None
    last_sequence: int = Field(..., ge=0)
    last_event_hash: str = Field(..., pattern=_SHA256_PATTERN)


__all__ = ["ModelCutoverFamilyState"]
