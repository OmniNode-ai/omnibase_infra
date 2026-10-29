# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The ledger projection's record of one ``lane-work`` resource (OMN-21428)."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.models.lease.model_lane_work_holder import ModelLaneWorkHolder


class ModelLaneWorkClaim(BaseModel):
    """The highest-fence grant seen for a resource and whether its CLAIM is still open."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    lease_id: UUID
    fence: int = Field(ge=1)
    holder: ModelLaneWorkHolder
    claimed_at: str
    open: bool


__all__ = ["ModelLaneWorkClaim"]
