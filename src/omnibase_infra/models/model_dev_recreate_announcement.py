# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The event published before a dev recreate (OMN-25912)."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.enums.enum_dev_recreate_reason import EnumDevRecreateReason


class ModelDevRecreateAnnouncement(BaseModel):
    """The event published before a recreate: the next recreate time and why."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    event_id: UUID
    runtime_lane: str = Field(min_length=1)
    requested_by: str = Field(min_length=1)
    announced_at: datetime
    next_recreate_at: datetime
    reason: EnumDevRecreateReason
    immediate: bool
