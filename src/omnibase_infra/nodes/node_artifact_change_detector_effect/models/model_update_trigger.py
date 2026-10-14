# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Domain event model for artifact update triggers."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.enums import EnumUpdateTriggerType


class ModelUpdateTrigger(BaseModel):
    """Represents an event that may require artifact updates.

    Emitted by the change detector EFFECT node when a PR event,
    contract change, or manual reconciliation request is received.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    trigger_id: UUID
    trigger_type: EnumUpdateTriggerType
    source_repo: str
    source_ref: str | None = None
    changed_files: list[str] = Field(default_factory=list)
    ticket_ids: list[str] = Field(default_factory=list)
    actor: str | None = None
    reason: str = ""
    timestamp: datetime
