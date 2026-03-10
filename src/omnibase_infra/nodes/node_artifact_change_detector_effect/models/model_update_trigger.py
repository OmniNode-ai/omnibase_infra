# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Domain event model for artifact update triggers."""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelUpdateTrigger(BaseModel):
    """Represents an event that may require artifact updates.

    Emitted by the change detector EFFECT node when a PR event,
    contract change, or manual reconciliation request is received.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    trigger_id: UUID
    trigger_type: Literal[
        "pr_opened",
        "pr_updated",
        "pr_merged",
        "contract_changed",
        "schema_changed",
        "manual_plan_request",
    ]
    source_repo: str
    source_ref: str | None = None
    changed_files: list[str] = Field(default_factory=list)
    ticket_ids: list[str] = Field(default_factory=list)
    actor: str | None = None
    reason: str = ""
    timestamp: datetime
