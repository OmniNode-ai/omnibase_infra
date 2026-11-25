# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The event that states a dev runtime merge went its window with no build (OMN-26316)."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class ModelDevMergeBuildOverdueEvent(BaseModel):
    """One merge that waited past ``due_at`` and the request id its build request carries."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    repository: str
    merge_sha: str
    pr_number: int
    due_at: datetime
    request_id: UUID
    runtime_lane: str
