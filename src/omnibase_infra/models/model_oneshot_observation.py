# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Only the fields needed to grade one run-to-completion execution."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ModelOneshotObservation(BaseModel):
    """Only the fields needed to grade one run-to-completion execution."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)
    completion_timeout_seconds: int = Field(gt=0)
    lane: str
    container: str
    keepalive: bool = False
    present: bool = True
    running: bool = False
    docker_status: str = ""
    started_at: str = ""
    observed_at: datetime | None = None
