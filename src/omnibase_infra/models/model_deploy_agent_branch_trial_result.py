# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Readback distinguishes a queued trial from code actually loaded and restored."""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class ModelDeployAgentBranchTrialResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    correlation_id: UUID
    requested_head: str
    status: Literal["queued", "running", "expired", "failed"]
    return_ref: Literal["origin/dev"] = "origin/dev"
    loaded_code_sha: str | None = None
    observed_at: datetime | None = None
    restored_loaded_code_sha: str | None = None
    installed_revision: str | None = None
    installer_observed: str | None = None
    error: str = ""
