# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One dev-lane branch install dropped, and why (OMN-17427)."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict

from omnibase_infra.enums.enum_dev_lane_branch_install_release_reason import (
    EnumDevLaneBranchInstallReleaseReason,
)


class ModelDevLaneBranchInstallRelease(BaseModel):
    """A line of the release history."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    install_key: str
    reason: EnumDevLaneBranchInstallReleaseReason
    detail: str
    released_at: datetime
