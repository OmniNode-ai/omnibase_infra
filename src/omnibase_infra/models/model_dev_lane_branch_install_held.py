# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One registered install that is held, and why (OMN-26300)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.enums.enum_dev_lane_branch_install_hold import (
    EnumDevLaneBranchInstallHold,
)


class ModelDevLaneBranchInstallHeld(BaseModel):
    """An install that is not whole, with a reason that is never empty."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    code: EnumDevLaneBranchInstallHold
    reason: str = Field(min_length=1)
