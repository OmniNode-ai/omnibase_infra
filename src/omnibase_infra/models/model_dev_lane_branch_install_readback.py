# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One install's verdict across every file and service it names (OMN-17427)."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict

from omnibase_infra.enums.enum_dev_lane_branch_install_state import (
    EnumDevLaneBranchInstallState,
)
from omnibase_infra.models.model_dev_lane_branch_install_file_readback import (
    ModelDevLaneBranchInstallFileReadback,
)


class ModelDevLaneBranchInstallReadback(BaseModel):
    """Whether one declared install is whole in the running containers."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    install_key: str
    repo: str
    pr: int
    head: str
    ticket: str
    expires_at: datetime
    state: EnumDevLaneBranchInstallState
    files: tuple[ModelDevLaneBranchInstallFileReadback, ...]
