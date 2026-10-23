# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""What one file of one install reads as in one container (OMN-17427)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from omnibase_infra.enums.enum_dev_lane_branch_install_file_state import (
    EnumDevLaneBranchInstallFileState,
)


class ModelDevLaneBranchInstallFileReadback(BaseModel):
    """The recorded and the observed sha256 of a file, and the state they give."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    service: str
    container: str
    path: str
    expected_sha256: str
    observed_sha256: str
    state: EnumDevLaneBranchInstallFileState
    detail: str = ""
