# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One recorded branch build judged against its pull request (OMN-22996)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from omnibase_infra.enums.enum_dev_lane_return_reason import EnumDevLaneReturnReason
from omnibase_infra.models.model_dev_lane_branch_build import ModelDevLaneBranchBuild


class ModelDevLaneReturnDue(BaseModel):
    """A recorded build, the reason it is due and the evidence read for it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    build: ModelDevLaneBranchBuild
    reason: EnumDevLaneReturnReason
    detail: str
