# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The record file of dev-lane branch builds (OMN-22996)."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict

from omnibase_infra.models.model_dev_lane_branch_build import ModelDevLaneBranchBuild

SCHEMA = "dev-lane-branch-builds.v1"


class ModelDevLaneBranchBuilds(BaseModel):
    """Every dev-lane build of an unmerged pull request's branch still on record."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: str = SCHEMA
    builds: tuple[ModelDevLaneBranchBuild, ...] = ()
    last_return_attempt: datetime | None = None
