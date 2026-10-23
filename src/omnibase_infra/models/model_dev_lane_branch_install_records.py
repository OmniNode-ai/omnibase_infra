# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The record file of declared dev-lane branch installs (OMN-17427)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from omnibase_infra.models.model_dev_lane_branch_install import (
    ModelDevLaneBranchInstall,
)

SCHEMA = "dev-lane-branch-installs.v1"


class ModelDevLaneBranchInstallRecords(BaseModel):
    """Every install currently declared on the dev lane."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: str = SCHEMA
    installs: tuple[ModelDevLaneBranchInstall, ...] = ()
