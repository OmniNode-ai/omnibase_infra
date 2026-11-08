# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Outcome of one operation on the dev lane's record of branch builds (OMN-22996)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from omnibase_infra.models.model_dev_lane_branch_build import ModelDevLaneBranchBuild
from omnibase_infra.models.model_dev_lane_return_due import ModelDevLaneReturnDue


class ModelDevLaneReturnToDevResult(BaseModel):
    """The record file touched and the builds the operation concerns.

    ``recorded`` is every build now on record. ``due`` are the builds whose pull
    request has merged or closed (or whose record expired) while the lane still
    runs the branch, so the lane has to go back to dev's head. ``dropped`` are the
    builds the operation removed from the record: the ones whose lane already runs
    dev, or that a later successful dev job replaced. ``command`` is the one rebuild
    at the tracking ref the agent runs now, present only when something is due, the
    agent holds nothing back and no return started inside the last hour.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    operation: Literal["record", "settle"]
    path: Path
    recorded: tuple[ModelDevLaneBranchBuild, ...] = ()
    due: tuple[ModelDevLaneReturnDue, ...] = ()
    dropped: tuple[ModelDevLaneReturnDue, ...] = ()
    command: dict[str, str] | None = None
