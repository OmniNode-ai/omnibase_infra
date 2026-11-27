# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""What the overdue dev merge build publish reads (OMN-26316)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.models.model_overdue_dev_merge_build_decision import (
    ModelOverdueDevMergeBuildDecision,
)


class ModelDevMergeBuildPublishRequest(BaseModel):
    """The decision, the merge's pull request, the omnibase_infra commit the build resets its clone to, and the lane.

    ``build_ref`` is always a commit of the agent's own repository, never the sibling merge's.
    ``runtime_lane`` is the lane the node's settings declare; it is carried as a plain string
    so the handler can refuse a value that is not ``dev``. ``already_requested`` is the caller's
    read of its own once-per-sha store. ``pr_number`` is ``None`` when the merge commit names
    no pull request.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    decision: ModelOverdueDevMergeBuildDecision
    repository: str = Field(min_length=1)
    pr_number: int | None = Field(default=None, ge=1)
    build_ref: str = Field(pattern=r"^[0-9a-f]{40}$")
    runtime_lane: str
    already_requested: bool = False
