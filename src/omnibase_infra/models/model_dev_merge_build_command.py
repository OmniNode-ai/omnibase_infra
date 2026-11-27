# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The one signed build request an overdue dev merge sends (OMN-26316)."""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelDevMergeBuildCommand(BaseModel):
    """The fields of the signed build request path's typed request (``node_branch_deploy_command_effect``).

    ``correlation_id`` is the decision's request id, one value per repository and sha, so a
    second tick sends a duplicate the agent refuses. ``runtime_lane`` is ``dev``: the handler
    refuses any other value before this model is built. ``git_ref`` is the dev head sha,
    which carries the merge.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    correlation_id: UUID
    mode: Literal["deploy"] = "deploy"
    form: Literal["pr"] = "pr"
    runtime_lane: Literal["dev"] = "dev"
    repo: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    pr_number: int = Field(ge=1)
    git_ref: str = Field(pattern=r"^[0-9a-f]{40}$")
    rollback_actor: str = Field(min_length=1)
    rollback_reason: str = Field(min_length=1)
    requested_by: str = Field(min_length=1)
