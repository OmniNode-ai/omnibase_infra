# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Pinned, bounded request delivered by the deploy agent's signed command bus."""

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelDeployAgentBranchTrialRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    operation: Literal["deploy_agent.branch_trial"] = "deploy_agent.branch_trial"
    correlation_id: UUID
    target: Literal["dev"]
    branch: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,199}$")
    head: str = Field(pattern=r"^[0-9a-f]{40}$")
    duration_seconds: int = Field(ge=1, le=3600)
    installer: Literal["image-prune"] | None = None
